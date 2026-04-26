# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RTDETRMM：独立的多模态 RT-DETR 家族模型入口。

设计目标：
- 不继承 RTDETR；不依赖 ultralytics.models.rtdetr.*。
- 不依赖文件名是否包含 "-mm"；而是基于 YAML/CKPT 的内容判据严格 Fail-Fast。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ultralytics.engine.model import Model
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.val_results import COCOValResults, MMValResults

from .mm_predictor import RTDETRMMPredictor  # 新推理引擎适配器
from .train import RTDETRMMTrainer
from .val import RTDETRMMValidator
from .utils import require_multimodal
from ultralytics.nn.mm.utils import INPUT_SOURCE_ALIASES, resolve_input_source_role


class RTDETRMM(Model):
    """RT-DETR MultiModal（独立家族）。"""

    @staticmethod
    def _set_complexity_handshake(target: Any, *, pending=None, emitted=None, direct_ingress=None) -> None:
        """Persist lightweight owner/suppression hints via torch_utils constants only."""
        from ultralytics.utils.torch_utils import set_multimodal_complexity_owner_state
        # 只通过 torch_utils 的统一通道设置（使用 _mm_ 前缀的私有属性）
        # 不再向 kwargs dict 写入无前缀的 public key，避免 check_dict_alignment 报错
        if hasattr(target, '__dict__') and not isinstance(target, dict):
            set_multimodal_complexity_owner_state(
                target, pending=pending, emitted=emitted, direct_ingress=direct_ingress
            )

    def _log_canonical_complexity_summary(self, imgsz: int) -> bool:
        """Emit the canonical multimodal complexity summary from the model-level owner."""
        try:
            report = compute_default_multimodal_complexity_report(self.model, imgsz=imgsz)
            from ultralytics.utils import LOGGER as _LOGGER

            log_default_complexity(self.model, report, _LOGGER)
        except Exception:
            self._set_complexity_handshake(self.model, pending=False, emitted=False, direct_ingress=True)
            return False

        self._set_complexity_handshake(self.model, pending=False, emitted=True, direct_ingress=False)
        return True

    def __init__(
        self,
        model: Union[str, Path] = "rtdetr-r18-mm.pt",
        ch: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        # Fail-Fast：必须能从内容判据识别为多模态结构
        require_multimodal(model, who="RTDETRMM")

        self.input_channels = ch
        self.modality_config: Dict[str, Any] = {}
        self.is_multimodal = True

        # 基类会触发 _new/_load，依赖 task_map
        super().__init__(model=str(model), task="detect", verbose=verbose)

        # 解析 YAML 结构并填充 modality_config；确保 mm_router 可用
        self._configure_multimodal_settings(verbose=verbose)
        self._ensure_mm_router(verbose=verbose)

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": RTDETRMMPredictor,  # 新推理引擎适配器
                "validator": RTDETRMMValidator,
                "trainer": RTDETRMMTrainer,
                "model": RTDETRDetectionModel,
            }
        }

    def _configure_multimodal_settings(self, verbose: bool = False) -> None:
        """Detect routing markers from self.model.yaml, infer channels and modality set."""
        try:
            model_yaml = getattr(self.model, "yaml", None)
            if not isinstance(model_yaml, dict) or not model_yaml:
                self.input_channels = self.input_channels or 3
                self.modality_config = {
                    "primary_channels": list(range(self.input_channels or 3)),
                    "secondary_channels": [],
                    "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                    "default_modality": "RGB",
                }
                return

            has_mm = self._detect_multimodal_layers(model_yaml)
            model_channels = int(model_yaml.get("ch", model_yaml.get("channels", 3)) or 3)
            if has_mm and self._has_dual_modality_layers(model_yaml):
                model_channels = (model_channels if model_channels > 3 else 3) * 2
            elif has_mm:
                model_channels = model_channels if model_channels >= 3 else 3

            if self.input_channels is None:
                self.input_channels = model_channels
            self._validate_input_channels()

            pch = min(3, self.input_channels)
            sch = max(0, self.input_channels - pch)
            if has_mm and self._has_dual_modality_layers(model_yaml):
                self.modality_config = {
                    "primary_channels": list(range(pch)),
                    "secondary_channels": list(range(pch, pch + sch)),
                    "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                    "default_modality": "Dual",
                }
            elif has_mm:
                self.modality_config = {
                    "primary_channels": list(range(pch)),
                    "secondary_channels": list(range(pch, pch + sch)),
                    "supported_modalities": ["RGB", "X", "A", "B"],
                    "default_modality": "RGB",
                }
            else:
                self.modality_config = {
                    "primary_channels": list(range(pch)),
                    "secondary_channels": [],
                    "supported_modalities": ["RGB"],
                    "default_modality": "RGB",
                }

            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.info(
                    f"RTDETRMM configured: ch={self.input_channels}, "
                    f"modalities={self.modality_config.get('supported_modalities', [])}"
                )

        except Exception as e:
            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.warning(f"RTDETRMM: multimodal settings failed: {e}")
            self.input_channels = self.input_channels or 3
            self.modality_config = {
                "supported_modalities": list(INPUT_SOURCE_ALIASES.keys()),
                "default_modality": "RGB",
            }

    def _detect_multimodal_layers(self, model_yaml: dict) -> bool:
        all_layers = model_yaml.get("backbone", []) + model_yaml.get("head", [])
        for layer in all_layers:
            if isinstance(layer, list) and len(layer) >= 5 and resolve_input_source_role(layer[4]) is not None:
                return True
        return False

    def _has_dual_modality_layers(self, model_yaml: dict) -> bool:
        all_layers = model_yaml.get("backbone", []) + model_yaml.get("head", [])
        for layer in all_layers:
            if isinstance(layer, list) and len(layer) >= 5:
                role = resolve_input_source_role(layer[4])
                if role == "FUSED":
                    return True
        return False

    def _validate_input_channels(self) -> None:
        if self.input_channels is not None and self.input_channels < 1:
            raise ValueError(
                f"RTDETRMM: Invalid input channels: {self.input_channels}. Must be >= 1."
            )

    def _ensure_mm_router(self, verbose: bool = False) -> None:
        """确保底层 RTDETRDetectionModel 持有 mm_router。"""
        m = getattr(self, "model", None)
        if m is None:
            return

        # 直连 PyTorch 模型
        if hasattr(m, "mm_router") and m.mm_router is not None:
            return

        # AutoBackend(PyTorch)
        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is not None and hasattr(inner, "mm_router") and inner.mm_router is not None:
                return

        # 尝试补建
        try:
            from ultralytics.nn.mm import MultiModalRouter

            config_dict = getattr(m, "yaml", None)
            if hasattr(m, "mm_router"):
                m.mm_router = MultiModalRouter(config_dict, verbose=verbose)
            elif hasattr(m, "model") and getattr(m, "pt", False):
                inner = getattr(m, "model", None)
                if inner is not None and hasattr(inner, "mm_router"):
                    inner.mm_router = MultiModalRouter(config_dict, verbose=verbose)
        except Exception as e:
            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.warning(f"RTDETRMM: 无法创建 MultiModalRouter：{e}")

    def get_modality_info(self) -> Dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "modality_config": dict(self.modality_config),
            "model_type": "RTDETRMM",
            "task": getattr(self, "task", "detect"),
            "is_multimodal": True,
        }

    def vis(
        self,
        rgb_source: Optional[Union[str, np.ndarray, list]] = None,
        x_source: Optional[Union[str, np.ndarray, list]] = None,
        method: str = "heat",
        layers: Optional[list[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        project: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """可视化入口：委托给 RTDETRMMVisualizationRunner（独立家族）。"""
        from .visualize.runner import RTDETRMMVisualizationRunner

        return RTDETRMMVisualizationRunner.run(
            model=self.model,
            rgb_source=rgb_source,
            x_source=x_source,
            method=method,
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=str(project) if project is not None else None,
            name=name,
            out_dir=str(out_dir) if out_dir is not None else None,
            device=device,
            **kwargs,
        )

    def preflight(self, **kwargs):
        """
        Quick validation of RTDETRMM YAML config: verify it can complete a full training
        iteration (forward + loss + backward + optimizer step) using synthetic data.

        No real dataset required. Supports CLI mode: yolo preflight model=xxx.yaml

        Args:
            iters (int): Number of training iterations. Default 1.
            device (str | int): Device. Default 'cpu'.
            batch (int): Synthetic batch size. Default 2.
            imgsz (int): Image size. Default 640.
            scale (str): Model scale key (n/s/m/l/x). Requires YAML to define 'scales'.
                         Default: uses the first key in 'scales'.
            Xch (int): X modality channels. Default 3.
                        Also accepts primary_modality via dataset config for non-RGB primary modalities.
            verbose (bool): Verbose output. Default True.
            half (bool): Use FP16. Default False.

        Returns:
            PreflightReport: Report with pass/fail status per stage.
        """
        from ultralytics.engine.preflight import PreflightRunner

        model_path = ""
        if hasattr(self, "model") and hasattr(self.model, "yaml"):
            import tempfile
            import yaml as _yaml

            cfg_copy = dict(self.model.yaml) if isinstance(self.model.yaml, dict) else {}
            tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
            _yaml.dump(cfg_copy, tmp)
            tmp.close()
            model_path = tmp.name

        runner = PreflightRunner(model_path, **kwargs)
        return runner.run()

    def val(self, validator=None, coco: bool = False, **kwargs: Any):
        """
        多模态 RT-DETR 的验证入口：统一 `val(coco=...)`。

        关键点：
        - RT-DETR 的后处理默认按 `imgsz`（单一标量）缩放 bbox，天然假设输入为方形；
        - `Model.val()` 的通用默认值是 `rect=True`，会产生非方形 batch shape，从而导致 bbox 缩放错误、mAP 异常偏低；
        - 因此这里对 RTDETRMM 默认设置 `rect=False`（用户显式传入 rect 时尊重用户设置）。
        """
        kwargs.setdefault("rect", False)
        imgsz = int(kwargs.get("imgsz", self.overrides.get("imgsz", 640)))
        self._set_complexity_handshake(self.model, pending=True, emitted=False, direct_ingress=False)
        owner_emitted = self._log_canonical_complexity_summary(imgsz)
        self._set_complexity_handshake(
            kwargs,
            pending=not owner_emitted,
            emitted=owner_emitted,
            direct_ingress=not owner_emitted,
        )
        if not coco:
            standard = super().val(validator=validator, **kwargs)
            results_dict = getattr(standard, "results_dict", {}) or {}
            save_dir = getattr(standard, "save_dir", None)
            out = MMValResults(standard=standard, coco=None, results_dict=results_dict, save_dir=save_dir)
            self.metrics = out
            return out

        # COCO 验证器仅支持 PyTorch 模型
        self._check_is_pytorch_model()

        # COCO 验证默认参数（允许用户覆盖）；RT-DETR 仍保持默认 rect=False
        kwargs.setdefault("save_json", True)
        kwargs.setdefault("plots", True)
        kwargs.setdefault("conf", 0.05)
        kwargs.setdefault("mode", "cocoval")

        # 目录隔离：仅当用户未指定 save_dir/name 时写入默认 name
        if kwargs.get("save_dir", None) is None and kwargs.get("name", None) is None:
            kwargs["name"] = "val-coco"

        args = {**self.overrides, **kwargs, "mode": "cocoval"}
        self._set_complexity_handshake(
            args,
            pending=not owner_emitted,
            emitted=owner_emitted,
            direct_ingress=not owner_emitted,
        )

        from ultralytics.models.rtdetr.cocoval import RTDETRMMCOCOValidator

        validator = (validator or RTDETRMMCOCOValidator)(
            dataloader=None,
            save_dir=None,
            pbar=None,
            args=args,
            _callbacks=self.callbacks,
        )
        stats = validator(model=self.model)
        standard_metrics = getattr(validator, "metrics", None)
        if standard_metrics is None:
            raise RuntimeError("RTDETRMM.val(coco=True): validator 未生成 metrics，无法构建 MMValResults。")

        save_dir = getattr(standard_metrics, "save_dir", None)
        coco_obj = COCOValResults.from_stats_dict(stats, save_dir=save_dir)
        out = MMValResults(standard=standard_metrics, coco=coco_obj, results_dict=stats, save_dir=save_dir)
        self.metrics = out
        return out

    def cocoval(self, validator=None, **kwargs: Any):
        """
        使用 COCO 评估指标对 RTDETRMM 进行验证。

        说明：
        - 返回包含 `metrics/coco/*` 的指标字典（12 项 COCO 标准指标为主）；
        - 默认 `rect=False`，避免 RT-DETR 在非方形输入下因 bbox 缩放假设导致指标异常（用户显式传入 rect 时尊重用户设置）。
        """
        res = self.val(validator=validator, coco=True, **kwargs)
        results = res.results_dict
        # 兼容旧行为：cocoval 结束后 self.metrics 为 dict
        self.metrics = results
        return results
