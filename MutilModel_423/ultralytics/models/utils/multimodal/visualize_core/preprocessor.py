"""Preprocessor for visualize_core.

Provides unified letterbox and normalization for single or dual modality inputs.
No pseudocolor is applied to X modality per project requirement; X is kept in its
original channel count and numeric range before normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import cv2


class Preprocessor:
    @staticmethod
    def passthrough(x: Any) -> Any:
        return x

    @staticmethod
    def model_input_channels(model: Any, default: int = 3) -> int:
        """
        Infer expected input channel count for visualization preprocessing.

        Notes:
        - Prefer multimodal router's Dual channel config (3+Xch) when available, because some
          multimodal architectures (e.g., mm-mid) keep the first Conv in_channels=3 but still
          accept a Dual input tensor that is routed internally by the router.
        - Fallback to the first Conv2d.in_channels for non-multimodal models.
        """

        # 1) Prefer multimodal router config (Dual = 3 + Xch)
        router = None
        for chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for a in chain:
                if hasattr(m, a):
                    m = getattr(m, a)
                else:
                    ok = False
                    break
            if ok and m is not None:
                router = m
                break

        if router is not None:
            try:
                sources = getattr(router, "INPUT_SOURCES", None)
                if isinstance(sources, dict) and "Dual" in sources:
                    return int(sources["Dual"])
            except Exception:
                pass

        # 2) Fallback: first Conv2d.in_channels (typical for standard YOLO)
        try:
            import torch.nn as nn  # local import to keep module lightweight

            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    return int(m.in_channels)

            # Generic fallback: any module exposing in_channels/out_channels
            for m in model.modules():
                if hasattr(m, "in_channels") and hasattr(m, "out_channels"):
                    ic = int(getattr(m, "in_channels"))
                    if ic > 0:
                        return ic
        except Exception:
            pass

        return int(default)

    @staticmethod
    def model_input_size(model: Any, default: int = 640) -> int:
        try:
            if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
                return int(model.args.imgsz)
        except Exception:
            pass
        return int(default)

    @staticmethod
    def _letterbox(im: np.ndarray, new_shape: Union[int, Tuple[int, int]] = 640, color=(114, 114, 114), stride: int = 32) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int, int, int]]:
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (top, bottom, left, right)

    @staticmethod
    def _to_float01(img: np.ndarray) -> np.ndarray:
        """Convert image to float32 [0,1].

        - uint8: normalize via /255
        - uint16: normalize via /65535
        - float32 already in [0,1]: passthrough
        - float64: cast to float32
        - integer types (int64/int32/int16): cast + normalize
        - Other dtypes: REJECTED
        """
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        if img.dtype == np.float32:
            return img
        if img.dtype == np.float64:
            return img.astype(np.float32)
        if np.issubdtype(img.dtype, np.integer):
            img_f = img.astype(np.float32)
            max_val = img_f.max()
            if max_val > 1.0:
                img_f = img_f / max_val if max_val > 0 else img_f
            return img_f
        raise ValueError(
            f"可视化预处理器收到不支持的dtype={img.dtype}。"
            f"支持: uint8, uint16, float32, float64, integer types。"
        )

    @staticmethod
    def letterbox_single(image: np.ndarray, size: int) -> np.ndarray:
        lb, _, _ = Preprocessor._letterbox(image, new_shape=size)
        return Preprocessor._to_float01(lb)

    @staticmethod
    def letterbox_dual(primary: np.ndarray, secondary: np.ndarray, size: int) -> np.ndarray:
        # First image defines padding for consistency
        primary_lb, _, padding = Preprocessor._letterbox(primary, new_shape=size)
        secondary_lb, _, _ = Preprocessor._letterbox(secondary, new_shape=size)
        # Normalize
        primary_f = Preprocessor._to_float01(primary_lb)
        secondary_f = Preprocessor._to_float01(secondary_lb)
        # Ensure channel dims
        if primary_f.ndim == 2:
            primary_f = primary_f[:, :, None]
        if secondary_f.ndim == 2:
            secondary_f = secondary_f[:, :, None]
        # Concatenate channels [Primary(3), Secondary(Xch)] without pseudocolor
        return np.concatenate([primary_f, secondary_f], axis=2)

    @staticmethod
    def letterbox_dual_aligned(
        primary: np.ndarray,
        secondary: np.ndarray,
        size: int,
        align_base: str = "primary",
        stride: int = 32,
    ) -> np.ndarray:
        """
        Letterbox two modalities with the SAME ratio/padding derived from the base image.

        This enforces pixel-wise alignment on the final canvas by reusing the scaling ratio and
        padding computed from the base modality (primary or secondary) for the other modality.

        Args:
            primary: Primary (RGB) image (HWC)
            secondary: Secondary (X) modality image (HWC or HW or HWC with 1+ channels)
            size: square target size (int)
            align_base: 'primary' or 'secondary' indicating which modality defines ratio/padding
            stride: model stride for padding alignment (default: 32)

        Returns:
            Concatenated HWC float32 array with channels [Primary(3), Secondary(Xch)] in [0,1].
        """
        base = primary if str(align_base).lower() in ("primary", "rgb") else secondary

        # Compute base letterbox once to obtain target padding
        base_lb, ratio, pad = Preprocessor._letterbox(base, new_shape=size, stride=stride)
        r = ratio[0]
        top, bottom, left, right = pad

        def _apply_with_ratio(im: np.ndarray) -> np.ndarray:
            shape = im.shape[:2]
            # Resize with base ratio r
            new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
            if shape[::-1] != new_unpad:
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            # Apply the same padding tuple as base
            return cv2.copyMakeBorder(
                im,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )

        if base is primary:
            primary_lb = base_lb
            secondary_lb = _apply_with_ratio(secondary)
        else:
            secondary_lb = base_lb
            primary_lb = _apply_with_ratio(primary)

        # Normalize to [0,1]
        primary_f = Preprocessor._to_float01(primary_lb)
        secondary_f = Preprocessor._to_float01(secondary_lb)

        # Ensure explicit channel dims
        if primary_f.ndim == 2:
            primary_f = primary_f[:, :, None]
        if secondary_f.ndim == 2:
            secondary_f = secondary_f[:, :, None]

        return np.concatenate([primary_f, secondary_f], axis=2)

    @staticmethod
    def prepare_inputs(inputs: Dict[str, np.ndarray], model: Any) -> np.ndarray:
        size = Preprocessor.model_input_size(model)
        # Support both new keys (PRIMARY/SECONDARY) and legacy keys (rgb/x)
        primary_key = 'PRIMARY' if 'PRIMARY' in inputs else ('rgb' if 'rgb' in inputs else None)
        secondary_key = 'SECONDARY' if 'SECONDARY' in inputs else ('x' if 'x' in inputs else None)
        if primary_key and secondary_key:
            return Preprocessor.letterbox_dual(inputs[primary_key], inputs[secondary_key], size)
        if primary_key:
            return Preprocessor.letterbox_single(inputs[primary_key], size)
        # Fallback: use first available value
        if secondary_key:
            return Preprocessor.letterbox_single(inputs[secondary_key], size)
        first_val = next(iter(inputs.values()))
        return Preprocessor.letterbox_single(first_val, size)
