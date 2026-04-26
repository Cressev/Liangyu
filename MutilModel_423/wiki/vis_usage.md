# YOLOMM / RTDETRMM 可视化 vis 使用说明（简版）

本文仅说明 vis 的参数层级与用途，并提供双模态示例（YOLOMM 与 RTDETRMM）。

## 参数层级与用途
- 一级参数（vis 通用）
  - `rgb_source`：RGB 输入。仅支持图片“文件路径”或“目录路径”。目录模式下将按文件名去后缀（stem）与 `x_source` 目录一一配对；若只提供 RGB 目录则视为单侧输入（需结合 `modality` 做消融）。
  - `x_source`：X 模态输入。仅支持图片“文件路径”或“目录路径”。目录模式需与 `rgb_source` 同为目录并按同名文件配对；若只提供 X 目录则视为单侧输入（需结合 `modality` 做消融）。
  - `method`：可视化方法选择，取 `heat|heatmap` 或 `feature|feature_map`。
  - `layers`：待可视化的层索引列表（必填，整数，从 0 起）。
  - `modality`：`auto/dual/rgb/x`；双模态可强制消融为 `rgb` 或 `x`。
  - `save`：是否保存渲染结果到磁盘。
  - `project` / `name`：输出目录控制，推荐使用；`out_dir` 为兼容参数（已废弃）。
  - `device`：期望运行设备；与模型当前设备不一致将报错（不自动迁移）。

- 二级参数（方法专属，仅在对应方法中生效）
  - 热力图（`method='heat'|'heatmap'`）
    - `overlay`：热图叠加底图，`'rgb'|'x'|'dual'`。
    - `alg`：热力图算法（如 `'gradcam'`）。
    - `blend_alpha`：叠加透明度，默认 0.5（0~1）。
    - `colormap`：颜色映射，推荐 `'turbo'`（默认）或 `'viridis'`；也支持 `'inferno'|'magma'|'plasma'|'jet'` 等。
    - `layout`：输出布局，`'overlay'|'panel'|'both'`；默认 `'panel'`（原图｜热图｜叠加三联图，更适合人眼查看）。
    - `panel_scale`：当 `layout='panel'|'both'` 时生效，三联图整体缩放倍数（默认 1.0，可设 1.5/2.0 放大）。
    - `panel_title`：当 `layout='panel'|'both'` 时生效，是否在三联图顶部绘制标题（默认 True）。
    - `align_base`：双模态 letterbox 对齐基准（`'rgb'|'x'`，默认 `'rgb'`）。
    - `ablation_fill`：仅提供单侧输入时，另一侧消融填充方式（`'zeros'|'mean'`）。
    - `imgsz` / `vis_imgsz`：可视化前处理输入尺寸（整数）。提高该值可显著提升 CAM 空间分辨率与细腻度（代价是更慢/更占显存）。
    - `auto_imgsz`：当未显式指定 `imgsz/vis_imgsz` 时，是否根据原图大小自动上调可视化 imgsz（默认 True）。
    - `imgsz_cap`：`auto_imgsz=True` 时的上限（默认 1280）。原图再大也不会超过该值，以避免显存爆炸。
    - `cam_smooth`：是否对 CAM 上采样后做高斯平滑（默认 True），用于减少“块状”观感。
    - `cam_smooth_sigma`：平滑强度（默认 1.2，非负数；越大越“柔”和越细腻）。
    - `cam_resize_interp`：CAM 上采样插值方法（默认 `cubic`，可选 `nearest|linear|cubic|lanczos|area`）。
    - `export_components`：是否导出每层的素材子图（原图/热图/叠加/可选 panel），默认 True。
    - `export_panel`：`export_components=True` 时，是否额外导出各路三联图 panel（默认 True）。
    - `export_scale` / `save_scale`：导出素材整体缩放倍数（默认 1.0）。当原图较小但需要“论文级”大图时可设 2.0。
  - 特征图（`method='feature'|'feature_map'`）
    - `top_k`：按通道评分选取的可视化通道数（默认 8）。
    - `metric`：通道评分方式（`'sum'|'var'`）。
    - `normalize`：归一化方式（基础版 `'minmax'`）。
    - `colormap`：网格渲染色图（如 `'gray'|'jet'`）。
    - `align_base`：双模态对齐基准（`'rgb'|'x'`）。
    - `split`：是否额外导出单通道小图（布尔）。
    - `ablation_fill`：做消融时的填充值（`'zeros'|'mean'`）。

提示：若参数与输入条件不匹配（如缺少 `layers`、设备不一致、叠加底图与输入不符等），会直接报错；不做任何自动降级。

## 输出目录结构（热力图）

当 `save=True` 时，热力图会按“每层一个文件夹”导出，便于论文/报告自由拼图：

- `<project>/<name>/layer007/`
  - `heat_layer7_dual.png`（主输出：受 `layout` 影响）
  - `rgb/original.png`、`rgb/heatmap.png`、`rgb/overlay.png`（素材）
  - `x/original.png`、`x/heatmap.png`、`x/overlay.png`（素材）
  - `rgb/panel.png`、`x/panel.png`、`dual/panel.png`（当 `export_panel=True`）

目录模式（传两个目录）时，会额外按 `img_key` 再分一层子目录：`<project>/<name>/<img_key>/layer007/...`。

## 双模态示例（仅此项）

> 将示例路径替换为你本机的图片/权重路径。

### YOLOMM 热力图（双模态）
```python
from ultralytics import YOLOMM

model = YOLOMM('path/to/yolomm/best.pt')
model.vis(
    rgb_source='path/to/rgb.png',
    x_source='path/to/x.png',
    method='heat',
    layers=[7, 15, 18, 29],
    overlay='dual',       # 二级参数：默认双模态建议用 dual，一次看全两路
    alg='gradcam',        # 二级参数：热图算法
    colormap='turbo',     # 二级参数：默认 turbo，更自然
    layout='panel',       # 二级参数：三联图（原图｜热图｜叠加）
    panel_scale=1.2,      # 二级参数：放大输出，避免“看起来太小”
    save=True,
    project='runs/visualize/yolomm',
    name='exp_yolo_dual',
)
```

### RTDETRMM 热力图（双模态）
```python
from ultralytics import RTDETRMM

model = RTDETRMM('path/to/rtdetrmm/best.pt')
model.vis(
    rgb_source='path/to/rgb.png',
    x_source='path/to/x.png',
    method='heatmap',
    layers=[2, 4, 6, 18, 28],
    overlay='rgb',        # 二级参数：叠加在 RGB 底图
    alg='gradcam',        # 二级参数：热图算法
    save=True,
    project='runs/visualize/rtdetr',
    name='exp_rtdetr_dual',
)
```
