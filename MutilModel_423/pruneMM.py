# 这是提供参考示例的 YOLOMM 剪枝脚本
# 必须按照你的需求来更改，而不是盲目直接使用

import warnings
from pathlib import Path

from ultralytics import YOLOMM

# 屏蔽 albumentations 版本检查超时警告
warnings.filterwarnings("ignore", message=".*albumentations.*")
warnings.filterwarnings("ignore", message=".*fetch_version_info.*")


if __name__ == '__main__':
    # 正式剪枝时，建议直接加载已经训练完成的权重，而不是空白 YAML。
    # model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/YOLOMM-LST/weights/best.pt')
    model = YOLOMM('/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/ssa_prune_test/weights/best.pt')

    # 剪枝结果会在该目录下生成：
    # 1. pruned.pt          -> 后续继续训练/验证/推理应优先使用这个
    # 2. pruned.yaml        -> 剪枝后的参考结构说明，只反映层宽变化
    # 3. prune_report.json  -> 逐层结构化剪枝报告，用于看每层剪掉多少、保留了哪些索引
    # prune_save_dir = Path('ResTest/PruneExp/testt/yolomm-pruned-l1-r30')
    # prune_save_dir.mkdir(parents=True, exist_ok=True)

    model.prune(
        method='l1',      # 可选: 'l1', 'l2', 'lamp', 'bn', 'random'
        ratio=0.30,       # 剪枝比例，表示目标裁掉 30% 通道
        save_dir='./ResTest/ssa_prune_test/pruned',
        imgsz=640,        # 用于剪枝后前向验证和 GFLOPs 统计
        round_to=8,       # 保留通道数按 8 对齐，便于后续部署
        min_ch=8,         # 单层至少保留的输出通道数
        save_report=True, # 输出逐层结构化报告到 save_dir/prune_report.json
    )

    # 剪枝后如需验证精度，使用 val()：
    # model = YOLOMM(str(prune_save_dir / 'pruned.pt'))
    # model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml', device='0')
    #
    # 剪枝后如需微调训练，请使用 finetrainMM.py（model.finetrain()）
