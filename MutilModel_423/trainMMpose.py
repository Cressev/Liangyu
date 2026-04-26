# YOLOMM Pose 训练脚本参考示例
# 请根据实际需求修改配置参数

from ultralytics import YOLOMM

if __name__ == '__main__':
    # 加载多模态姿态估计模型配置
    model = YOLOMM('yolo11n-mm-mid-pose.yaml')

    # 训练配置
    model.train(
        data='/home/zhizi/work/multimodel/ultralyticmm/datasets/tiger/data.yaml',
        epochs=2,
        batch=16,
        imgsz=640,
        scale='n',  # 模型规模: n/s/m/l/x
        cache=True,
        exist_ok=True,
        project='ResTest',
        name='yolomm-pose-tiger',
        # modality='edge',  # 模态消融参数，非必要不开启
    )
