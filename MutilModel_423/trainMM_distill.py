#多模态蒸馏配置参考

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('yolo11n-mm-mid.yaml')
    model.train(
        data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
        epochs=10,
        batch=16,
        scale='n',
        device='0',
        cache=True,
        exist_ok=True,
        # 蒸馏配置
        distill=['/home/zhizi/work/multimodel/ultralyticmm/ultralyticsmm/distill.yaml', 'output'],#蒸馏配置 
        project='ResTest',
        name='DistillExp/dual-teacher-to-yolomm-n',
    )
