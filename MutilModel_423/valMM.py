from ultralytics import YOLOMM

model = YOLOMM('/mnt/SSD/work/multimodel/ultralyticmm/ultralyticsmm/ResTest/rgbt-tiny-26n-mid/weights/best.pt')
model.val(data='/home/zhizi/work/multimodel/ultralyticmm/data.yaml',
          split='val',device='0',
        #   modality='x',模态消融参数 非必要不得开启
          coco=True,# 打开之后输出为 coco 指标，COCO 指标和 YOLO 指标大概率不一致
          project='ResTest',
          name='rgbt-tiny-26n-mid-val')
