#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('rtdetr-l.yaml')
    model.train(data="/root/autodl-tmp/MutilModel_423/dataset/underwater/data_rgb.yaml",
                epochs=350,batch=36,
                # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
                # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
                # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
                amp=False,  # 关闭AMP混合精度
                device='0',
                verbose=True,
                # cache='auto',
                exist_ok=True,
                patience=50,
                # afss_enabled=True, #启动afss训练采样机制
                project='run',name='rtdetr-l')
