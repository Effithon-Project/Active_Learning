# jungyeon working directory


# Active Learning for EFFICIENT Learning
## Goals
1. SSD(ResNet50) + Kitti(Object detection) + Learning loss
    - base paper code [Github](github.com/Mephisto405/Learning-Loss-for-Active-Learning)
    - ssd code [Github](https://github.com/uvipen/SSD-pytorch)
    
    - issue
        - https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        
    ![plan](./img/plan.png)
2. new IDEA!!


---
# Table
- Papers & Articles
    - `OB`
    - `AL`
    - `+RL`
    - `+SSL`
- Code Reference
    - `KITTI`
    - `SSD`
    - `ETC`

---

## Papers & Articles
### OB
- [Deep Learning for Generic Object Detection: A Survey](https://arxiv.org/pdf/1809.02165v1.pdf)
- [Object Detection](https://github.com/hoya012/deep_learning_object_detection)
- [Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet and YOLOv3)](https://jonathan-hui.medium.com/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)
- [Object-Detection-Object-Detection-튜토리얼](https://rain-bow.tistory.com/entry/Object-Detection-Object-Detection-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC)

### AL
- []()

> if time permits

### +RL

- []()

### +SSL

- []()





## Reference

### KITTI
- [kitti vis code](https://github.com/bostondiditeam/kitti/blob/master/tools/2D_BBox.ipynb) : random_vis.py 코드
- [kitti label](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt) : 라벨링 정보
- [torchvision kitti](https://pytorch.org/vision/master/_modules/torchvision/datasets/kitti.html) : 중간에 계속 끊겨서 그냥 페이지에서 신청해서 다운 받음
- [kitti dataloader pytorch](github.com/dusty-nv/pytorch-depth/blob/master/dataloaders/kitti_dataloader.py) : 데이터 로더 참고

### SSD
- [nvidia_deeplearningexamples_ssd](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/) : ssd모델 그냥 불러오는 것 같은데 어떻게 수정할 수 있는지 모르겠음
- [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)

### ETC
- [argparse](https://m.blog.naver.com/cjh226/220997049388)