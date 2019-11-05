# datafountain_siweituxin_autodriver_seg
competition url: https://www.datafountain.cn/competitions/366

## prerequirements:

+ python=3.6
+ pytorch==1.1.0
+ scipy==1.1.0
+ scikit-image


## for train
- download pretarined model to model/ 
- pretrained xception model url: https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi

```
cd experiment/deeplabv3+
python train.py
```

## for test
```
cd experiment/deeplabv3+
python test.py
```
# Note
for the detection part, please refer to my teammate's git repo: 
```
https://github.com/zhengye1995/datafountain_siweituxin_autodriver_det
```
