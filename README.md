# Spatio-Temporal Aggregation Transformer for Video-based Person Re-Identification

## Abstract

Temporal cues are crucial to the video-based person re-identification (ReID). However, it remains a challenging task to model the temporal cues so as to extract representative and discriminative video-level features for the person ReID. To address this problem, a Spatio-Temporal Aggregation Transformer (STAFormer) is proposed for the video-based person ReID in this paper. The proposed STAFormer is mainly composed of spatial and temporal aggregation modules, which allows for a unified framework for effective frame-level feature extraction as well as temporal cues aggregation with an attention mechanism. Specifically, the spatial aggregation module first extracts the frame-level features by adaptively fusing the local and global features
with an attention mechanism. Similarly, the temporal aggregation module also leverages the attention mechanism to aggregate these frame-level features into the video-level representations. Extensive experiments on four widely used benchmark datasets demonstrate that the proposed STAFormer is effective for video-based person ReID with the ability to jointly frame-level feature extraction and temporal cues aggregation.

<p align="center">
  <img src ="https://github.com/HWZhou99/STAFormer/blob/main/STAFormer.jpg" alt="STAFormer",width="300">
</p>

## Datasets
Dataset preparation instructions can be found in the repository [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID)


## Get started
```
#Train
python main.py \
   --arch make_model\
   --dataset ${mars, lsvid, ...} \
   --root ${path of dataset} \
   --gpu_devices 0,1 \
   --save_dir ${path for saving modles and logs}

#Test with all frames
  python main.py \
   --arch make_model \
   --dataset mars \
   --root ${path of dataset} \
   --gpu_devices 0,1 \
   --save_dir ${path for saving logs} \
   --evaluate --all_frames --resume ${path of pretrained model}
```

## Result

| Dataset | MARS | LS-VID |     
| ------- | ---- | ------ |
| mAP     | 89.1 | 85.6   |
| Rank-1  | 91.5 | 90.9   |

## Acknowledgments
This code is based on the implementations of [SINet]([https://github.com/jiyanggao/Video-Person-ReID])
