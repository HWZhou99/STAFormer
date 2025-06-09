# Spatio-Temporal Aggregation Transformer for Video-based Person Re-Identification

## Abstract

Temporal cues are crucial for video-based person re-identification (ReID). However, it remains a challenging problem to effectively model the temporal cues so that a representative and discriminative video-level feature can be extracted for person ReID. To address this problem, we propose Spatio-Temporal Aggregation Transformer (STAFormer) for the video-based person ReID. The proposed STAFormer is mainly composed of spatial and temporal aggregation modules, which establishes a unified framework for effective frame-level feature extraction as well as temporal cues aggregation with the attention mechanism. Specifically, the spatial aggregation module first extracts the frame-level features by adaptively fusing the local and global features with the attention mechanism. Similarly, the temporal aggregation module also leverages the attention mechanism to aggregate these frame-level features into the video-level representations. Extensive experiments on four widely used benchmark datasets demonstrate that the proposed STAFormer is effective for video-based person ReID with the ability to jointly extract frame-level features and aggregate temporal cues.

<p align="center">
  <img src ="https://github.com/HWZhou99/STAFormer/blob/main/STAFormer.png" alt="STAFormer",width="300">
</p>

## Requirements
Install all required dependencies with:
```
pip install -r requirements.txt
```
Python 3.7+ and PyTorch ≥1.8.0 are recommended.

## Datasets
All experiments in this project are conducted on the MARS dataset, which remains the largest benchmark for video-based person Re-ID to data. Please follow [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID) to prepare the data. The instructions are copied here:
1. Create a directory named ```mars/``` under ```data/```.
2. Download the MARS dataset from the following link and place it into the ```data/mars/``` directory: [MARS](https://drive.google.com/drive/folders/1N3SzngJ14tqkm0_b-vnGo93gMT2qhZNW?usp=drive_link)
3. Extract ```bbox_train.zip``` and ```bbox_test.zip``` into ```data/mars/```.
4. Download split information from [https://github.com/liangzheng06/MARS-evaluation/tree/master/info](https://github.com/liangzheng06/MARS-evaluation/tree/master/info) and put ```info/``` in ```data/mars```. Your folder structure should look like this:
```
data/
└── mars/
    ├── bbox_train/
    ├── bbox_test/
    └── info/
```
5. Use the parameter ```--dataset mars``` when training.

Other Video Person Re-ID datasets used in this project can be downloaded from the following links:

- [iLIDS-VID](https://www.eecs.qmul.ac.uk/~sgg/papers/WangEtAl_ECCV14.pdf)
- [DukeMTMC-VideoReID](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf)
- [LS-VID](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Global-Local_Temporal_Representations_for_Video_Person_Re-Identification_ICCV_2019_paper.pdf)

Please download and extract these datasets into the ```data/``` directory at the root of the project.

> **Note:** The LS-VID dataset requires access permission. Please refer to the official website of the dataset for detailed instructions.

  

## Get started
To train the model, please run:
```
python main.py \
   --arch make_model\
   --dataset ${mars, lsvid, ...} \
   --root ${path of dataset} \
   --gpu_devices 0,1 \
   --save_dir ${path for saving modles and logs}
```
Replace ```--root ${path of dataset}``` and ```--save_dir ${path for saving modles and logs}``` with your actual paths.

To test the model, please run:
```
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

| Dataset | mAP  | Rank-1 | model | 
| ------- | ---- | ------ | ------|
| MARS    | 89.1 |  91.5  |[mars](https://drive.google.com/file/d/1D274cxIb3sUMGYAwtGSQ8-3-BlJ13cku/view?usp=sharing)|
| LS-VID  | 85.6 |  90.9  |[ls-vid](https://drive.google.com/file/d/1JMEQ5fuTbsiKnkRLAtIIMK8b7inMR0uN/view?usp=drive_link)|

## Acknowledgments
Thanks for [SINet](https://github.com/baist/SINet) of Shutao Bai providing video reid code base.
