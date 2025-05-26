# Spatio-Temporal Aggregation Transformer for Video-based Person Re-Identification

## Abstract

Temporal cues are crucial to the video-based person re-identification (ReID). However, it remains a challenging task to model the temporal cues so as to extract representative and discriminative video-level features for the person ReID. To address this problem, a Spatio-Temporal Aggregation Transformer (STAFormer) is proposed for the video-based person ReID in this paper. The proposed STAFormer is mainly composed of spatial and temporal aggregation modules, which allows for a unified framework for effective frame-level feature extraction as well as temporal cues aggregation with an attention mechanism. Specifically, the spatial aggregation module first extracts the frame-level features by adaptively fusing the local and global features
with an attention mechanism. Similarly, the temporal aggregation module also leverages the attention mechanism to aggregate these frame-level features into the video-level representations. Extensive experiments on four widely used benchmark datasets demonstrate that the proposed STAFormer is effective for video-based person ReID with the ability to jointly frame-level feature extraction and temporal cues aggregation.

![STAFormer框架图](D:\OneDrive\Desktop\STAFormer框架图.jpg)