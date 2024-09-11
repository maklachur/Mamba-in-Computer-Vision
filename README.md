## Mamba in Computer Vision Applications

- [General Purpose](#general-purpose)
- [Image Classification, Object Detection, and Segmentation](#image-classification-object-detection-and-segmentation)
- [Image Enhancement](#image-enhancement)
- [Generation and Restoration](#generation-and-restoration)
- [Point Cloud Analysis](#point-cloud-analysis)
- [Video Processing](#video-processing)
- [Remote Sensing](#remote-sensing)
- [Medical Image Analysis](#medical-image-analysis)
  - [Medical Image Classification](#medical-image-classification)
  - [Medical Image Segmentation](#medical-image-segmentation)
    - [Medical 2D Image Segmentation](#medical-2d-image-segmentation)
    - [Medical 3D Image Segmentation](#medical-3d-image-segmentation)
  - [Medical Image Reconstruction](#medical-image-reconstruction)
  - [Other Tasks in Medical Imaging](#other-tasks-in-medical-imaging)
- [Multimodal](#multimodal)
- [Other Tasks](#other-tasks)

## General Purpose

- **FractalVMamba: Scalable Visual State Space Model with Fractal Scanning** - May 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.14480)]
- **MSVMamba: Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model** - May 23, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.14174)] [[Code](https://github.com/YuHengsss/MSVMamba)] ![Stars](https://img.shields.io/github/stars/YuHengsss/MSVMamba)
- **Mamba-R: Vision Mamba ALSO Needs Registers** - May 23, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.14858)] [[Homepage](https://wangf3014.github.io/mambar-page/)] [[Code](https://github.com/wangf3014/Mamba-Reg)] ![Stars](https://img.shields.io/github/stars/wangf3014/Mamba-Reg)
- **Vim-F: Visual State Space Model Benefiting from Learning in the Frequency Domain** - May 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.18679)] [[Code](https://github.com/yws-wxs/Vim-F)] ![Stars](https://img.shields.io/github/stars/yws-wxs/Vim-F)
- **SUM: Saliency Unification through Mamba for Visual Attention Modeling** - April 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.17815)] [[Code](https://github.com/Arhosseini77/SUM)] ![Stars](https://img.shields.io/github/stars/Arhosseini77/SUM)
- **SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time Series** - April 24, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.15360)] [[Code](https://github.com/badripatro/Simba)] ![Stars](https://img.shields.io/github/stars/badripatro/Simba)
- **MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection** - Mar 29, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.19888)] [[Code](https://github.com/MambaMixer/M2)] ![Stars](https://img.shields.io/github/stars/MambaMixer/M2)
- **Heracles: A Hybrid SSM-Transformer Model for High-Resolution Image and Time-Series Analysis** - Mar 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.18063)] [[Code](https://github.com/badripatro/heracles)] ![Stars](https://img.shields.io/github/stars/badripatro/heracles)
- **PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition** - March 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.17695)] [[Code](https://github.com/ChenhongyiYang/PlainMamba)] ![Stars](https://img.shields.io/github/stars/ChenhongyiYang/PlainMamba)
- **EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba** - March 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.09977)] [[Code](https://github.com/TerryPei/EfficientVMamba)] ![Stars](https://img.shields.io/github/stars/TerryPei/EfficientVMamba)
- **LocalMamba: Visual State Space Model with Windowed Selective Scan** - March 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.09338)] [[Code](https://github.com/hunto/LocalMamba)] ![Stars](https://img.shields.io/github/stars/hunto/LocalMamba)
- **Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data** - February 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.05892)] [[Code](https://github.com/jacklishufan/Mamba-ND)] ![Stars](https://img.shields.io/github/stars/jacklishufan/Mamba-ND)
- **VMamba: Visual State Space Model** - January 19, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.10166)] [[Code](https://github.com/MzeroMiko/VMamba)] ![Stars](https://img.shields.io/github/stars/MzeroMiko/VMamba)
- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** - January 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.09417)] [[Code](https://github.com/hustvl/Vim)] ![Stars](https://img.shields.io/github/stars/hustvl/Vim)

## Image Classification, Object Detection, and Segmentation (General)

### Image Classification
- **Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning** - April 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.15761)] [[Code](https://github.com/ChiShengChen/ResVMamba)] ![Stars](https://img.shields.io/github/stars/ChiShengChen/ResVMamba)
- **InsectMamba: Insect Pest Classification with State Space Model** - April 4, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.03611)]

### Object Detection
- **RWKV-SAM: Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model** - June 27, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.19369)] [[Code](https://github.com/HarborYuan/ovsam)] ![Stars](https://img.shields.io/github/stars/HarborYuan/ovsam)
- **Voxel Mamba: Group-Free State Space Models for Point Cloud-based 3D Object Detection** - June 18, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.10700)] [[Code](https://github.com/gwenzhang/Voxel-Mamba)] ![Stars](https://img.shields.io/github/stars/gwenzhang/Voxel-Mamba)
- **Mamba-YOLO: SSMs-Based YOLO For Object Detection** - June 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.05835)] [[Code](https://github.com/HZAI-ZJNU/Mamba-YOLO)] ![Stars](https://img.shields.io/github/stars/HZAI-ZJNU/Mamba-YOLO)
- **SOAR: Advancements in Small Body Object Detection for Aerial Imagery Using State Space Models and Programmable Gradients** - May 5, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.01699)] [[Code](https://github.com/yash2629/S.O.A.R)] ![Stars](https://img.shields.io/github/stars/yash2629/S.O.A.R)

### Segmentation
- **Chen et al. (2024): Segmentation Model with Enhanced Mamba Scanning Strategies** - July 24, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.07865)]
- **HTD-Mamba: Efficient Hyperspectral Target Detection with Pyramid State Space Model** - July 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.06841)] [[Code](https://github.com/shendb2022/HTD-Mamba)] ![Stars](https://img.shields.io/github/stars/shendb2022/HTD-Mamba)
- **Fusion-Mamba: Efficient Image Fusion with State Space Model** - April 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.07932)]
- **MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection** - March 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.02148)] [[Code](https://github.com/txchen-USTC/MiM-ISTD)] ![Stars](https://img.shields.io/github/stars/txchen-USTC/MiM-ISTD)

## Image Enhancement

- **BVI-RLV: A Fully Registered Dataset and Benchmarks for Low-Light Video Enhancement** - July 3, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.03535)] [[Code](https://ieee-dataport.org/open-access/bvi-lowlight-fully-registered-datasets-low-light-image-and-video-enhancement)] ![Stars](https://img.shields.io/github/stars/russellllaputa/BVI-Mamba)
- **MLFSR: Mamba-based Light Field Super-Resolution with Efficient Subspace Scanning** - June 23, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.16083)]
- **LFMamba: Light Field Image Super-Resolution with State Space Model** - June 18, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.12463)]
- **HDMba: Hyperspectral Remote Sensing Imagery Dehazing with State Space Model** - June 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.12463)] [[Code](https://github.com/RsAI-lab/HDMba)] ![Stars](https://img.shields.io/github/stars/RsAI-lab/HDMba)
- **WaterMamba: Visual State Space Model for Underwater Image Enhancement** - May 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.08419)]
- **DVMSR: Distillated Vision Mamba for Efficient Super-Resolution** - May 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.03008)] [[Code](https://github.com/nathan66666/DVMSR)] ![Stars](https://img.shields.io/github/stars/nathan66666/DVMSR)
- **FourierMamba: Fourier Learning Integration with State Space Models for Image Deraining** - May 29, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.19450)]
- **RetinexMamba: Retinex-based Mamba for Low-light Image Enhancement** - May 6, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.03349)] [[Code](https://github.com/YhuoyuH/RetinexMamba)] ![Stars](https://img.shields.io/github/stars/YhuoyuH/RetinexMamba)
- **SRODT: Sparse Reconstruction of Optical Doppler Tomography Based on State Space Model** - April 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.07022)]
-  **MambaUIE&SR: Unraveling the Ocean's Secrets with Only 2.8 GFLOPs** - April 22, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.13884)] [[Code](https://github.com/1024AILab/MambaUIE)] ![Stars](https://img.shields.io/github/stars/1024AILab/MambaUIE)
- **PixMamba: Leveraging State Space Models in a Dual-Level Architecture for Underwater Image Enhancement** - April 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.08444)] [[Code](https://github.com/weitunglin/pixmamba)] ![Stars](https://img.shields.io/github/stars/weitunglin/pixmamba)
- **LLEMamba: Low-Light Enhancement via Relighting-Guided Mamba with Deep Unfolding Network** - March 3, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.01028)]
- **UVM-Net: U-shaped Vision Mamba for Single Image Dehazing** - February 15, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.04139)] [[Code](https://github.com/zzr-idam/UVM-Net)] ![Stars](https://img.shields.io/github/stars/zzr-idam/UVM-Net)
- **FDVM-Net: FD-Vision Mamba for Endoscopic Exposure Correction** - February 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.06378)] [[Code](https://github.com/zzr-idam/FDV-NET)] ![Stars](https://img.shields.io/github/stars/zzr-idam/FDV-NET)

## Generation and Restoration

- **DiM-3D: Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs** - June 7, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.05038)]
- **DiM2: Scaling Diffusion Mamba with Bidirectional SSMs for Efficient Image and Video Generation** - May 24, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.15881)]
- **Dimba: Transformer-Mamba Diffusion Models** - June 3, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.01159)] [[Homepage](https://dimba-project.github.io/)] [[Code](https://github.com/feizc/Dimba)] ![Stars](https://img.shields.io/github/stars/feizc/Dimba)
- **GMSR: Gradient-Guided Mamba for Spectral Reconstruction from RGB Images** - May 13, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.07777)] [[Code](https://github.com/wxy11-27/GMSR)] ![Stars](https://img.shields.io/github/stars/wxy11-27/GMSR)
- **DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis** - May 23, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.14224)] [[Code](https://github.com/tyshiwo1/DiM-DiffusionMamba/)] ![Stars](https://img.shields.io/github/stars/tyshiwo1/DiM-DiffusionMamba)
- **UniTraj: Deciphering Movement: Unified Trajectory Generation Model for Multi-Agent** - May 27, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.17680)] [[Code](https://github.com/colorfulfuture/UniTraj-pytorch)] ![Stars](https://img.shields.io/github/stars/colorfulfuture/UniTraj-pytorch)
- **CU-Mamba: Selective State Space Models with Channel Learning for Image Restoration** - April 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.11778)]
- **ZigMa: A DiT-style Zigzag Mamba Diffusion Model** - April 1, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.13802)] [[Homepage](https://taohu.me/zigma/)] [[Code](https://github.com/CompVis/zigma)] ![Stars](https://img.shields.io/github/stars/CompVis/zigma)
- **T-Mamba: Frequency-Enhanced Gated Long-Range Dependency for Tooth 3D CBCT Segmentation** - April 1, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.01065)] [[Code](https://github.com/isbrycee/T-Mamba)] ![Stars](https://img.shields.io/github/stars/isbrycee/T-Mamba)
- **Gamba: Marry Gaussian Splatting with Mamba for Single View 3D Reconstruction** - March 29, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.18795)]
- **MambaIR: A Simple Baseline for Image Restoration with State-Space Model** - March 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.15648)] [[Code](https://github.com/csguoh/MambaIR)] ![Stars](https://img.shields.io/github/stars/csguoh/MambaIR)
- **MotionMamba: Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM** - March 19, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.07487)] [[Homepage](https://steve-zeyu-zhang.github.io/MotionMamba/)] [[Code](https://github.com/steve-zeyu-zhang/MotionMamba/)] ![Stars](https://img.shields.io/github/stars/steve-zeyu-zhang/MotionMamba)
- **VmambaIR: Visual State Space Model for Image Restoration** - March 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.11423)] [[Code](https://github.com/AlphacatPlus/VmambaIR)] ![Stars](https://img.shields.io/github/stars/AlphacatPlus/VmambaIR)
- **Serpent: Scalable and Efficient Image Restoration via Multi-scale Structured State Space Models** - March 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.17902)]
- **DiS: Scalable Diffusion Models with State Space Backbone** - February 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.05608)] [[Code](https://github.com/feizc/DiS)] ![Stars](https://img.shields.io/github/stars/feizc/DiS)
- **MMA: Activating Wider Areas in Image Super-Resolution** - March 13, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.08330)] [[Code]([https://github.com/MMA-Lab/MMA](https://github.com/ArsenalCheng/MMA))] ![Stars](https://img.shields.io/github/stars/ArsenalCheng/MMA)

## Point Cloud Analysis

- **Mamba24/8D: Enhancing Global Interaction in Point Clouds via State Space Model** - June 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.17442)]
- **PointABM: Integrating Bidirectional State Space Model with Multi-Head Self-Attention for Point Cloud Analysis** - June 10, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.06069)]
- **PoinTramba: A Hybrid Transformer-Mamba Framework for Point Cloud Analysis** - May 24, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.15463)] [[Code](https://github.com/xiaoyao3302/PoinTramba)] ![Stars](https://img.shields.io/github/stars/xiaoyao3302/PoinTramba)
- **MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models** - May 23, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.14338)]
- **OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition** - May 13, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.07966)]
- **Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model** - April 23, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.14966)] [[Code](https://github.com/xhanxu/Mamba3D)] ![Stars](https://img.shields.io/github/stars/xhanxu/Mamba3D)
- **3DMambaComplete: Exploring Structured State Space Model for Point Cloud Completion** - April 10, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.07106)]
- **3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering** - April 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.05522)]
- **PointMamba: A Simple State Space Model for Point Cloud Analysis** - April 2, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.10739)] [[Code](https://github.com/LMD0311/PointMamba)] ![Stars](https://img.shields.io/github/stars/LMD0311/PointMamba)
- **Point Mamba: A Novel Point Cloud Backbone Based on State Space Model with Octree-Based Ordering Strategy** - March 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.06467)] [[Code](https://github.com/IRMVLab/Point-Mamba)] ![Stars](https://img.shields.io/github/stars/IRMVLab/Point-Mamba)
- **Point Cloud Mamba (PCM): Point Cloud Learning via State Space Model** - March 1, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.00762)] [[Code](https://github.com/SkyworkAI/PointCloudMamba)] ![Stars](https://img.shields.io/github/stars/SkyworkAI/PointCloudMamba)

## Video Processing

- **VideoMambaPro: A Leap Forward for Mamba in Video Understanding** - June 27, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.19006)] [[Code](https://github.com/hotfinda/VideoMambaPro)] ![Stars](https://img.shields.io/github/stars/hotfinda/VideoMambaPro)
- **SSM-Based Event Vision: State Space Models for Event Cameras** - CVPR 2024 [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zubic_State_Space_Models_for_Event_Cameras_CVPR_2024_paper.pdf)] [[Code](https://github.com/uzh-rpg/ssms_event_cameras)] ![Stars](https://img.shields.io/github/stars/uzh-rpg/ssms_event_cameras)
- **DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark** - May 30, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.19707)] [[Code](https://github.com/chenhaoxing/DeMamba)] ![Stars](https://img.shields.io/github/stars/chenhaoxing/DeMamba)
- **Matten: Video Generation with Mamba-Attention** - May 5, 2024, arXiv [[Paper (https://arxiv.org/abs/2405.03025)]
- **RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos** - April 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.06483)] [[Code](https://github.com/zizheng-guo/RhythmMamba)] ![Stars](https://img.shields.io/github/stars/zizheng-guo/RhythmMamba)
- **Simba: Simplified Mamba-Based Architecture for Vision and Multivariate Time Series** - April 4, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.15360)] [[Code](https://github.com/badripatro/Simba)] ![Stars](https://img.shields.io/github/stars/badripatro/Simba)
- **VideoMamba Suite: State Space Model as a Versatile Alternative for Video Understanding** - March 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.09626)] [[Code](https://github.com/OpenGVLab/video-mamba-suite)] ![Stars](https://img.shields.io/github/stars/OpenGVLab/video-mamba-suite)
- **SSM Diffusion: SSM Meets Video Diffusion Models: Efficient Long-Term Video Generation with Structured State Spaces** - March 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.07711)] [[Code](https://github.com/shim0114/SSM-Meets-Video-Diffusion-Models)] ![Stars](https://img.shields.io/github/stars/shim0114/SSM-Meets-Video-Diffusion-Models)
- **VideoMamba: State Space Model for Efficient Video Understanding** - March 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.06977)] [[Code](https://github.com/OpenGVLab/VideoMamba)] ![Stars](https://img.shields.io/github/stars/OpenGVLab/VideoMamba)
- **SSSMLV: Selective Structured State-Spaces for Long-Form Video Understanding** - CVPR 2023, arXiv [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Selective_Structured_State-Spaces_for_Long-Form_Video_Understanding_CVPR_2023_paper.html)]
- **ViS4mer: Video Synthesis Framework with Mamba-Based Attention Mechanism** - ECCV 2022 [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19833-5_6)] [[Code](https://github.com/md-mohaiminul/ViS4mer)] ![Stars](https://img.shields.io/github/stars/md-mohaiminul/ViS4mer)

## Remote Sensing

- **PyramidMamba: Rethinking Pyramid Feature Fusion with Selective Space State Model for Semantic Segmentation of Remote Sensing Imagery** - June 16, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.10828)] [[Code](https://github.com/WangLibo1995/GeoSeg)] ![Stars](https://img.shields.io/github/stars/WangLibo1995/GeoSeg)
- **CDMamba: Remote Sensing Image Change Detection with Mamba** - June 6, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.04207)] [[Code](https://github.com/zmoka-zht/CDMamba)] ![Stars](https://img.shields.io/github/stars/zmoka-zht/CDMamba)
- **RSDehamba: Lightweight Vision Mamba for Remote Sensing Satellite Image Dehazing** - May 16, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.10030)]
- **CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation** - May 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.10530)] [[Code](https://github.com/XiaoBuL/CM-UNet)] ![Stars](https://img.shields.io/github/stars/XiaoBuL/CM-UNet)
- **FMSR: Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution** - May 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.04964)]
- **RSCaMa: Remote Sensing Image Change Captioning with State Space Model** - May 2, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.18895)] [[Code](https://github.com/Chen-Yang-Liu/RSCaMa)] ![Stars](https://img.shields.io/github/stars/Chen-Yang-Liu/RSCaMa)
- **RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation** - April 3, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.02457)] [[Code](https://github.com/sstary/SSRS)] ![Stars](https://img.shields.io/github/stars/sstary/SSRS)
- **ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model** - April 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.03425)] [[Code](https://github.com/ChenHongruixuan/MambaCD)] ![Stars](https://img.shields.io/github/stars/ChenHongruixuan/MambaCD)
- **HSIDMamba: Exploring Bidirectional State-Space Models for Hyperspectral Denoising** - April 15, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.09697)]
- **SpectralMamba: Efficient Mamba for Hyperspectral Image Classification** - April 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.08489)] [[Code](https://github.com/danfenghong/SpectralMamba)] ![Stars](https://img.shields.io/github/stars/danfenghong/SpectralMamba)
- **Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model** - April 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.01705)] [[Code](https://github.com/zhuqinfeng1999/Samba)] ![Stars](https://img.shields.io/github/stars/zhuqinfeng1999/Samba)
- **SS-Mamba: Spectral-Spatial Mamba for Hyperspectral Image Classification** - April 29, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.18401)]
- **3DSS-Mamba: 3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification** - May 21, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.12487)]
- **S^2 Mamba: A Spatial-spectral State Space Model for Hyperspectral Image Classification** - April 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.18213)] [[Code](https://github.com/PURE-melo/S2Mamba)] ![Stars](https://img.shields.io/github/stars/PURE-melo/S2Mamba)
- **RS-Mamba: Remote Sensing Image Classification with State Space Model** - March 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.19654)]
- **RSMamba: Remote Sensing Image Classification with State Space Model** - March 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.19654)]
- **Pan-Mamba: Effective Pan-sharpening with State Space Model** - March 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.12192)] [[Code](https://github.com/alexhe101/Pan-Mamba)] ![Stars](https://img.shields.io/github/stars/alexhe101/Pan-Mamba)
- **LE-Mamba: Local Enhancement and Mamba-based Architecture for Remote Sensing Image Super-Resolution** - February 21, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.09293)] [[Code](https://github.com/294coder/Efficient-MIF)] ![Stars](https://img.shields.io/github/stars/294coder/Efficient-MIF)
- **Seg-LSTM: Performance of xLSTM for Semantic Segmentation of Remotely Sensed Images** - June 20, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.14086)] [[Code](https://github.com/zhuqinfeng1999/Seg-LSTM)] ![Stars](https://img.shields.io/github/stars/zhuqinfeng1999/Seg-LSTM)
- **VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting** - CVPRW 2024 [[Paper](https://openaccess.thecvf.com/content/CVPR2024W/PRECOGNITION/html/Tang_VMRNN_Integrating_Vision_Mamba_and_LSTM_for_Efficient_and_Accurate_CVPRW_2024_paper.html)] [[Code]([https://github.com/VMRNN/VMRNN](https://github.com/yyyujintang/VMRNN-PyTorch))] ![Stars](https://img.shields.io/github/stars/yyyujintang/VMRNN-PyTorch)

## Medical Image Analysis
### Medical Image Classification

- **Nasiri-Sarvi et al. (2024): AI-Based Breast Cancer Detection Using State Space Models** - July 15, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.06784)]
- **Vim4Path: Vision Mamba Framework for Computational Pathology** - June 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.13748)] [[Code](https://github.com/vim4path/Vim4Path)] ![Stars](https://img.shields.io/github/stars/vim4path/Vim4Path)
- **MedMamba: Vision Mamba for Medical Image Classification** - April 2, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.03849)] [[Code](https://github.com/YubiaoYue/MedMamba)] ![Stars](https://img.shields.io/github/stars/YubiaoYue/MedMamba)
- **CMViM: Contrastive Masked Vim Autoencoder for 3D Multi-modal Representation Learning for AD Classification** - March 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.16520)]
- **MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology** - March 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.06800)] [[Code](https://github.com/isyangshu/MambaMIL)] ![Stars](https://img.shields.io/github/stars/isyangshu/MambaMIL)
- **MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models** - March 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.05160)]
- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** - January 17, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.09417)] [[Code](https://github.com/hustvl/Vim)] ![Stars](https://img.shields.io/github/stars/hustvl/Vim)

### Medical Image Segmentation  
#### Medical 2D Image Segmentation

- **xLSTM-UNet: Integrating Long Short-Term Memory Networks with UNet for Medical Image Segmentation** - July 19, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.08593)] [[Code](https://github.com/xLSTM-UNet/xLSTM-UNet)] ![Stars](https://img.shields.io/github/stars/xLSTM-UNet/xLSTM-UNet)
- **SliceMamba: Vision Mamba-based Sliced Representation for 2D Medical Image Segmentation** - June 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.08321)] [[Code](https://github.com/SliceMamba/SliceMamba)] ![Stars](https://img.shields.io/github/stars/SliceMamba/SliceMamba)
- **MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba** - June 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.05992)] [[Code](https://github.com/PixDeep/MHS-VM)] ![Stars](https://img.shields.io/github/stars/PixDeep/MHS-VM)
- **MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion Segmentation** - May 24, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.15925)] [[Code](https://github.com/chunyuyuan/MUCM-Net)] ![Stars](https://img.shields.io/github/stars/chunyuyuan/MUCM-Net)
- **AC-MambaSeg: An Adaptive Convolution and Mamba-based Architecture for Enhanced Skin Lesion Segmentation** - May 5, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.03011)] [[Code](https://github.com/vietthanh2710/AC-MambaSeg)] ![Stars](https://img.shields.io/github/stars/vietthanh2710/AC-MambaSeg)
- **HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation** - May 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.05007)]
- **ViM-UNet: Vision Mamba Integrated UNet for Accurate Medical Image Segmentation** - April 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.09783)] [[Code](https://github.com/ViM-UNet/ViM-UNet)] ![Stars](https://img.shields.io/github/stars/ViM-UNet/ViM-UNet)
- **UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation** - April 24, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.20035)] [[Code](https://github.com/wurenkai/UltraLight-VM-UNet)] ![Stars](https://img.shields.io/github/stars/wurenkai/UltraLight-VM-UNet)
- **H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation** - March 20, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.13642)] [[Code](https://github.com/wurenkai/H-vmunet)] ![Stars](https://img.shields.io/github/stars/wurenkai/H-vmunet)
- **ProMamba: Prompt-Mamba for Polyp Segmentation** - March 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.13660)]
- **VM-UNet V2: Rethinking Vision Mamba UNet for Medical Image Segmentation** - March 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.09157)] [[Code](https://github.com/nobodyplayer1/VM-UNetV2)] ![Stars](https://img.shields.io/github/stars/nobodyplayer1/VM-UNetV2)
- **Semi-Mamba-UNet: Pixel-Level Contrastive and Pixel-Level Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation** - March 29, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.07245)] [[Code](https://github.com/ziyangwang007/Mamba-UNet)] ![Stars](https://img.shields.io/github/stars/ziyangwang007/Mamba-UNet)
- **Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation** - March 30, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.05079)] [[Code](https://github.com/ziyangwang007/Mamba-UNet)] ![Stars](https://img.shields.io/github/stars/ziyangwang007/Mamba-UNet)
- **P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation** - March 15, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.08506)]
- **Weak-Mamba-UNet: Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation** - February 16, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.10887)] [[Code](https://github.com/ziyangwang007/Mamba-UNet)] ![Stars](https://img.shields.io/github/stars/ziyangwang007/Mamba-UNet)
- **Swin-UMamba: Mamba-based UNet with ImageNet-based Pretraining** - March 6, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.03302)] [[Code](https://github.com/JiarunLiu/Swin-UMamba)] ![Stars](https://img.shields.io/github/stars/JiarunLiu/Swin-UMamba)
- **VM-UNet: Vision Mamba UNet for Medical Image Segmentation** - February 4, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.02491)] [[Code](https://github.com/JCruan519/VM-UNet)] ![Stars](https://img.shields.io/github/stars/JCruan519/VM-UNet)
- **U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation** - January 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.04722)] [[Homepage](https://wanglab.ai/u-mamba.html)] [[Code](https://github.com/bowang-lab/U-Mamba)] ![Stars](https://img.shields.io/github/stars/bowang-lab/U-Mamba)

## Medical 3D Image Segmentation

- **TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction for 3D Medical Image Segmentation** - July 27, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.15584)] [[Code](https://github.com/ydchen0806/TokenUnify)] ![Stars](https://img.shields.io/github/stars/ydchen0806/TokenUnify)
- **T-Mamba: Frequency-Enhanced Gated Long-Range Dependency for Tooth 3D CBCT Segmentation** - April 1, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.01065)] [[Code](https://github.com/isbrycee/T-Mamba)] ![Stars](https://img.shields.io/github/stars/isbrycee/T-Mamba)
- **LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation** - March 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.05246)] [[Code](https://github.com/MrBlankness/LightM-UNet)] ![Stars](https://img.shields.io/github/stars/MrBlankness/LightM-UNet)
- **Vivim: a Video Vision Mamba for Medical Video Object Segmentation** - March 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.14168)] [[Code](https://github.com/scott-yjyang/Vivim)] ![Stars](https://img.shields.io/github/stars/scott-yjyang/Vivim)
- **nnMamba: 3D Biomedical Image Segmentation, Classification, and Landmark Detection with State Space Model** - March 10, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.03526)] [[Code](https://github.com/lhaof/nnMamba)] ![Stars](https://img.shields.io/github/stars/lhaof/nnMamba)
- **SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation** - February 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.13560)] [[Code](https://github.com/ge-xing/SegMamba)] ![Stars](https://img.shields.io/github/stars/ge-xing/SegMamba)
- **Mamba-HUNet: Hybrid UNet Based on Mamba for Multi-modal Brain Tumor Segmentation** - February 19, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.09217)] [[Code](https://github.com/Hybrid-UNet/Mamba-HUNet)] ![Stars](https://img.shields.io/github/stars/Hybrid-UNet/Mamba-HUNet)
- **TM-UNet: Transformer Mamba Integrated UNet for Efficient 3D Medical Image Segmentation** - February 15, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.09012)] [[Code](https://github.com/Transform-Mamba/TM-UNet)] ![Stars](https://img.shields.io/github/stars/Transform-Mamba/TM-UNet)
- **LKM-UNet: Light and Knowledge-aware Mamba UNet for Faster and Accurate 3D Image Segmentation** - February 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.08629)] [[Code](https://github.com/LKM-UNet/LKM-UNet)] ![Stars](https://img.shields.io/github/stars/LKM-UNet/LKM-UNet)

## Medical Image Reconstruction

- **MMR-Mamba: Multi-Modal MRI Reconstruction with Mamba and Spatial-Frequency Information Fusion** - June 27, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.18950)]
- **MambaMIR-GAN: Generative Adversarial Network-based Mamba for Enhanced Medical Image Reconstruction** - March 19, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.13897)] [[Code](https://github.com/MambaMIR/MambaMIR-GAN)] ![Stars](https://img.shields.io/github/stars/MambaMIR/MambaMIR-GAN)
- **MambaMIR: An Arbitrary-Masked Mamba for Joint Medical Image Reconstruction and Uncertainty Estimation** - March 19, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.18451)] [[Code](https://github.com/ShumengLI/Mamba4MIS)] ![Stars](https://img.shields.io/github/stars/ShumengLI/Mamba4MIS)

### Other Tasks in Medical Imaging

- **SR-Mamba: Effective Surgical Phase Recognition with State Space Model** - July 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.08333)] [[Code](https://github.com/rcao-hk/SR-Mamba)] ![Stars](https://img.shields.io/github/stars/rcao-hk/SR-Mamba)
- **Deform-Mamba: Deformable Mamba Network for MRI Super-Resolution** - July 8, 2024, arXiv [[Paper](https://arxiv.org/abs/2407.05969)]
- **SMamba-UNet: Soft Masked Mamba Diffusion Model for CT to MRI Conversion** - June 22, 2024, arXiv [[Paper](https://arxiv.org/abs/2406.15910)] [[Code](https://github.com/wongzbb/DiffMa-Diffusion-Mamba)] ![Stars](https://img.shields.io/github/stars/wongzbb/DiffMa-Diffusion-Mamba)
- **I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling** - May 22, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.14022)] [[Code](https://github.com/icon-lab/I2I-Mamba)] ![Stars](https://img.shields.io/github/stars/icon-lab/I2I-Mamba)
- **BI-Mamba: Cardiovascular Disease Detection from Multi-View Chest X-rays with BI-Mamba** - May 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.18533)]
- **VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis** - May 9, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.05667)]
- **MGDC Tracker: Motion-Guided Dual-Camera Tracker for Low-Cost Skill Evaluation of Gastric Endoscopy** - April 20, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.05146)] [[Code](https://github.com/PieceZhang/MotionDCTrack)] ![Stars](https://img.shields.io/github/stars/PieceZhang/MotionDCTrack)
- **VMambaMorph: a Visual Mamba-based Framework with Cross-Scan Module for Deformable 3D Image Registration** - April 7, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.05105v2)] [[Code](https://github.com/ziyangwang007/VMambaMorph)] ![Stars](https://img.shields.io/github/stars/ziyangwang007/VMambaMorph)
- **MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction** - March 13, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.08479)] [[Code](https://github.com/LinjieFu-U/mamba_dose)] ![Stars](https://img.shields.io/github/stars/LinjieFu-U/mamba_dose)

## MultiModal

- **Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models** - May 27, 2024, arXiv [[Paper](https://arxiv.org/abs/2405.15574)] [[Code](https://github.com/ByungKwanLee/Meteor)] ![Stars](https://img.shields.io/github/stars/ByungKwanLee/Meteor)
- **Mamba-FETrack: Frame-Event Tracking via State Space Model** - April 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.18174)] [[Code](https://github.com/Event-AHU/Mamba_FETrack)] ![Stars](https://img.shields.io/github/stars/Event-AHU/Mamba_FETrack)
- **TransMA: A Transformer-based Multi-modal Mamba Architecture for Efficient Cross-modal Retrieval** - April 18, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.09156)] [[Code](https://github.com/TransMA/TransMA)] ![Stars](https://img.shields.io/github/stars/TransMA/TransMA)
- **Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation** - April 5, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.04256)] [[Code](https://github.com/zifuwan/Sigma)] ![Stars](https://img.shields.io/github/stars/zifuwan/Sigma)
- **FusionMamba: Efficient Image Fusion with State Space Model** - April 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.07932)]
- **SurvMamba: State Space Model with Multi-grained Multi-modal Interaction for Survival Prediction** - April 11, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.08027)]
- **SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding** - April 1, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.01174)]
- **ReMamber: Referring Image Segmentation with Mamba Twister** - March 26, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.17839)]
- **CMViM: Contrastive Masked Vim Autoencoder for 3D Multi-modal Representation Learning for AD classification** - March 25, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.16520)]
- **Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference** - March 22, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.14520)] [[Homepage](https://sites.google.com/view/cobravlm)] [[Code](https://github.com/h-zhao1997/cobra)] ![Stars](https://img.shields.io/github/stars/h-zhao1997/cobra)
- **VL-Mamba: Exploring State Space Models for Multimodal Learning** - March 20, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.13600)] [[Homepage](https://yanyuanqiao.github.io/vl-mamba/)] [[Code](https://github.com/ZhengYu518/VL-Mamba)] ![Stars](https://img.shields.io/github/stars/ZhengYu518/VL-Mamba)
- **MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models** - March 14, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.09471)]
- **MambaMorph: a Mamba-based Framework for Medical MR-CT Deformable Registration** - March 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2401.13934)] [[Code](https://github.com/Guo-Stone/MambaMorph)] ![Stars](https://img.shields.io/github/stars/Guo-Stone/MambaMorph)
- **BoardMamba: Multi-modal Mamba for Table Understanding and Interaction** - March 5, 2024, arXiv [[Paper](https://arxiv.org/abs/2403.05168)] [[Code](https://github.com/board-mamba/BoardMamba)] ![Stars](https://img.shields.io/github/stars/board-mamba/BoardMamba)
- **TM-Mamba: Text-Mamba for Multimodal Knowledge Alignment and Retrieval** - February 28, 2024, arXiv [[Paper](https://arxiv.org/abs/2402.09458)] [[Code](https://github.com/tm-mamba/TM-Mamba)] ![Stars](https://img.shields.io/github/stars/tm-mamba/TM-Mamba)
- **MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion** - April 12, 2024, arXiv [[Paper](https://arxiv.org/abs/2404.08406)]

## Other Tasks
Will be added ... 
