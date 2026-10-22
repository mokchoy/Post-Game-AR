# Research informing the tracking and occlusion experiments

The application is a video-see-through AR preview: a virtual panel has a fixed
metric pose in the tag/world frame. Pose estimation and real/virtual visibility
are separate problems. Sparse natural landmarks can localize the camera, but
do not provide dense foreground depth or automatically hide a virtual panel
behind a hand. A 3D mesh is not mandatory; camera-Z depth or a depth-conditioned
visibility estimate can support compositing.

| Paper | Relevance and use in this prototype |
|---|---|
| [UcoSLAM: Simultaneous Localization and Mapping by Fusion of KeyPoints and Squared Planar Markers](https://arxiv.org/abs/1902.03729), Muñoz-Salinas and Medina-Carnicer, Pattern Recognition 2020 | Closest architectural reference for combining natural features with known-scale square markers. Markers establish scale and aid relocalization; mapped features support tracking away from a marker. This prototype evaluates its own fixed-map implementation; it does not claim UcoSLAM equivalence or import UcoSLAM. |
| [Parallel Tracking and Mapping for Small AR Workspaces](https://robots.ox.ac.uk/~gk/PTAM/), Klein and Murray, ISMAR 2007 | Foundation for a moving monocular camera and a small persistent workspace map. Translated views matter for triangulation; a count of repeated corners is not a substitute for viewpoint coverage. |
| [DynaSLAM: Tracking, Mapping and Inpainting in Dynamic Scenes](https://arxiv.org/abs/1806.05620), Bescós et al., RA-L 2018 | Explains why dynamic foreground features must be distinguished from the static scene map. The current experiment keeps people/hands out of map capture and tests robust rejection during occlusion; it does not add DynaSLAM's segmentation/inpainting stack. |
| [Virtual Occlusions Through Implicit Depth](https://nianticlabs.github.io/implicit-depth/), Watson et al., CVPR 2023; [author code and models](https://github.com/nianticlabs/implicit-depth) | Predicts visibility directly from RGB and virtual geometry depth, with temporal information. This is the relevant comparison against explicit depth regression. Exported rectified frames, camera poses, intrinsics and virtual camera-Z depth support its input preparation. Actual model inference and scene-specific accuracy are pending the accepted tracking recordings. |
| [DepthLab: Real-time 3D Interaction with Depth Maps for Mobile Augmented Reality](https://research.google/pubs/depthlab-real-time-3d-interaction-with-depth-maps-for-mobile-augmented-reality/), Du et al., UIST 2020 | Reference for turning usable depth into AR occlusion and interaction. It does not make a plain webcam a measured-depth sensor; the depth estimation source and validity still need evaluation. |
| [Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Video_Depth_Anything_Consistent_Depth_Estimation_for_Super-Long_Videos_CVPR_2025_paper.html), Chen et al., CVPR 2025 | Addresses temporal inconsistency in monocular video depth. It is a follow-up candidate if single-frame estimates produce unacceptable boundary flicker. Published speed is not a benchmark of this laptop. |
| [Depth Anything V2](https://arxiv.org/abs/2406.09414), Yang et al., NeurIPS 2024; [metric models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) | The optional initial depth adapter uses Small/Hypersim metric weights and the author's inference implementation. Its output is depth in metres before visualization normalization. Learned metric output still needs scale validation against this camera and scene. |

The implemented ablations deliberately isolate temporal LK tracking from ORB
redetection/map matching. The pose solver still requires independent landmark
PnP eligibility before joint fitting; diagnostics expose this choice. A future
change allowing fewer landmarks to assist a valid tag fit would not, by itself,
prove successful tag-independent localization.

Acceptance uses tag-reference agreement, physical coverage tests, application
latency and manually labelled visibility masks. Neither small fitted reprojection
residuals nor synthetic tests establish real-camera registration accuracy. No
neural model speed, mask quality, or sustained physical tracking result is claimed
until the corresponding measurements exist.
