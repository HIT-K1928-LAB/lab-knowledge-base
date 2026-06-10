# 目标检测

目标检测要求模型同时回答“图像里有什么”和“目标在哪里”。它从图像分类扩展到空间定位，是自动驾驶、安防、机器人、遥感和工业质检中的核心任务。

## 1. 学习目标
- 理解检测任务的基本输出：类别、边界框、置信度。
- 掌握 IoU、NMS、AP/mAP、召回率等核心概念。
- 区分两阶段、一阶段、anchor-free、Transformer 检测器的设计差异。
- 能够读懂检测框架中的 backbone、neck、head、loss、assigner、post-process。
- 能够使用主流检测仓库训练一个自定义数据集。

## 2. 发展脉络
- 传统阶段依赖滑动窗口、Haar、HOG、DPM 等特征和分类器组合，检测流程复杂且计算昂贵。
- 2014 年，R-CNN 将 CNN 特征引入候选区域分类，深度检测开始成为主流。
- 2015 年，Fast R-CNN 与 Faster R-CNN 通过共享特征和 RPN 显著提升效率，形成两阶段检测基准。
- 2015—2018 年，YOLO、SSD、RetinaNet 推动一阶段检测发展，速度和工程可用性大幅提升。
- 2019 年以后，FCOS、CenterNet 等 anchor-free 方法减少 anchor 设计依赖。
- 2020 年以后，DETR 及其改进版本把目标检测建模为端到端集合预测问题，推动 Transformer 检测器发展。

## 3. 方法地图
| 类别 | 代表方法 | 核心思想 | 适合重点理解 |
| --- | --- | --- | --- |
| 传统检测 | Viola-Jones / HOG / DPM | 滑窗 + 手工特征 + 分类器 | 早期检测 pipeline |
| 两阶段检测 | R-CNN / Fast R-CNN / Faster R-CNN | 先产生候选框，再分类和回归 | proposal、RoI Pooling、RPN |
| 特征金字塔 | FPN / Mask R-CNN | 多尺度特征融合，提高小目标检测能力 | neck 结构、多尺度预测 |
| 一阶段检测 | YOLO / SSD / RetinaNet | 直接在特征图上密集预测类别和框 | anchor、dense prediction、速度优势 |
| Anchor-free | FCOS / CenterNet | 直接预测中心点或边界距离 | label assignment、中心采样 |
| Transformer 检测 | DETR / Deformable DETR / DINO | 用 object query 和集合匹配进行端到端预测 | query、Hungarian matching、注意力 |
| 开放词表检测 | GLIP / Grounding DINO | 融合语言提示和检测框预测 | 视觉语言 grounding |

## 4. 入门路线
1. 先理解边界框、IoU、NMS、AP/mAP，能手算简单例子。
2. 阅读 R-CNN、Fast R-CNN、Faster R-CNN，掌握两阶段检测的完整流程。
3. 阅读 YOLO、SSD、RetinaNet，理解一阶段检测为什么更快，以及类别不均衡如何处理。
4. 在 `mmdetection` 或 `detectron2` 中训练 Faster R-CNN / RetinaNet，观察配置文件如何组织。

## 5. 进阶路线
1. 学习 FPN、Mask R-CNN，理解多尺度特征对检测和实例分割的意义。
2. 学习 FCOS、CenterNet，比较 anchor-based 与 anchor-free 的标签分配差异。
3. 学习 DETR、Deformable DETR，理解 Hungarian matching 和 object query。
4. 进一步关注 DINO、Grounding DINO，把检测扩展到开放词表和文本引导检测。

## 6. 论文精读
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | Rich feature hierarchies for accurate object detection and semantic segmentation | [paper](https://arxiv.org/abs/1311.2524) | R-CNN、候选区域、CNN 特征迁移 |
| 2 | Fast R-CNN | [paper](https://arxiv.org/abs/1504.08083) | RoI Pooling、多任务损失、共享卷积特征 |
| 3 | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | [paper](https://arxiv.org/abs/1506.01497) | RPN、anchor、两阶段检测标准范式 |
| 4 | Feature Pyramid Networks for Object Detection | [paper](https://arxiv.org/abs/1612.03144) | 多尺度特征融合、检测 neck 设计 |
| 5 | You Only Look Once: Unified, Real-Time Object Detection | [paper](https://arxiv.org/abs/1506.02640) | YOLO、统一回归、实时检测 |
| 6 | SSD: Single Shot MultiBox Detector | [paper](https://arxiv.org/abs/1512.02325) | 多尺度默认框、单阶段检测 |
| 7 | Focal Loss for Dense Object Detection | [paper](https://arxiv.org/abs/1708.02002) | RetinaNet、类别不均衡、focal loss |
| 8 | Fully Convolutional One-Stage Object Detection | [paper](https://arxiv.org/abs/1904.01355) | FCOS、anchor-free、中心采样 |
| 9 | End-to-End Object Detection with Transformers | [paper](https://arxiv.org/abs/2005.12872) | DETR、集合预测、Hungarian matching |
| 10 | Deformable DETR: Deformable Transformers for End-to-End Object Detection | [paper](https://arxiv.org/abs/2010.04159) | 可变形注意力、收敛速度、小目标 |

## 7. 代码实践
| 仓库 | 覆盖方向 | 建议用途 |
| --- | --- | --- |
| [mmdetection](https://github.com/open-mmlab/mmdetection) | 检测算法集合 | 系统学习配置、训练、评估与模型 zoo |
| [detectron2](https://github.com/facebookresearch/detectron2) | 检测/分割研究平台 | 阅读高质量工程实现 |
| [ultralytics](https://github.com/ultralytics/ultralytics) | YOLO 系列 | 快速训练和部署实时检测模型 |
| [detr](https://github.com/facebookresearch/detr) | DETR | 阅读端到端检测官方实现 |
| [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | Deformable DETR | 理解可变形注意力检测器 |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | 开放词表检测 | 进一步学习文本引导检测 |

## 8. 推荐学习顺序
`IoU / NMS / mAP → R-CNN → Fast R-CNN → Faster R-CNN → FPN → YOLO / SSD → RetinaNet → FCOS → DETR → Deformable DETR`

## 9. 常见问题
- 检测模型效果差时，不要只看 backbone，也要检查标注格式、类别分布、输入尺度、anchor/assigner 和 NMS 阈值。
- 小目标检测通常更依赖输入分辨率、多尺度特征和数据增强。
- DETR 类模型理论上端到端，但训练配置和数据规模仍然非常关键。

