# 语义分割

语义分割是像素级分类任务：为图像中的每个像素预测语义类别。它比图像分类更关注空间结构，比目标检测更强调边界和密集预测，是自动驾驶、医学影像、遥感、场景理解的重要基础。

## 1. 学习目标
- 理解像素级监督、上采样、跳跃连接、多尺度上下文建模。
- 掌握 mIoU、pixel accuracy、class IoU 等评价指标。
- 区分语义分割、实例分割、全景分割的任务边界。
- 理解编码器-解码器、空洞卷积、金字塔池化、Transformer decoder 的作用。
- 能够使用分割框架训练自定义数据集并分析错误样例。

## 2. 发展脉络
- 2015 年，FCN 把分类网络改成全卷积结构，实现端到端密集预测。
- 同期，U-Net 通过编码器-解码器和 skip connection 成为医学图像分割的经典架构。
- 2016—2018 年，PSPNet、DeepLab 系列用金字塔池化、ASPP 和空洞卷积增强上下文感知能力。
- 2019 年以后，HRNet、OCR、Fast-SCNN 等模型分别从高分辨率保持、上下文关系和实时推理方向推进。
- 2021 年后，SETR、SegFormer 等 Transformer 分割模型兴起，语义分割逐渐进入注意力建模时代。
- 2022 年以后，Mask2Former 等通用分割框架将语义、实例、全景分割统一到 mask classification 范式中。

## 3. 方法地图
| 类别 | 代表方法 | 核心思想 | 适合重点理解 |
| --- | --- | --- | --- |
| FCN 范式 | FCN | 把分类网络转为全卷积，输出像素级预测 | dense prediction、反卷积/上采样 |
| 编码器-解码器 | U-Net / SegNet | 编码语义、解码空间细节 | skip connection、边界恢复 |
| 上下文聚合 | PSPNet / DeepLab | 多尺度池化或空洞卷积扩大感受野 | PSP、ASPP、dilated convolution |
| 高分辨率表示 | HRNet / OCRNet | 保持高分辨率特征并建模类别上下文 | 表示融合、上下文关系 |
| 实时分割 | ENet / BiSeNet / Fast-SCNN | 牺牲部分精度换取低延迟 | 速度、模型压缩、双分支结构 |
| Transformer 分割 | SETR / SegFormer | 用 Transformer encoder 提取全局上下文 | token、位置编码、MLP decoder |
| 通用分割 | MaskFormer / Mask2Former | 用 mask classification 统一多种分割任务 | mask query、masked attention |

## 4. 入门路线
1. 先区分语义分割、实例分割、全景分割，并理解 mIoU 的计算方式。
2. 阅读 FCN 和 U-Net，掌握 dense prediction、上采样和 skip connection。
3. 阅读 PSPNet 和 DeepLabv3+，理解全局上下文和多尺度上下文为什么重要。
4. 用 `mmsegmentation` 训练一个小数据集，观察输入尺寸、类别不均衡和数据增强对结果的影响。

## 5. 进阶路线
1. 学习 HRNet / OCRNet，理解保持高分辨率特征与建模像素-类别关系的意义。
2. 学习 SegFormer，比较 CNN decoder 和 Transformer encoder 的差异。
3. 学习 Mask2Former，理解 mask classification 如何统一语义、实例、全景分割。
4. 进一步关注开放词表分割、交互式分割和基础模型分割，如 CLIPSeg、SAM 系列。

## 6. 论文精读
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | Fully Convolutional Networks for Semantic Segmentation | [paper](https://arxiv.org/abs/1411.4038) | FCN、上采样、跳跃连接 |
| 2 | U-Net: Convolutional Networks for Biomedical Image Segmentation | [paper](https://arxiv.org/abs/1505.04597) | U 型结构、skip connection、小样本医学分割 |
| 3 | Pyramid Scene Parsing Network | [paper](https://arxiv.org/abs/1612.01105) | PSPNet、全局上下文、金字塔池化 |
| 4 | Rethinking Atrous Convolution for Semantic Image Segmentation | [paper](https://arxiv.org/abs/1706.05587) | DeepLabv3、ASPP、空洞卷积 |
| 5 | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation | [paper](https://arxiv.org/abs/1802.02611) | DeepLabv3+、encoder-decoder、深度可分离卷积 |
| 6 | Fast-SCNN: Fast Semantic Segmentation Network | [paper](https://arxiv.org/abs/1902.04502) | 轻量实时分割、双路径设计 |
| 7 | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | [paper](https://arxiv.org/abs/2105.15203) | 层次化 Transformer、轻量 MLP decoder |
| 8 | Masked-attention Mask Transformer for Universal Image Segmentation | [paper](https://arxiv.org/abs/2112.01527) | Mask2Former、通用分割、masked attention |

## 7. 代码实践
| 仓库 | 覆盖方向 | 建议用途 |
| --- | --- | --- |
| [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | 通用语义分割 | 训练、评估、复现实验的首选框架 |
| [fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org) | FCN | 阅读经典 FCN 参考实现 |
| [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) | U-Net | 快速理解 U-Net 训练流程 |
| [DeepLab2](https://github.com/google-research/deeplab2) | DeepLab 系列 / Panoptic-DeepLab | 学习 Google 系分割实现 |
| [SegFormer](https://github.com/NVlabs/SegFormer) | Transformer 分割 | 阅读官方 SegFormer 配置和训练代码 |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) | 通用分割 | 学习统一分割框架 |

## 8. 推荐学习顺序
`mIoU / pixel accuracy → FCN → U-Net → PSPNet → DeepLabv3+ → Fast-SCNN → SegFormer → Mask2Former`

## 9. 常见问题
- 分割训练中 crop size、batch size、类别权重、ignore label 经常比模型结构更影响结果。
- 分割结果要同时看整体 mIoU 和小类别 IoU，平均指标可能掩盖长尾类别问题。
- 边界质量通常需要结合可视化检查，不能只看表格指标。

