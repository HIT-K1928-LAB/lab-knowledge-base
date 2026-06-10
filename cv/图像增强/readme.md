# 图像增强

图像增强在这里按广义低层视觉理解，包含超分辨率、去噪、去模糊、去雾、去雨、低照度增强和通用图像复原。它关注如何从退化图像恢复更清晰、更自然、更适合下游任务的图像。

## 1. 学习目标
- 理解图像退化模型：低分辨率、噪声、模糊、雾、雨、低照度等。
- 掌握低层视觉中常见损失：L1/L2、感知损失、对抗损失、SSIM、频域损失。
- 区分 PSNR/SSIM 这类保真指标和 LPIPS/MOS 这类感知质量指标。
- 理解 CNN、GAN、Transformer 在图像复原中的作用差异。
- 能够根据任务选择合适的数据集、评价指标和主观可视化方式。

## 2. 发展脉络
- 早期图像增强依赖 Retinex 理论、暗通道先验、稀疏表示、滤波和优化模型。
- 2015 年后，SRCNN、DnCNN 等方法把 CNN 引入超分辨率和去噪，低层视觉开始深度学习化。
- 2017—2019 年，SRGAN、ESRGAN 等方法用 GAN 和感知损失提升视觉真实感，但也带来 hallucination 风险。
- 2018 年后，RetinexNet、Zero-DCE 等低照度增强方法开始关注无参考或弱监督增强。
- 2021 年以后，SwinIR、Restormer 等 Transformer 结构成为图像复原强基线，并能覆盖多种退化。
- 近几年，通用复原、大模型先验、扩散模型和文本引导增强逐渐成为新趋势。

## 3. 方法地图
| 类别 | 代表方法 | 核心思想 | 适合重点理解 |
| --- | --- | --- | --- |
| 超分辨率 | SRCNN / EDSR / ESRGAN / SwinIR | 从低分辨率重建高分辨率细节 | 上采样、残差学习、感知质量 |
| 去噪 | DnCNN / FFDNet / Restormer | 学习噪声残差或直接恢复干净图像 | 残差学习、盲去噪、真实噪声 |
| 去模糊 | DeblurGAN / MPRNet / Restormer | 恢复运动模糊或散焦模糊图像 | 多阶段复原、频域与空间信息 |
| 去雾 / 去雨 | DehazeNet / DehazeFormer / MPRNet | 去除天气和成像介质带来的退化 | 物理模型、Transformer 建模 |
| 低照度增强 | RetinexNet / Zero-DCE / Retinexformer | 增强亮度、对比度并抑制噪声 | Retinex 分解、曲线估计、无参考训练 |
| 通用复原 | SwinIR / Restormer / NAFNet | 一个架构覆盖多种退化 | 高效 attention、归一化、可扩展训练 |

## 4. 入门路线
1. 先学习 PSNR、SSIM、LPIPS，明确“更高指标”和“视觉更好”可能不一致。
2. 阅读 SRCNN 和 DnCNN，理解超分和去噪中 CNN 的基本用法。
3. 阅读 ESRGAN，理解对抗训练和感知损失为什么能提升视觉细节。
4. 跑通 SwinIR 或 Restormer 的测试脚本，观察不同退化任务的输入输出差异。

## 5. 进阶路线
1. 学习 RetinexNet、Zero-DCE，理解低照度增强中的物理假设和无参考约束。
2. 学习 MPRNet、Restormer，理解多阶段复原和高分辨率 Transformer 的工程设计。
3. 学习 NAFNet，关注“去掉复杂模块后仍然有效”的轻量复原思想。
4. 进一步关注扩散模型图像复原、盲复原和真实退化建模。

## 6. 论文精读
### 6.1 超分辨率
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | Image Super-Resolution Using Deep Convolutional Networks | [paper](https://arxiv.org/abs/1501.00092) | SRCNN、端到端超分、浅层 CNN |
| 2 | Enhanced Deep Residual Networks for Single Image Super-Resolution | [paper](https://arxiv.org/abs/1707.02921) | EDSR、残差块、去 BatchNorm |
| 3 | ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks | [paper](https://arxiv.org/abs/1809.00219) | GAN、感知质量、RRDB |
| 4 | SwinIR: Image Restoration Using Swin Transformer | [paper](https://arxiv.org/abs/2108.10257) | Swin Transformer、通用图像复原 |

### 6.2 去噪 / 去模糊 / 通用复原
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising | [paper](https://arxiv.org/abs/1608.03981) | DnCNN、残差去噪、盲去噪 |
| 2 | Multi-Stage Progressive Image Restoration | [paper](https://arxiv.org/abs/2102.02808) | MPRNet、多阶段渐进复原 |
| 3 | Restormer: Efficient Transformer for High-Resolution Image Restoration | [paper](https://arxiv.org/abs/2111.09881) | 高分辨率 Transformer、高效 attention |
| 4 | Simple Baselines for Image Restoration | [paper](https://arxiv.org/abs/2204.04676) | NAFNet、简化模块、强基线思想 |

### 6.3 低照度 / 去雾
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | Deep Retinex Decomposition for Low-Light Enhancement | [paper](https://arxiv.org/abs/1808.04560) | RetinexNet、反射/照明分解 |
| 2 | Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement | [paper](https://arxiv.org/abs/2001.06826) | Zero-DCE、曲线估计、零参考约束 |
| 3 | Vision Transformers for Single Image Dehazing | [paper](https://arxiv.org/abs/2204.03883) | DehazeFormer、去雾 Transformer |
| 4 | Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement | [paper](https://arxiv.org/abs/2303.06705) | Retinex 与 Transformer 结合 |

## 7. 代码实践
| 仓库 | 覆盖方向 | 建议用途 |
| --- | --- | --- |
| [SwinIR](https://github.com/JingyunLiang/SwinIR) | 超分 / 去噪 / JPEG artifact removal | 学习 Transformer 图像复原强基线 |
| [Restormer](https://github.com/swz30/Restormer) | 去噪 / 去模糊 / 去雨 / 复原 | 学习高分辨率复原训练与测试 |
| [BasicSR](https://github.com/XPixelGroup/BasicSR) | 超分与复原工具箱 | 训练低层视觉模型的常用基础框架 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 真实场景超分 | 学习真实退化建模与工程部署 |
| [Zero-DCE](https://github.com/bsun0802/Zero-DCE) | 低照度增强 | 学习无参考增强约束 |
| [RetinexNet](https://github.com/weichen582/RetinexNet) | 低照度增强 | 学习 Retinex 分解路线 |
| [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) | 低照度增强 | 学习近年的低照度强基线 |
| [NAFNet](https://github.com/megvii-research/NAFNet) | 通用复原 | 学习简洁高效复原网络 |

## 8. 数据集获取
| 数据集 | 链接 | 适用场景 | 备注 |
| --- | --- | --- | --- |
| DIV2K | [download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | 单图像超分辨率 | SR 方向最常用数据集之一 |
| Flickr2K | [download](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) | 超分训练补充数据 | 常与 DIV2K 合并为 DF2K |
| SIDD | [download](https://abdokamel.github.io/sidd/) | 真实图像去噪 | 手机真实噪声数据集 |
| GoPro | [download](https://seungjunnah.github.io/Datasets/gopro.html) | 动态场景去模糊 | 去模糊经典 benchmark |
| LOL | [download](https://daooshee.github.io/BMVC2018website/) | 低照度增强 | RetinexNet 论文常用数据集 |
| RESIDE | [download](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) | 单图像去雾 | 去雾方向常用 benchmark |

## 9. 推荐学习顺序
`PSNR / SSIM / LPIPS → SRCNN → DnCNN → EDSR → ESRGAN → RetinexNet → Zero-DCE → SwinIR → Restormer → NAFNet`

## 10. 常见问题
- 图像增强不能只看数值指标，一定要结合视觉结果和下游任务表现。
- 真实退化通常比合成退化复杂，训练数据的退化模型非常关键。
- 感知质量强的方法可能生成不存在的细节，科研和工程中需要明确风险。
