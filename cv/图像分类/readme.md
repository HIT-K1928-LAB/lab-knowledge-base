# 图像分类

图像分类是计算机视觉最基础的任务之一：给定一张图像，预测其所属类别。它既是理解 CNN、Transformer、预训练模型的入口，也是目标检测、分割、检索、视觉定位等任务的 backbone 基础。

## 1. 学习目标
- 理解从手工特征到深度神经网络的范式变化。
- 掌握 CNN 主干网络的核心设计：卷积、池化、归一化、残差连接、多分支结构。
- 理解模型规模、深度、宽度、分辨率与训练策略之间的关系。
- 理解 Vision Transformer 如何把图像转化为 token 序列，并替代或补充 CNN。
- 能够阅读经典分类论文，并使用主流代码库复现实验或迁移到下游任务。

## 2. 发展脉络
- 1998 年左右，LeNet 将卷积、池化和端到端训练用于手写字符识别，形成 CNN 的雏形。
- 2012 年，AlexNet 在 ImageNet 上大幅提升性能，ReLU、Dropout、GPU 训练和数据增强成为深度视觉的关键组件。
- 2014—2015 年，VGG、GoogLeNet、ResNet 分别代表了“规则堆叠小卷积”“多尺度分支”“残差学习”三条重要路线。
- 2016—2019 年，DenseNet、MobileNet、EfficientNet 等方法开始系统研究特征复用、轻量化和模型缩放。
- 2020 年以后，ViT、DeiT、Swin Transformer 把注意力机制引入分类主干，大规模预训练成为分类模型的重要前提。
- 近几年，ConvNeXt、MAE、DINO、CLIP 等工作让分类 backbone 与自监督、视觉语言预训练、通用表征学习紧密结合。

## 3. 方法地图
| 类别 | 代表方法 | 核心思想 | 适合重点理解 |
| --- | --- | --- | --- |
| 传统特征 | SIFT / HOG / LBP + SVM | 人工设计局部或梯度特征，再训练浅层分类器 | 深度学习之前的特征工程思想 |
| 经典 CNN | LeNet / AlexNet / VGG | 卷积层逐级提取局部到全局特征 | CNN 基本组件与层级特征 |
| 多分支网络 | GoogLeNet / Inception | 多尺度卷积分支并行提取信息 | 感受野、多尺度设计 |
| 残差网络 | ResNet / ResNeXt / DenseNet | 通过残差或密集连接缓解深层网络训练困难 | 深层网络优化 |
| 轻量网络 | MobileNet / ShuffleNet / EfficientNet | 深度可分离卷积、通道重排、复合缩放 | 移动端与高效模型设计 |
| Transformer | ViT / DeiT / Swin | 把图像切成 patch token，用注意力建模全局关系 | token 化、注意力、预训练 |
| 大规模预训练 | MAE / DINO / CLIP | 通过自监督或图文对比学习通用视觉表征 | 下游迁移与基础模型 |

## 4. 入门路线
1. 先学习 LeNet、AlexNet、VGG，理解卷积网络的基本结构和训练技巧。
2. 再学习 GoogLeNet 和 ResNet，理解多尺度结构与残差连接为什么重要。
3. 使用 `torchvision` 或 `timm` 跑通 CIFAR-10 / ImageNet 子集训练，熟悉数据增强、优化器、学习率调度。
4. 观察模型从浅层到深层的特征可视化，建立“层级特征”的直觉。

## 5. 进阶路线
1. 阅读 EfficientNet、MobileNet，理解精度、参数量、FLOPs、延迟之间的差异。
2. 阅读 ViT、Swin Transformer，比较 CNN 的局部归纳偏置和 Transformer 的全局建模能力。
3. 阅读 ConvNeXt，理解现代 CNN 如何吸收 Transformer 的训练和结构经验。
4. 进一步学习 MAE、DINO、CLIP，把分类模型放入大规模预训练与下游迁移的背景下理解。

## 6. 论文精读
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | Gradient-Based Learning Applied to Document Recognition | [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) | LeNet-5、卷积/池化/端到端训练 |
| 2 | ImageNet Classification with Deep Convolutional Neural Networks | [paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | AlexNet、ReLU、Dropout、GPU 训练、数据增强 |
| 3 | Very Deep Convolutional Networks for Large-Scale Image Recognition | [paper](https://arxiv.org/abs/1409.1556) | VGG、3×3 卷积堆叠、深度与表达能力 |
| 4 | Going Deeper with Convolutions | [paper](https://arxiv.org/abs/1409.4842) | Inception、多尺度分支、计算量控制 |
| 5 | Deep Residual Learning for Image Recognition | [paper](https://arxiv.org/abs/1512.03385) | ResNet、残差连接、退化问题 |
| 6 | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | [paper](https://arxiv.org/abs/1905.11946) | 复合缩放、效率和精度平衡 |
| 7 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | [paper](https://arxiv.org/abs/2010.11929) | patch embedding、class token、预训练规模 |
| 8 | A ConvNet for the 2020s | [paper](https://arxiv.org/abs/2201.03545) | ConvNeXt、现代 CNN 设计回归 |

## 7. 代码实践
| 仓库 | 覆盖方向 | 建议用途 |
| --- | --- | --- |
| [torchvision](https://github.com/pytorch/vision) | AlexNet / VGG / ResNet / Inception | 学习标准模型定义和预训练权重调用 |
| [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 大量 CNN / Transformer backbone | 快速比较不同分类模型 |
| [vision_transformer](https://github.com/google-research/vision_transformer) | ViT | 阅读官方 ViT 训练与模型配置 |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | ConvNeXt | 理解现代 CNN 训练配置 |
| [MAE](https://github.com/facebookresearch/mae) | 自监督预训练 | 从分类扩展到表征学习 |

## 8. 数据集获取
| 数据集 | 链接 | 适用场景 | 备注 |
| --- | --- | --- | --- |
| MNIST | [download](http://yann.lecun.com/exdb/mnist/) | 手写数字分类入门 | 适合验证训练流程 |
| CIFAR-10 / CIFAR-100 | [download](https://www.cs.toronto.edu/~kriz/cifar.html) | 小规模自然图像分类 | 适合课堂实验和快速 baseline |
| ImageNet | [download](https://image-net.org/download.php) | 大规模通用图像分类 | 需要注册并遵守数据许可 |
| Places365 | [download](http://places2.csail.mit.edu/download.html) | 场景分类 | 适合理解场景识别和分类迁移 |

## 9. 推荐学习顺序
`LeNet → AlexNet → VGG → GoogLeNet → ResNet → EfficientNet → ViT → ConvNeXt → MAE / CLIP`

## 10. 常见问题
- 分类论文不只要看模型结构，也要看训练策略、数据增强、正则化和评估协议。
- 论文里的参数量、FLOPs 和真实推理速度不是同一个概念，做工程实验时需要分别统计。
- 下游任务常用分类 backbone，因此理解分类网络能直接帮助学习检测、分割和检索。
