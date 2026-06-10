# 线框解析

线框解析（wireframe parsing）和线段检测（line segment detection）关注从图像中恢复线段、端点、交点以及结构化几何关系。它常用于建筑理解、室内场景建模、视觉定位、SLAM、三维重建和几何约束建模。

## 1. 学习目标
- 理解边缘、线段、端点、junction、wireframe 之间的关系。
- 掌握传统线段检测和深度线框解析的主要差异。
- 理解 precision / recall、sAP、line matching 等评价方式。
- 能够区分“检测线段”和“解析结构化线框”的任务目标。
- 能够使用 HAWP、LETR、DeepLSD 等仓库进行测试和可视化。

## 2. 发展脉络
- 传统阶段以 Hough Transform、LSD、EDLines 等为代表，强调边缘响应、梯度方向和几何一致性。
- 2019 年，L-CNN 将 wireframe parsing 建模为端到端学习问题，同时预测 junction 和 line proposal。
- 2020 年，HAWP 提出 holistic attraction field，用几何场统一描述线段与端点吸引关系。
- 2021 年后，LETR、F-Clip 等方法探索 Transformer 或更简洁的全卷积表示，减少复杂后处理。
- 2022 年后，DeepLSD 等方法进一步结合深度图像梯度和 refinement，提高真实场景线段质量。
- 当前趋势包括与 3D 重建、语义理解、结构化场景解析、定位和地图构建结合。

## 3. 方法地图
| 类别 | 代表方法 | 核心思想 | 适合重点理解 |
| --- | --- | --- | --- |
| 传统线段检测 | Hough / LSD / EDLines | 基于边缘、梯度和几何规则检测线段 | 梯度方向、阈值、几何约束 |
| Junction-based | L-CNN | 先预测 junction，再组合成 line proposal | 点线关系、结构解析 |
| Attraction field | HAWP | 用吸引场编码像素到线段的几何关系 | 线段表示学习、端点吸引 |
| Transformer line detection | LETR | 用 Transformer 直接预测线段集合 | 端到端预测、query 表示 |
| 全卷积 line parsing | F-Clip | 使用更紧凑的全卷积结构做线段解析 | 简洁结构、速度与精度平衡 |
| 深度梯度 refinement | DeepLSD | 结合深度梯度估计和线段细化 | 低层几何与深度学习结合 |
| 多视图/3D 扩展 | Line mapping / wireframe reconstruction | 将 2D 线段用于 3D 重建或定位 | 几何验证、三维约束 |

## 4. 入门路线
1. 先学习边缘检测和 Hough 变换，理解线段检测为什么可以从梯度开始。
2. 阅读 LSD，掌握传统线段检测的输入、输出和 false detection control。
3. 阅读 L-CNN 和 HAWP，理解深度线框解析如何预测 junction 和 line。
4. 使用 HAWP 或 DeepLSD 对室内/建筑图像做可视化，观察误检、漏检和断裂问题。

## 5. 进阶路线
1. 学习 LETR，理解如何把线段检测表述为集合预测问题。
2. 学习 F-Clip，理解全卷积解析在结构复杂度和效果上的平衡。
3. 学习 DeepLSD，关注深度图像梯度、线段 refinement 和传统几何结合。
4. 进一步阅读线段辅助视觉定位、Manhattan world、结构化 3D 重建等方向。

## 6. 论文精读
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | LSD: A Fast Line Segment Detector with a False Detection Control | [paper](https://www.ipol.im/pub/art/2012/gjmr-lsd/) | 传统线段检测、梯度与误检控制 |
| 2 | End-to-End Wireframe Parsing | [paper](https://arxiv.org/abs/1905.03246) | L-CNN、junction proposal、line proposal |
| 3 | Holistically-Attracted Wireframe Parsing | [paper](https://arxiv.org/abs/2003.01663) | HAWP、attraction field、结构化线框 |
| 4 | Line Segment Detection Using Transformers without Edges | [paper](https://arxiv.org/abs/2101.01909) | LETR、Transformer、端到端线段集合预测 |
| 5 | Fully Convolutional Line Parsing | [paper](https://arxiv.org/abs/2104.11207) | F-Clip、全卷积线段解析 |
| 6 | DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients | [paper](https://arxiv.org/abs/2212.07766) | 深度梯度、线段检测与细化 |

## 7. 代码实践
| 仓库 | 覆盖方向 | 建议用途 |
| --- | --- | --- |
| [lsd](https://github.com/centreborelli/lsd) | 传统线段检测 | 学习传统 LSD 输入输出 |
| [lcnn](https://github.com/zhou13/lcnn) | Wireframe parsing | 阅读 L-CNN 官方实现 |
| [hawp](https://github.com/cherubicXN/hawp) | HAWP | 学习 attraction field 表示 |
| [LETR](https://github.com/mlpc-ucsd/LETR) | Transformer line detection | 学习端到端线段检测 |
| [F-Clip](https://github.com/Delay-Xili/F-Clip) | 全卷积 line parsing | 学习简洁高效线框解析 |
| [DeepLSD](https://github.com/cvg/DeepLSD) | 线段检测与 refinement | 学习现代线段检测强基线 |
| [TP-LSD](https://github.com/Siyuada7/TP-LSD) | 线段检测 | 补充参考实现 |

## 8. 数据集获取
| 数据集 | 链接 | 适用场景 | 备注 |
| --- | --- | --- | --- |
| Wireframe Dataset | [download](https://github.com/zhou13/lcnn) | 线框解析 | L-CNN 仓库提供数据准备说明 |
| HAWP Dataset | [download](https://github.com/cherubicXN/hawp) | 线框解析训练与评估 | HAWP 仓库提供数据链接和格式说明 |
| York Urban Line Segment Database | [download](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) | 城市场景线段检测 | 传统线段检测常用数据集 |
| Wireframe on Hugging Face | [download](https://huggingface.co/datasets/lh9171338/Wireframe) | 快速获取 wireframe 数据 | 适合快速下载和实验 |

## 9. 推荐学习顺序
`边缘检测 → Hough Transform → LSD → L-CNN → HAWP → LETR → F-Clip → DeepLSD → 线段辅助定位 / 重建`

## 10. 常见问题
- 线段检测结果的视觉质量很重要，数值指标可能无法完全反映结构连续性。
- 室内、建筑和自然场景的线段分布差异很大，跨域泛化需要单独评估。
- 如果用于定位或重建，还需要关注线段匹配和几何一致性，而不只是检测本身。
