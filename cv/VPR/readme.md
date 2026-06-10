# VPR（Visual Place Recognition）

VPR，即视觉地点识别，目标是根据查询图像在数据库中检索同一地点或相近地点。它是视觉定位、SLAM 回环检测、机器人导航和地理定位中的关键模块。

## 1. 学习目标
- 理解地点识别与图像检索、视觉定位、回环检测之间的关系。
- 掌握全局描述子、局部特征、重排序、几何验证等核心组件。
- 理解召回率 Recall@K、mAP、检索库规模和正负样本定义。
- 掌握 NetVLAD、GeM、DELG、CosPlace、MixVPR 等代表方法的训练范式。
- 能够用现有模型在 Oxford / Pitts / MSLS 等数据集上进行检索实验。

## 2. 发展脉络
- 早期 VPR 以 Bag-of-Words、FAB-MAP、局部特征和概率建图为主，重点解决外观变化和回环检测。
- 2015 年后，NetVLAD 将 CNN 特征和 VLAD 聚合结合，成为现代深度 VPR 的重要起点。
- 随后，GeM、DELF/DELG 等方法从全局检索和局部重排序两个角度提升精度。
- 2022 年前后，CosPlace 把大规模地点识别转化为分类代理任务，提高训练效率和可扩展性。
- 2023 年以后，MixVPR、EigenPlaces、AnyLoc 等方法进一步关注特征混合、视角鲁棒和基础模型迁移。
- 当前趋势包括视觉基础模型、跨季节/跨天气泛化、长时序检索、多模态地理定位和几何验证融合。

## 3. 方法地图
| 类别 | 代表方法 | 核心思想 | 适合重点理解 |
| --- | --- | --- | --- |
| 传统检索 | BoW / FAB-MAP | 局部特征量化成视觉词袋，并用概率模型识别地点 | 早期 VPR 与回环检测 |
| 全局描述子 | NetVLAD / GeM / AP-GeM | 将整图编码为一个可检索向量 | 聚合层、度量学习、Recall@K |
| 局部+全局 | DELF / DELG / SuperPoint + HLoc | 先全局召回，再局部匹配或几何验证 | rerank、pose estimation |
| 分类代理 | CosPlace / EigenPlaces | 将地理区域划分为类别，用分类损失学表征 | 大规模训练、类别构造 |
| 特征混合 | MixVPR | 对 backbone 特征进行混合，构建强全局描述子 | 简洁强基线、feature mixing |
| 基础模型 | AnyLoc / DINOv2-based VPR | 利用通用视觉表征做零样本或弱监督地点识别 | 泛化性、无需专门训练 |
| 序列方法 | SeqSLAM / 时序 VPR | 使用连续帧和轨迹一致性提升鲁棒性 | 长时序匹配、动态环境 |

## 4. 入门路线
1. 先学习图像检索基础：embedding、距离度量、Top-K、Recall@K、mAP。
2. 阅读 FAB-MAP 和 NetVLAD，理解从视觉词袋到深度聚合的变化。
3. 跑通一个 NetVLAD / CosPlace 模型，在小规模数据库上完成 query-to-database 检索。
4. 学习局部特征匹配和几何验证，理解为什么 VPR 常和定位 pipeline 结合。

## 5. 进阶路线
1. 学习 DELG 和 Hierarchical Localization，理解全局召回 + 局部匹配 + 位姿估计的完整链路。
2. 学习 CosPlace、EigenPlaces，理解分类代理如何构造地点类别。
3. 学习 MixVPR 和 AnyLoc，比较专门训练模型与通用基础模型的优缺点。
4. 进一步关注跨季节、跨天气、昼夜变化、动态物体干扰下的泛化问题。

## 6. 论文精读
| 顺序 | 论文 | 链接 | 精读重点 |
| --- | --- | --- | --- |
| 1 | FAB-MAP: Probabilistic Localization and Mapping in the Space of Appearance | [paper](https://doi.org/10.1177/0278364908090961) | 视觉词袋、概率建图、回环检测 |
| 2 | NetVLAD: CNN architecture for weakly supervised place recognition | [paper](https://arxiv.org/abs/1511.07247) | VLAD 聚合、弱监督、深度 VPR 起点 |
| 3 | Fine-tuning CNN Image Retrieval with No Human Annotation | [paper](https://arxiv.org/abs/1711.02512) | GeM、检索微调、无人工标注 |
| 4 | Unifying Deep Local and Global Features for Image Search | [paper](https://arxiv.org/abs/2001.05027) | DELG、全局检索与局部特征统一 |
| 5 | Rethinking Visual Geo-localization for Large-Scale Applications | [paper](https://arxiv.org/abs/2204.02287) | CosPlace、分类代理、大规模训练 |
| 6 | MixVPR: Feature Mixing for Visual Place Recognition | [paper](https://arxiv.org/abs/2303.02190) | 特征混合、简洁强基线 |
| 7 | EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition | [paper](https://arxiv.org/abs/2308.10832) | 视角鲁棒、地点划分策略 |
| 8 | AnyLoc: Towards Universal Visual Place Recognition | [paper](https://arxiv.org/abs/2308.00688) | 基础模型特征、通用 VPR |

## 7. 代码实践
| 仓库 | 覆盖方向 | 建议用途 |
| --- | --- | --- |
| [netvlad](https://github.com/Relja/netvlad) | NetVLAD | 阅读经典官方实现 |
| [CosPlace](https://github.com/gmberton/CosPlace) | CosPlace | 学习分类代理训练范式 |
| [EigenPlaces](https://github.com/gmberton/EigenPlaces) | EigenPlaces | 学习视角鲁棒地点识别 |
| [MixVPR](https://github.com/amaralibey/MixVPR) | MixVPR | 学习现代 VPR 简洁强基线 |
| [AnyLoc](https://github.com/AnyLoc/AnyLoc) | AnyLoc | 学习基于基础模型的 VPR |
| [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) | 检索 + 匹配 + 定位 | 把 VPR 串到 6-DoF 定位中 |
| [tensorflow/models/research/delf](https://github.com/tensorflow/models/tree/master/research/delf) | DELF / DELG | 学习局部与全局检索特征 |

## 8. 推荐学习顺序
`图像检索基础 → FAB-MAP → NetVLAD → GeM → DELG → CosPlace → MixVPR → EigenPlaces → AnyLoc → HLoc`

## 9. 常见问题
- VPR 的正负样本定义和地理距离阈值会显著影响指标，不同论文不能只看数字直接比较。
- 高 Recall@1 不一定代表能完成精确定位，定位还需要局部匹配和几何验证。
- 跨季节、跨昼夜、跨视角是 VPR 的主要难点，数据集选择非常重要。

