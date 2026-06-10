# VPR 评价指标

VPR（Visual Place Recognition）评价关注查询图像能否从数据库中检索到同一地点或足够接近的地点。常见输入包括查询特征、数据库特征、地理坐标和正样本列表。

## 1. 指标总览
| 指标 | 公式 | 含义 | 趋势 |
| --- | --- | --- | --- |
| Recall@K | $\frac{1}{Q}\sum_q \mathbb{1}(P_q \cap TopK_q \neq \emptyset)$ | 前 K 个检索结果中是否至少有一个正确地点 | 越高越好 |
| Precision@K | $\frac{\lvert P_q \cap TopK_q \rvert}{K}$ | 前 K 个结果中正确结果的比例 | 越高越好 |
| AP | $\sum_k P(k)\Delta R(k)$ | 单个 query 的检索 PR 曲线面积 | 越高越好 |
| mAP | $\frac{1}{Q}\sum_q AP_q$ | 多个 query 的 AP 平均 | 越高越好 |
| Top-1 Distance Error | $d(q, r_1)$ | 第 1 个检索结果与 query 的地理距离 | 越低越好 |

## 2. Recall@K
### 公式
$$
Recall@K = \frac{1}{Q}\sum_{q=1}^{Q}\mathbb{1}(P_q \cap TopK_q \neq \emptyset)
$$

其中 $P_q$ 是 query 的正样本集合，$TopK_q$ 是前 K 个检索结果。

### 物理含义
Recall@K 表示每个查询在前 K 个候选结果里能否找到至少一个正确地点。VPR 最常用 Recall@1、Recall@5、Recall@10。

### 高低代表什么
- 越高越好。
- Recall@1 高说明模型可以直接给出可靠的首位地点候选。
- Recall@K 高但 Recall@1 低说明候选召回可以，但排序还不够好。

### 伪代码
```text
for each query:
    rank database images by feature distance
    take top k results
    if any result is in positive set:
        hit += 1
recall_at_k = hit / number_of_queries
```

### Python 实现
```python
import numpy as np


def pairwise_l2(query_features, db_features):
    query_features = np.asarray(query_features, dtype=np.float64)
    db_features = np.asarray(db_features, dtype=np.float64)
    q2 = np.sum(query_features ** 2, axis=1, keepdims=True)
    d2 = np.sum(db_features ** 2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(q2 + d2 - 2 * query_features @ db_features.T, 0.0))


def recall_at_k(query_features, db_features, positive_indices, ks=(1, 5, 10)):
    distances = pairwise_l2(query_features, db_features)
    ranking = np.argsort(distances, axis=1)
    results = {}
    for k in ks:
        hits = 0
        for q_idx, positives in enumerate(positive_indices):
            positives = set(positives)
            top_k = set(ranking[q_idx, :k])
            hits += int(len(positives & top_k) > 0)
        results[k] = hits / len(positive_indices) if len(positive_indices) > 0 else 0.0
    return results
```

## 3. Precision@K
### 公式
$$
Precision@K = \frac{1}{Q}\sum_{q=1}^{Q}\frac{|P_q \cap TopK_q|}{K}
$$

### 物理含义
Precision@K 表示前 K 个检索结果中正确地点的比例。它比 Recall@K 更关注候选列表整体纯度。

### 高低代表什么
- 越高越好。
- Precision@K 高说明前 K 个候选中错误地点少。

### 伪代码
```text
for each query:
    get top k retrieved database images
    precision = number of positive results in top k / k
average precision over queries
```

### Python 实现
```python
import numpy as np


def precision_at_k(query_features, db_features, positive_indices, k=10):
    distances = pairwise_l2(query_features, db_features)
    ranking = np.argsort(distances, axis=1)
    precisions = []
    for q_idx, positives in enumerate(positive_indices):
        positives = set(positives)
        top_k = ranking[q_idx, :k]
        precisions.append(sum(int(idx in positives) for idx in top_k) / k)
    return float(np.mean(precisions)) if precisions else 0.0
```

## 4. AP 与 mAP
### 公式
$$
AP_q = \frac{1}{|P_q|}\sum_{k=1}^{N} Precision@k \cdot rel(k)
$$

$$
mAP = \frac{1}{Q}\sum_{q=1}^{Q} AP_q
$$

### 物理含义
AP 衡量所有正确地点在完整排序列表中的位置。越靠前出现正确地点，AP 越高。

### 高低代表什么
- 越高越好。
- mAP 高说明检索排序整体质量好，而不仅仅是 Top-1 好。

### 伪代码
```text
for each query:
    rank database images
    for each rank position:
        if retrieved image is positive:
            accumulate precision at this rank
    AP = accumulated precision / number of positives
mAP = mean AP over queries
```

### Python 实现
```python
import numpy as np


def average_precision_from_ranking(ranking, positives):
    positives = set(positives)
    if not positives:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, db_idx in enumerate(ranking, start=1):
        if db_idx in positives:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / len(positives)


def mean_average_precision(query_features, db_features, positive_indices):
    distances = pairwise_l2(query_features, db_features)
    rankings = np.argsort(distances, axis=1)
    aps = [average_precision_from_ranking(rankings[i], positives) for i, positives in enumerate(positive_indices)]
    return float(np.mean(aps)) if aps else 0.0
```

## 5. Top-1 Distance Error
### 公式
$$
Error@1_q = d(g_q, g_{r_1})
$$

其中 $g_q$ 是 query GPS 坐标，$g_{r_1}$ 是 Top-1 检索结果 GPS 坐标。

### 物理含义
该指标直接衡量首位检索结果在地理空间上离查询点有多远。

### 高低代表什么
- 越低越好。
- 对实际定位系统来说，比纯 Recall 更直观。

### 伪代码
```text
rank database by feature distance
take top-1 database image
compute geographic distance between query coordinate and top-1 coordinate
average over queries
```

### Python 实现
```python
import numpy as np


def haversine_distance_meters(latlon1, latlon2):
    latlon1 = np.radians(np.asarray(latlon1, dtype=np.float64))
    latlon2 = np.radians(np.asarray(latlon2, dtype=np.float64))
    lat1, lon1 = latlon1[..., 0], latlon1[..., 1]
    lat2, lon2 = latlon2[..., 0], latlon2[..., 1]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def top1_distance_error(query_features, db_features, query_latlon, db_latlon):
    distances = pairwise_l2(query_features, db_features)
    top1 = np.argmin(distances, axis=1)
    geo_errors = haversine_distance_meters(np.asarray(query_latlon), np.asarray(db_latlon)[top1])
    return float(np.mean(geo_errors))
```

## 6. 使用建议
- VPR 论文最常报告 Recall@1、Recall@5、Recall@10。
- 如果一个 query 有多个正样本，需要明确正样本半径阈值，例如 25m 或 10m。
- 实际定位系统建议同时报告 Recall@K 和几何定位误差。

