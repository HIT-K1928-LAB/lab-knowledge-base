# 线框解析评价指标

线框解析评价关注模型是否正确检测出图像中的线段、端点和结构关系。常见输出包括预测线段、预测分数、真实线段和真实 junction。

## 1. 指标总览
| 指标 | 公式 | 含义 | 趋势 |
| --- | --- | --- | --- |
| Line Matching Distance | $d(l_p,l_g)$ | 预测线段和真实线段的几何距离 | 越低越好 |
| Precision | $\frac{TP}{TP+FP}$ | 预测线段中有多少是真线段 | 越高越好 |
| Recall | $\frac{TP}{TP+FN}$ | 真实线段中有多少被检测到 | 越高越好 |
| F1-score | $\frac{2PR}{P+R}$ | Precision 和 Recall 的平衡 | 越高越好 |
| sAP | $\int_0^1 p(r)dr$ | 按线段结构匹配计算的 AP | 越高越好 |
| Junction AP | $\int_0^1 p_j(r_j)dr_j$ | 端点或 junction 检测 AP | 越高越好 |

## 2. Line Matching Distance
### 公式
给定预测线段 $l_p=(p_1,p_2)$ 和真实线段 $l_g=(g_1,g_2)$：
$$
d(l_p,l_g)=\min\left(\frac{\|p_1-g_1\|_2+\|p_2-g_2\|_2}{2},\frac{\|p_1-g_2\|_2+\|p_2-g_1\|_2}{2}\right)
$$

### 物理含义
线段两个端点没有固定顺序，因此匹配距离需要考虑端点正向和反向两种排列，取更小值。

### 高低代表什么
- 越低越好。
- 距离低说明线段端点位置更接近真实标注。

### 伪代码
```text
distance_forward = mean(distance(pred_start, gt_start), distance(pred_end, gt_end))
distance_reverse = mean(distance(pred_start, gt_end), distance(pred_end, gt_start))
line_distance = min(distance_forward, distance_reverse)
```

### Python 实现
```python
import numpy as np


def line_distance(pred_line, gt_line):
    pred_line = np.asarray(pred_line, dtype=np.float64).reshape(2, 2)
    gt_line = np.asarray(gt_line, dtype=np.float64).reshape(2, 2)
    forward = (np.linalg.norm(pred_line[0] - gt_line[0]) + np.linalg.norm(pred_line[1] - gt_line[1])) / 2
    reverse = (np.linalg.norm(pred_line[0] - gt_line[1]) + np.linalg.norm(pred_line[1] - gt_line[0])) / 2
    return float(min(forward, reverse))


def pairwise_line_distance(pred_lines, gt_lines):
    pred_lines = np.asarray(pred_lines, dtype=np.float64).reshape(-1, 2, 2)
    gt_lines = np.asarray(gt_lines, dtype=np.float64).reshape(-1, 2, 2)
    distances = np.zeros((len(pred_lines), len(gt_lines)), dtype=np.float64)
    for i, pred in enumerate(pred_lines):
        for j, gt in enumerate(gt_lines):
            distances[i, j] = line_distance(pred, gt)
    return distances
```

## 3. Precision、Recall、F1-score
### 公式
$$
Precision = \frac{TP}{TP+FP}, \quad Recall = \frac{TP}{TP+FN}
$$

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

### 物理含义
当预测线段与某条未匹配真实线段的距离小于阈值时，记为 TP；否则为 FP。未被匹配的真实线段为 FN。

### 高低代表什么
- Precision 越高，误检线段越少。
- Recall 越高，漏检线段越少。
- F1 越高，检测质量越平衡。

### 伪代码
```text
sort predicted lines by confidence descending
for each predicted line:
    find nearest unmatched gt line
    if distance <= threshold:
        mark as true positive
    else:
        mark as false positive
false_negative = number_of_gt_lines - true_positive
compute precision, recall, f1
```

### Python 实现
```python
import numpy as np


def match_lines(pred_lines, pred_scores, gt_lines, distance_threshold=5.0):
    pred_lines = np.asarray(pred_lines, dtype=np.float64).reshape(-1, 2, 2)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_lines = np.asarray(gt_lines, dtype=np.float64).reshape(-1, 2, 2)

    order = np.argsort(-pred_scores)
    pred_lines = pred_lines[order]
    matched_gt = np.zeros(len(gt_lines), dtype=bool)
    tp = np.zeros(len(pred_lines), dtype=np.float64)
    fp = np.zeros(len(pred_lines), dtype=np.float64)

    for i, pred in enumerate(pred_lines):
        if len(gt_lines) == 0:
            fp[i] = 1
            continue
        distances = np.array([line_distance(pred, gt) for gt in gt_lines])
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= distance_threshold and not matched_gt[best_idx]:
            tp[i] = 1
            matched_gt[best_idx] = True
        else:
            fp[i] = 1

    fn = len(gt_lines) - int(tp.sum())
    precision = tp.sum() / (tp.sum() + fp.sum()) if tp.sum() + fp.sum() > 0 else 0.0
    recall = tp.sum() / len(gt_lines) if len(gt_lines) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1), int(fn)
```

## 4. sAP
### 公式
$$
sAP = \int_0^1 p(r)dr
$$

### 物理含义
sAP 是结构化线段检测中的 Average Precision。它根据预测分数排序，并用线段匹配阈值判断 TP/FP，再计算 PR 曲线面积。

### 高低代表什么
- 越高越好。
- sAP 高说明模型能把正确线段排在高置信度位置，并且整体召回较好。

### 伪代码
```text
sort predicted lines by confidence descending
mark each prediction as TP or FP using line matching threshold
compute cumulative precision and recall
sAP = area under precision-recall curve
```

### Python 实现
```python
import numpy as np


def average_precision(recalls, precisions):
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[points + 1] - recalls[points]) * precisions[points + 1]))


def structural_average_precision(pred_lines, pred_scores, gt_lines, distance_threshold=5.0):
    pred_lines = np.asarray(pred_lines, dtype=np.float64).reshape(-1, 2, 2)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_lines = np.asarray(gt_lines, dtype=np.float64).reshape(-1, 2, 2)
    order = np.argsort(-pred_scores)
    pred_lines = pred_lines[order]

    matched_gt = np.zeros(len(gt_lines), dtype=bool)
    tp = np.zeros(len(pred_lines), dtype=np.float64)
    fp = np.zeros(len(pred_lines), dtype=np.float64)

    for i, pred in enumerate(pred_lines):
        if len(gt_lines) == 0:
            fp[i] = 1
            continue
        distances = np.array([line_distance(pred, gt) for gt in gt_lines])
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= distance_threshold and not matched_gt[best_idx]:
            tp[i] = 1
            matched_gt[best_idx] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / len(gt_lines) if len(gt_lines) > 0 else np.zeros_like(cum_tp)
    precisions = np.divide(cum_tp, cum_tp + cum_fp, out=np.zeros_like(cum_tp), where=(cum_tp + cum_fp) > 0)
    return average_precision(recalls, precisions)
```

## 5. Junction AP
### 公式
端点距离为：
$$
d(p,g)=\|p-g\|_2
$$

Junction AP 与 sAP 类似，只是匹配对象从线段变为点。

### 物理含义
Junction AP 衡量端点或交点检测是否准确。线框解析中 junction 质量会直接影响线段组合质量。

### 高低代表什么
- 越高越好。
- Junction AP 高说明端点定位和置信度排序较好。

### 伪代码
```text
sort predicted junctions by score
for each predicted junction:
    find nearest unmatched gt junction
    if point distance <= threshold:
        mark as TP
    else:
        mark as FP
compute AP from cumulative precision and recall
```

### Python 实现
```python
import numpy as np


def point_average_precision(pred_points, pred_scores, gt_points, distance_threshold=3.0):
    pred_points = np.asarray(pred_points, dtype=np.float64).reshape(-1, 2)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(-pred_scores)
    pred_points = pred_points[order]

    matched_gt = np.zeros(len(gt_points), dtype=bool)
    tp = np.zeros(len(pred_points), dtype=np.float64)
    fp = np.zeros(len(pred_points), dtype=np.float64)

    for i, pred in enumerate(pred_points):
        if len(gt_points) == 0:
            fp[i] = 1
            continue
        distances = np.linalg.norm(gt_points - pred, axis=1)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= distance_threshold and not matched_gt[best_idx]:
            tp[i] = 1
            matched_gt[best_idx] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / len(gt_points) if len(gt_points) > 0 else np.zeros_like(cum_tp)
    precisions = np.divide(cum_tp, cum_tp + cum_fp, out=np.zeros_like(cum_tp), where=(cum_tp + cum_fp) > 0)
    return average_precision(recalls, precisions)
```

## 6. 使用建议
- 线框解析建议同时报告 sAP、F1 和可视化结果。
- 不同论文的线段距离阈值可能不同，比较结果时必须确认评估协议。
- 如果线段用于后续定位或建图，还应额外评估匹配成功率和几何一致性。

