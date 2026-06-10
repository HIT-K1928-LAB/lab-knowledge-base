# 目标检测评价指标

目标检测评价同时考察类别是否正确、边界框是否准确、置信度排序是否合理。常见输入包括预测框、预测类别、预测分数和真实框。

## 1. 指标总览
| 指标 | 公式 | 含义 | 趋势 |
| --- | --- | --- | --- |
| IoU | $\frac{\lvert B_p \cap B_g \rvert}{\lvert B_p \cup B_g \rvert}$ | 预测框和真实框的重叠程度 | 越高越好 |
| Precision | $\frac{TP}{TP+FP}$ | 预测出的目标中有多少是真的 | 越高越好 |
| Recall | $\frac{TP}{TP+FN}$ | 真实目标中有多少被检测到 | 越高越好 |
| AP | $\int_0^1 p(r)dr$ | 单类别 PR 曲线下的面积 | 越高越好 |
| mAP | $\frac{1}{C}\sum_c AP_c$ | 多类别 AP 的平均值 | 越高越好 |
| F1-score | $\frac{2PR}{P+R}$ | Precision 与 Recall 的平衡 | 越高越好 |

## 2. IoU
### 公式
$$
IoU(B_p,B_g)=\frac{Area(B_p \cap B_g)}{Area(B_p \cup B_g)}
$$

### 物理含义
IoU 衡量预测框和真实框的重叠程度。检测评估通常设置 IoU 阈值，例如 IoU ≥ 0.5 才算匹配成功。

### 高低代表什么
- 越高越好，最大值为 1。
- IoU 低说明定位不准确，即使类别正确也可能被判为错误检测。

### 伪代码
```text
intersection = overlap_area(pred_box, gt_box)
union = area(pred_box) + area(gt_box) - intersection
iou = intersection / union
```

### Python 实现
```python
import numpy as np


def box_iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
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
- TP：预测框类别正确，且与某个未匹配真实框的 IoU 达到阈值。
- FP：预测框没有匹配到真实框，或类别错误，或重复检测。
- FN：真实目标没有被任何预测框匹配。

### 高低代表什么
- Precision 越高，误检越少。
- Recall 越高，漏检越少。
- F1 越高，说明误检和漏检之间更平衡。

### 伪代码
```text
sort predictions by confidence descending
for each prediction:
    find unmatched ground-truth box with largest IoU and same class
    if best_iou >= threshold:
        mark as true positive
    else:
        mark as false positive
false_negative = number_of_gt - number_of_true_positive
compute precision, recall, f1
```

### Python 实现
```python
import numpy as np


def match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=np.float64)
    fp = np.zeros(len(pred_boxes), dtype=np.float64)

    for i, pred_box in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou(pred_box[None, :], gt_boxes)[0]
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_threshold and not matched_gt[best_idx]:
            tp[i] = 1
            matched_gt[best_idx] = True
        else:
            fp[i] = 1

    fn = len(gt_boxes) - int(tp.sum())
    precision = tp.sum() / (tp.sum() + fp.sum()) if tp.sum() + fp.sum() > 0 else 0.0
    recall = tp.sum() / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1), int(fn)
```

## 4. AP 与 mAP
### 公式
$$
AP = \int_0^1 p(r)dr
$$

$$
mAP = \frac{1}{C}\sum_{c=1}^{C}AP_c
$$

### 物理含义
AP 是单个类别在不同置信度阈值下 Precision-Recall 曲线的面积。mAP 是多个类别 AP 的平均值。

### 高低代表什么
- 越高越好，最大值为 1。
- AP 高说明模型既能把目标排在高置信度位置，又能保持较好的召回。

### 伪代码
```text
sort predictions by confidence descending
mark each prediction as TP or FP
cumulative_tp = cumulative sum of TP
cumulative_fp = cumulative sum of FP
precision = cumulative_tp / (cumulative_tp + cumulative_fp)
recall = cumulative_tp / number_of_gt
ap = area under precision-recall curve
mAP = mean(AP over classes)
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

    changing_points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[changing_points + 1] - recalls[changing_points]) * precisions[changing_points + 1]))


def ap_for_one_class(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=np.float64)
    fp = np.zeros(len(pred_boxes), dtype=np.float64)

    for i, pred_box in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou(pred_box[None, :], gt_boxes)[0]
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_threshold and not matched_gt[best_idx]:
            tp[i] = 1
            matched_gt[best_idx] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(cum_tp)
    precisions = np.divide(cum_tp, cum_tp + cum_fp, out=np.zeros_like(cum_tp), where=(cum_tp + cum_fp) > 0)
    return average_precision(recalls, precisions)
```

## 5. 使用建议
- VOC 常看 `AP@0.5`，COCO 常看 `AP@[0.5:0.95]`。
- 小目标检测要额外关注按目标面积划分的 AP，如 AP-small、AP-medium、AP-large。
- 模型部署时不仅要看 mAP，也要看 FPS、延迟、显存和模型大小。

