# 语义分割评价指标

语义分割评价关注每个像素的预测类别是否正确。常见输入是预测 mask 和真实 mask，二者形状通常为 `[H, W]` 或 `[N, H, W]`。

## 1. 指标总览
| 指标 | 公式 | 含义 | 趋势 |
| --- | --- | --- | --- |
| Pixel Accuracy | $\frac{\sum_i n_{ii}}{\sum_i\sum_j n_{ij}}$ | 全部像素中预测正确的比例 | 越高越好 |
| Mean Accuracy | $\frac{1}{C}\sum_i\frac{n_{ii}}{\sum_j n_{ij}}$ | 每类像素准确率的平均值 | 越高越好 |
| IoU | $\frac{TP}{TP+FP+FN}$ | 单个类别预测区域与真实区域的重叠程度 | 越高越好 |
| mIoU | $\frac{1}{C}\sum_c IoU_c$ | 所有类别 IoU 的平均值 | 越高越好 |
| Dice | $\frac{2TP}{2TP+FP+FN}$ | 预测区域与真实区域的重合程度 | 越高越好 |
| Frequency Weighted IoU | $\sum_c f_c IoU_c$ | 按类别像素频率加权的 IoU | 越高越好 |

## 2. Confusion Matrix
### 公式
$$
n_{ij}=\#(y=i,\hat{y}=j)
$$

### 物理含义
分割任务中的混淆矩阵统计真实类别为 `i` 的像素被预测为 `j` 的数量。多数指标都可以由它计算得到。

### 伪代码
```text
initialize matrix C with shape [num_classes, num_classes]
for each pixel:
    if pixel is not ignored:
        C[true_class, predicted_class] += 1
```

### Python 实现
```python
import numpy as np


def segmentation_confusion_matrix(y_true, y_pred, num_classes, ignore_index=None):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    mask = (y_true >= 0) & (y_true < num_classes)
    if ignore_index is not None:
        mask &= y_true != ignore_index
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    valid = (y_pred >= 0) & (y_pred < num_classes)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    indices = num_classes * y_true + y_pred
    return np.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
```

## 3. Pixel Accuracy 与 Mean Accuracy
### 公式
$$
Pixel\ Accuracy = \frac{\sum_i n_{ii}}{\sum_i\sum_j n_{ij}}
$$

$$
Mean\ Accuracy = \frac{1}{C}\sum_i\frac{n_{ii}}{\sum_j n_{ij}}
$$

### 物理含义
- Pixel Accuracy：整体像素分类正确率。
- Mean Accuracy：每个类别先算准确率，再对类别求平均，能缓解大类像素数量过多的影响。

### 高低代表什么
- 越高越好。
- Pixel Accuracy 高但 mIoU 低，通常说明背景或大类预测好，小类预测差。

### 伪代码
```text
cm = confusion matrix
pixel_accuracy = sum(diagonal(cm)) / sum(cm)
class_accuracy = diagonal(cm) / row_sum(cm)
mean_accuracy = mean(valid class_accuracy)
```

### Python 实现
```python
import numpy as np


def pixel_accuracy(cm):
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total > 0 else 0.0


def mean_accuracy(cm):
    diag = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    acc = np.divide(diag, row_sum, out=np.full_like(diag, np.nan), where=row_sum > 0)
    return float(np.nanmean(acc))
```

## 4. IoU、mIoU、Frequency Weighted IoU
### 公式
$$
IoU_c = \frac{n_{cc}}{\sum_j n_{cj} + \sum_i n_{ic} - n_{cc}}
$$

$$
mIoU = \frac{1}{C}\sum_c IoU_c
$$

$$
FWIoU = \sum_c \frac{\sum_j n_{cj}}{\sum_i\sum_j n_{ij}} IoU_c
$$

### 物理含义
IoU 衡量预测区域和真实区域的交并比。mIoU 是语义分割最常用主指标。

### 高低代表什么
- 越高越好。
- mIoU 高说明模型在多数类别上都有较好的区域重叠。

### 伪代码
```text
for each class:
    intersection = cm[class, class]
    union = row_sum[class] + col_sum[class] - intersection
    iou[class] = intersection / union
miou = mean(valid iou)
fwiou = sum(class_frequency * iou)
```

### Python 实现
```python
import numpy as np


def intersection_over_union(cm):
    cm = cm.astype(np.float64)
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    return np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)


def mean_iou(cm):
    return float(np.nanmean(intersection_over_union(cm)))


def frequency_weighted_iou(cm):
    cm = cm.astype(np.float64)
    freq = cm.sum(axis=1) / cm.sum() if cm.sum() > 0 else np.zeros(cm.shape[0])
    iou = intersection_over_union(cm)
    valid = ~np.isnan(iou)
    return float((freq[valid] * iou[valid]).sum())
```

## 5. Dice 系数
### 公式
$$
Dice_c = \frac{2TP_c}{2TP_c + FP_c + FN_c}
$$

### 物理含义
Dice 衡量预测区域和真实区域的重合程度，医学图像分割中非常常用。

### 高低代表什么
- 越高越好，最大值为 1。
- Dice 对小目标区域较敏感，适合病灶、器官等区域分割。

### 伪代码
```text
for each class:
    tp = cm[class, class]
    fp = column_sum[class] - tp
    fn = row_sum[class] - tp
    dice = 2 * tp / (2 * tp + fp + fn)
```

### Python 实现
```python
import numpy as np


def dice_score(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)


def mean_dice(cm):
    return float(np.nanmean(dice_score(cm)))
```

## 6. 使用建议
- 通用语义分割优先报告 mIoU。
- 类别极不均衡时，同时报告每类 IoU、Mean Accuracy 和 Dice。
- 自动驾驶场景建议按类别组分析，如 road、person、vehicle、traffic sign。

