# 图像分类评价指标

图像分类评价关注模型预测类别与真实类别是否一致。常见输入包括真实标签 `y_true`、预测类别 `y_pred`，或模型输出分数/概率 `scores`。

## 1. 指标总览
| 指标 | 公式 | 含义 | 趋势 |
| --- | --- | --- | --- |
| Accuracy | $\frac{TP+TN}{N}$ | 所有样本中预测正确的比例 | 越高越好 |
| Top-k Accuracy | $\frac{1}{N}\sum_i \mathbb{1}(y_i \in TopK(s_i))$ | 真实类别是否出现在前 k 个预测中 | 越高越好 |
| Precision | $\frac{TP}{TP+FP}$ | 被预测为某类的样本中有多少是真的 | 越高越好 |
| Recall | $\frac{TP}{TP+FN}$ | 某类真实样本中有多少被找出来 | 越高越好 |
| F1-score | $\frac{2PR}{P+R}$ | Precision 与 Recall 的调和平均 | 越高越好 |
| Confusion Matrix | $C_{ij}=\#(y=i,\hat{y}=j)$ | 展示真实类别和预测类别的对应关系 | 对角线越大越好 |

## 2. Accuracy
### 公式
$$
Accuracy = \frac{\sum_{i=1}^{N}\mathbb{1}(\hat{y}_i=y_i)}{N}
$$

### 物理含义
Accuracy 表示整体分类正确率。它直观、易解释，但在类别极不均衡时可能掩盖少数类错误。

### 高低代表什么
- 越高越好，最大值为 1。
- 如果类别分布极不均衡，需要同时查看 Precision、Recall、F1 或每类准确率。

### 伪代码
```text
correct = 0
for each sample:
    if predicted_label == true_label:
        correct += 1
accuracy = correct / number_of_samples
```

### Python 实现
```python
import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))
```

## 3. Top-k Accuracy
### 公式
$$
Top\text{-}k = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}(y_i \in TopK(s_i))
$$

### 物理含义
Top-k Accuracy 判断真实类别是否出现在模型分数最高的 k 个类别中。ImageNet 常报告 Top-1 和 Top-5。

### 高低代表什么
- 越高越好。
- Top-5 明显高于 Top-1 说明模型能把正确类别排到候选集中，但第一选择仍不够稳定。

### 伪代码
```text
for each sample:
    get indices of top k scores
    check whether true label is in those indices
top_k_accuracy = number_of_hits / number_of_samples
```

### Python 实现
```python
import numpy as np


def top_k_accuracy(y_true, scores, k=5):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    top_k = np.argpartition(scores, kth=-k, axis=1)[:, -k:]
    hits = np.any(top_k == y_true[:, None], axis=1)
    return float(np.mean(hits))
```

## 4. Precision、Recall、F1-score
### 公式
对某一类别 $c$：
$$
Precision_c = \frac{TP_c}{TP_c + FP_c}
$$

$$
Recall_c = \frac{TP_c}{TP_c + FN_c}
$$

$$
F1_c = \frac{2 \cdot Precision_c \cdot Recall_c}{Precision_c + Recall_c}
$$

### 物理含义
- Precision：模型预测为某类时，有多少预测是可靠的。
- Recall：真实属于某类时，有多少被模型成功找回。
- F1：在 Precision 和 Recall 之间取平衡。

### 高低代表什么
- Precision、Recall、F1 都是越高越好。
- Precision 高、Recall 低：模型保守，漏检多。
- Precision 低、Recall 高：模型激进，误报多。

### 伪代码
```text
build confusion matrix
for each class:
    tp = confusion[class, class]
    fp = sum(confusion[:, class]) - tp
    fn = sum(confusion[class, :]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
macro_f1 = mean(f1 over classes)
```

### Python 实现
```python
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        matrix[target, pred] += 1
    return matrix


def precision_recall_f1(y_true, y_pred, num_classes=None, average="macro"):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    if average == "none":
        return precision, recall, f1
    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())
    if average == "micro":
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return float(p), float(r), float(f)
    raise ValueError("average must be 'macro', 'micro', or 'none'")
```

## 5. 使用建议
- 类别均衡时，Accuracy 可以作为主指标。
- 类别不均衡时，优先报告 macro Precision、macro Recall、macro F1。
- 多分类任务建议同时给出 confusion matrix，用于定位易混淆类别。

