# 图像增强评价指标

图像增强评价通常比较增强图像 `pred` 与参考清晰图像 `target` 的差异，也可能使用无参考质量指标。这里重点整理最常用的全参考指标。

## 1. 指标总览
| 指标 | 公式 | 含义 | 趋势 |
| --- | --- | --- | --- |
| MSE | $\frac{1}{N}\sum_i (x_i-y_i)^2$ | 像素均方误差 | 越低越好 |
| MAE | $\frac{1}{N}\sum_i \lvert x_i-y_i \rvert$ | 像素平均绝对误差 | 越低越好 |
| PSNR | $10\log_{10}\frac{MAX_I^2}{MSE}$ | 峰值信噪比，衡量保真度 | 越高越好 |
| SSIM | 亮度、对比度、结构相似性组合 | 感知结构相似度 | 越高越好 |
| LPIPS | 深度特征距离 | 感知差异 | 越低越好 |

## 2. MSE 与 MAE
### 公式
$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(x_i-y_i)^2
$$

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|x_i-y_i|
$$

### 物理含义
- MSE 对大误差更敏感，常用于 PSNR 计算。
- MAE 对异常像素更稳健，反映平均绝对偏差。

### 高低代表什么
- MSE 和 MAE 都是越低越好。
- MSE 很低通常意味着像素级保真度较好，但不一定代表视觉效果更自然。

### 伪代码
```text
diff = prediction - target
mse = mean(diff ** 2)
mae = mean(abs(diff))
```

### Python 实现
```python
import numpy as np


def mse(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))


def mae(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))
```

## 3. PSNR
### 公式
$$
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

其中 $MAX_I$ 是像素最大值，常见取值为 1 或 255。

### 物理含义
PSNR 衡量恢复图像相对参考图像的像素级保真度，是超分、去噪、去压缩等任务的常用指标。

### 高低代表什么
- 越高越好。
- PSNR 高说明像素误差小，但可能产生过平滑结果。

### 伪代码
```text
mse_value = mse(prediction, target)
if mse_value == 0:
    psnr = infinity
else:
    psnr = 10 * log10(max_value ** 2 / mse_value)
```

### Python 实现
```python
import numpy as np


def psnr(pred, target, max_value=1.0):
    mse_value = mse(pred, target)
    if mse_value == 0:
        return float("inf")
    return float(10.0 * np.log10((max_value ** 2) / mse_value))
```

## 4. SSIM
### 公式
$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

### 物理含义
SSIM 从亮度、对比度和结构三个角度衡量图像相似性，比 MSE/PSNR 更贴近人眼对结构的感知。

### 高低代表什么
- 越高越好，理论最大值为 1。
- SSIM 高说明结构保留较好，但颜色、纹理真实感仍需可视化判断。

### 伪代码
```text
compute mean of prediction and target
compute variance of prediction and target
compute covariance between prediction and target
ssim = formula using means, variances, covariance, C1, C2
```

### Python 实现
```python
import numpy as np


def ssim_global(pred, target, max_value=1.0, k1=0.01, k2=0.03):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2

    mu_x = pred.mean()
    mu_y = target.mean()
    var_x = pred.var()
    var_y = target.var()
    cov_xy = ((pred - mu_x) * (target - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator)
```

## 5. LPIPS
### 公式
$$
LPIPS(x,y)=\sum_l \frac{1}{H_lW_l}\sum_{h,w}\|w_l \odot (\phi_l(x)_{hw}-\phi_l(y)_{hw})\|_2^2
$$

### 物理含义
LPIPS 使用预训练深度网络特征距离衡量感知差异，更接近人眼对纹理和语义质量的判断。

### 高低代表什么
- 越低越好。
- LPIPS 低说明感知特征更接近，但它依赖预训练网络和实现版本。

### 伪代码
```text
extract deep features for prediction and target
normalize features at each layer
compute weighted feature distance
average over spatial dimensions and layers
```

### Python 实现
```python
def lpips_with_package(pred_tensor, target_tensor, net="alex"):
    """
    Requires: pip install lpips torch
    pred_tensor and target_tensor should be torch tensors in [-1, 1], shape [N, 3, H, W].
    """
    import lpips

    loss_fn = lpips.LPIPS(net=net)
    distance = loss_fn(pred_tensor, target_tensor)
    return float(distance.mean().item())
```

## 6. 使用建议
- 超分、去噪等保真任务常报告 PSNR 和 SSIM。
- 追求视觉真实感时建议补充 LPIPS 和主观可视化。
- 对 RGB 图像计算 PSNR/SSIM 时，需要明确是否只在 Y 通道计算，以及是否裁剪边界像素。

