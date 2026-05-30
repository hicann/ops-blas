# caxpy 算子

## 功能描述

复向量缩放加法：`y = alpha * x + y`

其中：
- `x`：输入复向量
- `y`：输入/输出复向量
- `alpha`：复数标量系数

## 接口定义

```cpp
int aclblasCaxpy(aclblasHandle handle, 
                 const std::complex<float> *x, 
                 std::complex<float> *y,
                 const std::complex<float> alpha, 
                 const int64_t n, 
                 const int64_t incx, 
                 const int64_t incy);
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | aclblasHandle | ACL流句柄 |
| x | const std::complex<float>* | 输入复向量 |
| y | std::complex<float>* | 输入/输出复向量 |
| alpha | const std::complex<float> | 复数标量系数 |
| n | int64_t | 向量长度 |
| incx | int64_t | x的步长（当前未使用） |
| incy | int64_t | y的步长（当前未使用） |

## 数学公式

对于每个元素 i：
```
y[i] = alpha * x[i] + y[i]
```

复数乘法展开：
```
real(y[i]) = (alpha_real * x_real - alpha_imag * x_imag) + y_real
imag(y[i]) = (alpha_real * x_imag + alpha_imag * x_real) + y_imag
```

## 使用示例

```cpp
#include "cann_ops_blas.h"
#include <complex>
#include <vector>

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    constexpr int n = 1024;
    std::vector<std::complex<float>> x(n, {1.0f, 0.5f});
    std::vector<std::complex<float>> y(n, {2.0f, 1.0f});
    std::complex<float> alpha = {2.0f, 1.0f};
    
    aclblasCaxpy(stream, x.data(), y.data(), alpha, n, 1, 1);
    aclrtSynchronizeStream(stream);
    
    // 结果: y[i] = (3.5, 3.0)
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

## 性能特点

- 支持多核并行计算
- 使用原子操作实现累加
- 采用Ping-Pong机制提高数据吞吐