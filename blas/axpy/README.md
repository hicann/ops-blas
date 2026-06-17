# AXPY 算子族

AXPY (Alpha X Plus Y) 是 BLAS Level 1 的基础向量运算，实现 `y = alpha * x + y`。

本算子族包含：
- **saxpy**: 单精度浮点 AXPY
- **caxpy**: 复数 AXPY

---

## saxpy

### 接口签名

```cpp
aclblasStatus_t aclblasSaxpy(
    aclblasHandle_t handle,
    int n,
    const float* alpha,
    float* x,
    int incx,
    float* y,
    int incy);
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | aclblasHandle_t | ops-blas 库上下文句柄 |
| n | int | 向量元素个数 |
| alpha | const float* | 指向标量乘数的指针 |
| x | float* | 输入向量 x 的设备内存指针 |
| incx | int | 向量 x 的步长 |
| y | float* | 输入/输出向量 y 的设备内存指针 |
| incy | int | 向量 y 的步长 |

### 支持的数据类型

- FP32 (float)

### 支持的目标芯片

- Ascend950PR (arch35)

### 实现细节

**双路径设计**：
- **连续路径** (incx==1 && incy==1)：使用 SIMD + DataCopy 连续搬运 + Axpy 融合计算
- **离散路径** (incx!=1 || incy!=1)：使用 SIMT 线程级 GM 直接读写

**性能优化**：
- TQueBind 优化：y 使用 TQueBind 减少 UB 占用，tileSize 从 20,992 提升至 31,744
- Axpy 融合：使用 Axpy API 单条指令完成 alpha*x+y 计算
- 多核均衡：perCoreN 对齐 + 尾核吸收 remainder

### 测试用例

测试用例位于 `test/saxpy/arch35/saxpy_test.csv`，包含：
- L0 门槛用例（6 条）：基本功能、边界条件、参数校验
- L1 功能用例（15 条）：多场景覆盖、大 shape、特殊值

### 精度标准

- MERE < 2^-13
- MARE < 10 × 2^-13

---

## caxpy

### 接口定义

```cpp
int aclblasCaxpy(aclblasHandle handle, 
                 const std::complex<float> *x, 
                 std::complex<float> *y,
                 const std::complex<float> alpha, 
                 const int64_t n, 
                 const int64_t incx, 
                 const int64_t incy);
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| handle | aclblasHandle | ACL流句柄 |
| x | const std::complex<float>* | 输入复向量 |
| y | std::complex<float>* | 输入/输出复向量 |
| alpha | const std::complex<float> | 复数标量系数 |
| n | int64_t | 向量长度 |
| incx | int64_t | x的步长（当前未使用） |
| incy | int64_t | y的步长（当前未使用） |

### 支持的数据类型

- complex (std::complex<float>)

### 支持的目标芯片

- Ascend910B (arch22)

### 数学公式

对于每个元素 i：
```
y[i] = alpha * x[i] + y[i]
```

复数乘法展开：
```
real(y[i]) = (alpha_real * x_real - alpha_imag * x_imag) + y_real
imag(y[i]) = (alpha_real * x_imag + alpha_imag * x_real) + y_imag
```

### 性能特点

- 支持多核并行计算
- 使用原子操作实现累加
- 采用Ping-Pong机制提高数据吞吐

---

## 编译与运行

```bash
# 编译 saxpy
bash build.sh --ops=saxpy --soc=ascend950

# 编译 caxpy
bash build.sh --ops=caxpy --soc=ascend910b

# 编译并运行
bash build.sh --ops=saxpy --soc=ascend950 --run
```

## 使用示例

### saxpy
```cpp
aclblasSaxpy(handle, n, &alpha, x, incx, y, incy);
aclrtSynchronizeStream(stream);  // 等待计算完成
```

### caxpy
```cpp
#include "cann_ops_blas.h"
#include <complex>
#include <vector>

constexpr int n = 1024;
std::vector<std::complex<float>> x(n, {1.0f, 0.5f});
std::vector<std::complex<float>> y(n, {2.0f, 1.0f});
std::complex<float> alpha = {2.0f, 1.0f};

aclblasCaxpy(stream, x.data(), y.data(), alpha, n, 1, 1);
aclrtSynchronizeStream(stream);
```

## 异步执行

本函数为异步执行（Kernel 通过 handle 的 stream 提交），函数返回时 Kernel 可能仍在执行。用户需自行通过 `aclrtSynchronizeStream` 同步以确保计算完成。

## 参考文档

- [BLAS SAXPY 标准](https://netlib.org/blas/saxpy.f)
- [BLAS CAXPY 标准](https://netlib.org/blas/caxpy.f)
- [Ascend C 编程指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1Alpha001/applicationdev/ascendc/ascendc_0001.html)
