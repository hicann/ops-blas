# aclblasScalex

## 接口

```c
aclblasStatus_t aclblasScalex(
    aclblasHandle_t   handle,
    int               n,
    const void        *alpha,
    aclDataType alphaType,
    void              *x,
    aclDataType xType,
    int               incx,
    aclDataType executionType);
```

## 功能

混合精度向量标量乘：`x[j] = alpha * x[j]`，其中 `j = (i - 1) * incx`，`i = 1, 2, ..., n`。

- `alpha` 为 FP32 标量因子（Host 或 Device 端指针）
- `x` 为输入/输出向量（Device 端指针），类型由 `xType` 指定
- 计算始终以 FP32 精度执行（`executionType`），结果写回 `x`
- 当 `n == 0` 时无实际操作，直接返回 `ACLBLAS_STATUS_SUCCESS`

## 参数

| 参数 | 类型 | 方向 | 说明 |
|------|------|------|------|
| handle | `aclblasHandle_t` | 输入 | ops-blas 库上下文句柄 |
| n | `int` | 输入 | 向量元素个数，`n >= 0` |
| alpha | `const void*` | 输入 | 标量因子指针，实际类型由 `alphaType` 指定 |
| alphaType | `aclDataType` | 输入 | alpha 数据类型，固定为 `ACL_FLOAT` |
| x | `void*` | 输入/输出 | Device 端向量指针，类型由 `xType` 指定 |
| xType | `aclDataType` | 输入 | 向量 x 的数据类型 |
| incx | `int` | 输入 | x 中相邻元素的步长，`incx != 0` |
| executionType | `aclDataType` | 输入 | 计算精度类型，固定为 `ACL_FLOAT` |

### 数据类型枚举

| 枚举值 | 值 | 对应 C++ 类型 |
|--------|-----|--------------|
| `ACL_FLOAT16` | 1 | `half` |
| `ACL_FLOAT` | 0 | `float` |
| `ACL_BF16` | 27 | `bfloat16_t` |

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | alpha=FP32, x∈{FP16, BF16, FP32}, execution=FP32 |
| 目标芯片 | Ascend950 (Ascend950DT / Ascend950PR) |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMD (incx=1) + SIMT (incx≠1) |

### 支持的 dtype 组合

| alpha 类型 | x 类型 | execution 类型 | 说明 |
|-----------|--------|---------------|------|
| `ACL_FLOAT` | `ACL_FLOAT16` | `ACL_FLOAT` | FP16 输入，FP32 精度计算 |
| `ACL_FLOAT` | `ACL_BF16` | `ACL_FLOAT` | BF16 输入，FP32 精度计算 |
| `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | 标准单精度向量缩放 |

### 精度要求

| x 数据类型 | MARE 上限 | MERE 上限 |
|-----------|----------|----------|
| FP32 | ≤ 10 × 2⁻¹³ | ≤ 2⁻¹³ |
| FP16 | ≤ 10 × 2⁻¹⁰ | ≤ 2⁻¹⁰ |
| BF16 | ≤ 10 × 2⁻⁷ | ≤ 2⁻⁷ |

### 返回值

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` | `ACLBLAS_STATUS_SUCCESS` |
| `alpha == nullptr`（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `x == nullptr`（n > 0） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `alphaType != ACL_FLOAT` | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `executionType != ACL_FLOAT` | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `xType` 非法 | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| Kernel 执行失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` |

## 实现架构

算子采用双路径实现，Host 侧根据 `incx` 值在 `scalex_kernel_do()` 中自动分发：

| 路径 | 条件 | Kernel | 说明 |
|------|------|--------|------|
| SIMD (AIV) | `incx == 1` | `scalex_aiv_kernel` | 连续向量访问，批量 DataCopy + Muls 计算 |
| SIMT (AIV) | `incx != 1` | `scalex_simt_kernel` | 跨步/非连续访问，`asc_vf_call` 启动 SIMT 线程 |

**混合精度路径（FP16/BF16）**：半精度输入先 Cast 为 FP32 → Muls 计算 → Cast 回原始类型 → 写回。FP32 路径则直接 Muls 运算，无需类型转换。

**多核切分**：
- SIMD 路径：每核处理 `perCoreN` 个元素，尾核多处理余数
- SIMT 路径：所有线程交错遍历 `totalN` 个元素，线程 `i` 从 `i` 号元素开始、步长为 `blockDim × numBlocks`；`incx < 0` 时通过 `isReverse` 标志反转索引映射 (`idx = (totalN - 1 - i) × |incx|`)

## 编译

```bash
bash build.sh --ops=scalex --soc=ascend950
```

## 测试

```bash
bash build.sh --ops=scalex --run --soc=ascend950
```

## 调用示例

```cpp
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

void scalex_example()
{
    int n = 1024;
    int incx = 1;
    float alpha = 2.5f;

    // 1. 创建 handle
    aclblasHandle_t handle;
    aclblasCreate(&handle);

    // 2. 分配 Device 端 x
    size_t xSize = n * sizeof(float);
    void *d_x;
    aclrtMalloc(&d_x, xSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 3. 准备 Host 端数据并拷贝到 Device
    std::vector<float> h_x(n, 1.0f);
    aclrtMemcpy(d_x, xSize, h_x.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 分配 Device 端 alpha 并拷贝
    void *d_alpha;
    aclrtMalloc(&d_alpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_alpha, sizeof(float), &alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 调用算子
    // x ∈ FP32: x = 2.5 * x
    aclblasStatus_t status = aclblasScalex(
        handle, n,
        d_alpha, ACL_FLOAT,
        d_x, ACL_FLOAT,
        incx, ACL_FLOAT);

    // 6. 读取结果
    std::vector<float> h_out(n, 0.0f);
    aclrtMemcpy(h_out.data(), n * sizeof(float), d_x, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // h_out[i] == 2.5f

    // 7. 释放资源
    aclrtFree(d_x);
    aclrtFree(d_alpha);
    aclblasDestroy(handle);
}
```

**FP16 混合精度示例**：

```cpp
// xType = ACL_FLOAT16，executionType = ACL_FLOAT
// alpha 仍为 FP32 Device 端指针
// 计算以 FP32 精度执行，结果 Cast 回 FP16 写回 d_x
aclblasStatus_t status = aclblasScalex(
    handle, n,
    d_alpha, ACL_FLOAT,       // alpha 固定 FP32
    d_x_half, ACL_FLOAT16,      // x 为 FP16
    incx, ACL_FLOAT);         // 计算精度 FP32
```

**incx 非连续示例（SIMT 路径）**：

```cpp
int n = 512;
int incx = 3;  // 步长为 3，向量元素索引: 0, 3, 6, ...

// x 总元素数 = (n - 1) * incx + 1 = 1534
size_t totalElements = (n - 1) * incx + 1;
size_t xSize = totalElements * sizeof(float);

void *d_x;
aclrtMalloc(&d_x, xSize, ACL_MEM_MALLOC_HUGE_FIRST);

std::vector<float> h_x(totalElements, 1.0f);
aclrtMemcpy(d_x, xSize, h_x.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);

aclblasStatus_t status = aclblasScalex(
    handle, n,
    d_alpha, ACL_FLOAT,
    d_x, ACL_FLOAT,
    incx, ACL_FLOAT);
```
