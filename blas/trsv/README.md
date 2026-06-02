# aclblasStrsv

## 接口

```c
aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int64_t n,
    const float *A,
    int64_t lda,
    float *x,
    int64_t incx);
```

## 功能

求解三角线性系统：

```
op(A) * x = b
```

其中 A 为 n x n 三角矩阵（上三角或下三角），b 为右端向量（输入时通过 x 传入），x 为解向量（输出时原地覆盖 b）。op(A) 可为 A、A^T 或 A^H（实数 FP32 场景下 A^T 与 A^H 等价）。

### 参数说明

| 参数 | 方向 | 位置 | 说明 |
|------|------|------|------|
| handle | in | Host | aclblas 库句柄，内部携带 stream |
| uplo | in | Host | `ACLBLAS_UPPER(121)` — A 为上三角矩阵；`ACLBLAS_LOWER(122)` — A 为下三角矩阵 |
| trans | in | Host | `ACLBLAS_OP_N(111)` — op(A) = A；`ACLBLAS_OP_T(112)` — op(A) = A^T；`ACLBLAS_OP_C(113)` — op(A) = A^H（FP32 下与 T 等价） |
| diag | in | Host | `ACLBLAS_NON_UNIT(131)` — 对角元从 A 读取；`ACLBLAS_UNIT(132)` — 对角元固定为 1 |
| n | in | Host | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功 |
| A | in | Device | n x lda 三角矩阵指针，仅相关三角部分被访问 |
| lda | in | Host | A 的 leading dimension，lda >= max(1, n) |
| x | in/out | Device | 输入时存储右端向量 b，输出时原地覆盖为解向量 x |
| incx | in | Host | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储 |

**注意**：A、x 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

### 参数约束

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `uplo` 无效 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `trans` 无效 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `diag` 无效 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `A == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

### 求解方向

| uplo | trans | 方向 | 说明 |
|------|-------|------|------|
| LOWER | N | 前向（Forward） | 逐行从上到下求解 |
| UPPER | T/C | 前向（Forward） | 逐行从上到下求解 |
| UPPER | N | 后向（Backward） | 逐行从下到上求解 |
| LOWER | T/C | 后向（Backward） | 逐行从下到上求解 |

### 算法流程

对每一行 row：
1. 读取 `x[row]` 作为初始值
2. 对 row 之前（Forward）或之后（Backward）的所有列 j，累加 `A[row][j] * x[j]` 并从初始值中减去
3. 若 diag = NON_UNIT，除以 A[row][row]

### 执行路径

| 路径 | 条件 | 策略 |
|------|------|------|
| 标量路径 | n < 129 | 单核 AI Core，S 标量逐行求解 |
| SIMT 路径 | n >= 129 | 多线程并行化每行内积计算，树形归约 |

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 |
| 目标芯片 | Ascend950PR |
| 目标架构 | arch35 (DAV_3510) |

## 目录结构

```
├── trsv
│   ├── README.md
│   └── arch35/
│       ├── strsv_host.cpp
│       ├── strsv_kernel.cpp
│       └── strsv_tiling_data.h
```

## 编译

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=trsv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=trsv --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/trsv/trsv_test
```

## 调用示例

以下示例求解下三角线性系统 A * x = b，其中 n = 4，diag = NON_UNIT：

```cpp
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ==========================================================================
// 示例: 求解 L * x = b
//
//        [ 1.0  0    0    0  ]  [ x0 ]   [ 1.0 ]
//   A =  [ 2.0  3.0  0    0  ]  [ x1 ] = [ 4.0 ]
//        [ 4.0  5.0  6.0  0  ]  [ x2 ]   [ 9.0 ]
//        [ 7.0  8.0  9.0  10.0]  [ x3 ]   [16.0 ]
//
//   真实解: x = [1.0, 0.6667, 0.2778, 0.1667]
// ==========================================================================

int main()
{
    constexpr int64_t n = 4;
    constexpr int64_t incx = 1;
    constexpr int64_t lda = 4;
    constexpr size_t aSize = n * lda * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);

    // 下三角矩阵 A (列主序, LDA=n)
    // 仅下三角部分有效
    float hA[n * lda] = {
        1.0f, 2.0f, 4.0f, 7.0f,   // 第1列
        0.0f, 3.0f, 5.0f, 8.0f,   // 第2列
        0.0f, 0.0f, 6.0f, 9.0f,   // 第3列
        0.0f, 0.0f, 0.0f, 10.0f   // 第4列
    };

    // 右端向量 b (原地被解 x 覆盖)
    float hX[n] = {1.0f, 4.0f, 9.0f, 16.0f};

    // 1. 初始化 ASCEND 运行环境
    aclInit(nullptr);

    // 2. 创建 handle 并绑定 stream
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    // 3. 分配 device 内存并拷贝数据
    float *dA = nullptr;
    float *dX = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dA), aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&dX), xSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX, xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 调用 aclblasStrsv: 下三角, 无转置, 非单位对角
    aclblasStatus_t status = aclblasStrsv(
        handle,
        ACLBLAS_LOWER,        // uplo   — A 为下三角矩阵
        ACLBLAS_OP_N,         // trans  — op(A) = A
        ACLBLAS_NON_UNIT,     // diag   — 对角元从 A 读取
        n,                    // n      — 矩阵阶数
        dA,                   // A      — 三角矩阵
        lda,                  // lda    — leading dimension
        dX,                   // x      — b (输入) / x (输出)
        incx);                // incx   — x 存储增量

    if (status != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasStrsv failed, status = " << status << std::endl;
        // 资源清理省略...
        return -1;
    }

    // 5. 同步 stream 并拷贝结果回 host
    aclrtSynchronizeStream(stream);

    std::memset(hX, 0, xSize);
    aclrtMemcpy(hX, xSize, dX, xSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 6. 打印结果
    std::cout << "解向量 x:" << std::endl;
    for (int64_t i = 0; i < n; ++i) {
        std::cout << "  x[" << i << "] = " << hX[i] << std::endl;
    }

    // 7. 资源释放
    aclrtFree(dA);
    aclrtFree(dX);
    aclrtDestroyStream(stream);
    aclblasDestroy(handle);

    // 8. 反初始化 ACL
    aclFinalize();

    return 0;
}

// 期望输出:
//   解向量 x:
//     x[0] = 1.0
//     x[1] = 0.666667
//     x[2] = 0.277778
//     x[3] = 0.166667
```
