# aclblasStpsv

## 接口

```c
aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const float *AP,
    float *x,
    int incx);
```

## 功能

求解三角线性系统：

```
op(A) * x = b
```

其中 A 为 n x n 三角矩阵（上三角或下三角），以 packed 格式存储，b 为右端向量（输入时通过 x 传入），x 为解向量（输出时原地覆盖 b）。op(A) 可为 A、A^T 或 A^H（实数 FP32 场景下 A^T 与 A^H 等价）。

### Packed 存储格式

三角矩阵 A 以列主序 packed 格式存储，共 `n*(n+1)/2` 个元素：

- **上三角**：`AP[i + j*(j+1)/2]` 存储 `A[i][j]`（`0 <= i <= j < n`）
- **下三角**：`AP[i + (2*n-j-1)*j/2]` 存储 `A[i][j]`（`0 <= j <= i < n`）

### 参数说明

| 参数 | 方向 | 位置 | 说明 |
|------|------|------|------|
| handle | in | Host | aclblas 库句柄，内部携带 stream |
| uplo | in | Host | `ACLBLAS_UPPER(121)` — A 为上三角矩阵；`ACLBLAS_LOWER(122)` — A 为下三角矩阵 |
| trans | in | Host | `ACLBLAS_OP_N(111)` — op(A) = A；`ACLBLAS_OP_T(112)` — op(A) = A^T；`ACLBLAS_OP_C(113)` — op(A) = A^H（FP32 下与 T 等价） |
| diag | in | Host | `ACLBLAS_NON_UNIT(131)` — 对角元从 AP 读取；`ACLBLAS_UNIT(132)` — 对角元固定为 1 |
| n | in | Host | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功 |
| AP | in | Device | packed 三角矩阵指针，共 `n*(n+1)/2` 个元素 |
| x | in/out | Device | 输入时存储右端向量 b，输出时原地覆盖为解向量 x |
| incx | in | Host | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储 |

**注意**：AP、x 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

**异步执行**：Host 侧不执行流同步，kernel 以异步方式提交到 stream。调用者需在 kernel 执行后自行调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeDevice` 进行同步，然后再通过 `aclrtMemcpy` 拷贝结果回 host。

### 参数约束

| 条件 | 返回值 |
|------|--------|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `uplo` 无效 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `trans` 无效 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `diag` 无效 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `AP == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

### 求解方向

| uplo | trans | 方向 | 说明 |
|------|-------|------|------|
| LOWER | N | 前向（Forward） | 逐行从上到下求解 |
| UPPER | T/C | 前向（Forward） | 逐行从上到下求解 |
| UPPER | N | 后向（Backward） | 逐行从下到上求解 |
| LOWER | T/C | 后向（Backward） | 逐行从下到上求解 |

### 执行路径

| 路径 | 条件 | 策略 |
|------|------|------|
| 标量路径 | n < 128 | 单核 AI Core，TPipe/TQue 三阶流水线逐行求解 |
| SIMT 路径 | n >= 128 | 多线程并行化每行内积计算，树形归约 |

### TilingData 下发

TilingData 通过值传递（pass-by-value）方式下发至 kernel，无需设备侧内存分配和 H2D 拷贝。AP 和 x 的 device 指针地址嵌入 TilingData 结构中一并传递。

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 |
| 目标芯片 | Ascend950PR |
| 目标架构 | arch35 (DAV_3510) |

## 目录结构

```
├── tpsv
│   ├── README.md
│   └── arch35/
│       ├── stpsv_host.cpp
│       ├── stpsv_kernel.cpp          // 标量路径 kernel (n < 128)
│       ├── stpsv_kernel_simt.cpp     // SIMT 路径 kernel (n >= 128)
│       ├── stpsv_kernel_utils.h      // packed 索引工具函数
│       └── stpsv_tiling_data.h
```

测试代码位于 `test/tpsv/stpsv/`：

```
test/tpsv/stpsv/
├── CMakeLists.txt
├── stpsv_param.h
├── stpsv_golden.h              // CPU golden（基于 CBLAS cblas_stpsv）
└── arch35/
    ├── stpsv_test.cpp
    ├── stpsv_test.csv
    └── stpsv_npu_wrapper.h
```

## 编译

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=tpsv --soc=ascend950

# 编译并运行测试
bash build.sh --ops=tpsv --soc=ascend950 --run
```
