## Syr算子实现

## 概述

本样例展示 `aclblasSsyr` 在 Ascend A5 平台上的基本使用流程。

Syr (Symmetric Rank-1 Update) 实现对称矩阵的秩-1更新操作，数学表达式为：

```
A := alpha * x * x^T + A
```

其中 A 为 n×n 对称矩阵，仅上三角或下三角区域被引用和更新。

## 支持的产品

- Atlas A5 系列产品 (Ascend950)

## 目录结构介绍

```
blas/syr/
├── README.md                       // 说明文档
├── arch22/
│   ├── syr_host.cpp                // Host 侧实现（arch22）
│   └── syr_kernel.cpp              // Kernel 侧实现（arch22）
└── arch35/
    ├── syr_host.cpp                // Host 侧实现（arch35）
    ├── syr_kernel.cpp              // Kernel 侧实现（arch35）
    └── syr_tiling_data.h           // Tiling 数据结构（arch35）
```

测试代码位于 `test/syr/ssyr/`：

```
test/syr/ssyr/
├── CMakeLists.txt                  // 编译工程文件
├── ssyr_param.h                    // 测试参数定义
├── ssyr_golden.h                   // Golden 参考实现（基于 CBLAS）
├── arch35/
│   ├── ssyr_test.cpp               // GTest 测试用例
│   ├── ssyr_test.csv               // CSV 测试数据
│   └── ssyr_npu_wrapper.h          // NPU 设备端调用封装
└── arch22/
    └── syr_test.cpp                // arch22 测试用例
```

## 算子描述

- 算子功能：
  Syr算子实现对称秩-1更新操作，将 `alpha * x * x^T` 加到对称矩阵 `A` 的指定三角区域。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Ssyr</td></tr>
  <tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">A</td><td align="center">n x n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">alpha</td><td align="center">scalar</td><td align="center">float32</td><td align="center">-</td></tr>
  <tr><td align="center">uplo</td><td align="center">scalar</td><td align="center">enum</td><td align="center">ACLBLAS_UPPER / ACLBLAS_LOWER</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">A</td><td align="center">n x n</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">syr_kernel</td></tr>
  </table>

- 调用实现：
  本样例为 Host API 调用示例，使用 `aclblasSsyr` 接口完成算子配置与执行。

- 接口定义：

```cpp
aclblasStatus_t aclblasSsyr(aclblasHandle_t handle,
                             aclblasFillMode_t uplo,
                             const int n,
                             const float *alpha,
                             const float *x, const int incx,
                             float *A, const int lda);
```

- 参数说明：

| 参数 | 说明 |
|------|------|
| handle | aclBLAS 库句柄 |
| uplo | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域 |
| n | 矩阵阶数，n >= 0 |
| alpha | 标量乘数指针 |
| x | 输入向量指针，长度至少 `1 + (n-1) * abs(incx)` |
| incx | x 的元素间步长，incx != 0 |
| A | 对称矩阵指针，维度 (lda, n) |
| lda | A 的主维度，lda >= max(1, n) |

- 返回码说明：

| 返回码 | 说明 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 执行成功 |
| ACLBLAS_STATUS_INVALID_VALUE | 参数无效（n<0, incx=0, lda不足, 空指针等） |
| ACLBLAS_STATUS_EXECUTION_FAILED | 内部执行错误（如获取核数失败） |

- 精度指标：

| 数据类型 | atol | rtol |
|---------|------|------|
| float32 | 0.002 | 0.001 |

## 关键设计

- 多核并行：循环行分布 (Cyclic Row Distribution)，多核负载均衡
- UB 缓存优化：incx==1 时将 x 向量加载到 UB，减少 GM 访问
- GM Fallback：incx!=1 或 xLen 超出 UB 容量时，回退到 GM 直接访问模式
- TilingData 传值下发：Kernel 通过值传递接收 TilingData，无需设备侧内存分配
- 异步执行：Host 侧不执行流同步，由调用方自行管理同步

## 编译运行

- 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

- 样例执行

```bash
bash build.sh --ops=syr --soc=ascend950 --run
```

执行结果如下，说明精度对比成功。

```
========================================
  Total: 45  Passed: 45  Failed: 0
========================================
  RESULT: ALL TESTS PASSED
```

## 注意事项

1. 本算子仅支持 Ascend950 (A5) 平台，旧版 ssyr 算子已从 A5 排除编译
2. 接口对齐 cuBLAS cublasSsyr，alpha 为指针类型 `const float *`
3. Host 侧不做流同步，调用方需在 kernel 执行后自行调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeDevice`
