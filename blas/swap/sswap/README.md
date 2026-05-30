# sswap 算子实现

## 概述

BLAS sswap 算子实现，提供与 cuBLAS `cublasSswap` 相同的功能。

sswap (Single-precision SWAP) 实现了两个向量对应元素的交换操作：

```
对于 i = 0, 1, ..., n-1:
  temp = x[i * incx]
  x[i * incx] = y[i * incy]
  y[i * incy] = temp
```

swap 是 BLAS Level-1 函数，属于纯数据搬运类算子，不涉及任何数值计算。本实现通过多核并行 + GM↔UB 交叉写回完成向量交换，精度要求为位精确（Bitwise Match）。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A5 训练系列产品（ascend950）

## 目录结构介绍

```
├── sswap
│   ├── README.md
│   ├── arch22/
│   │   ├── sswap_host.cpp
│   │   └── sswap_kernel.cpp
│   └── arch35/
│       ├── sswap_host.cpp
│       ├── sswap_kernel.cpp
│       └── sswap_tiling_data.h
```

## 算子描述

对应的接口为：

```cpp
aclblasStatus_t aclblasSswap(
    aclblasHandle_t handle,
    const int64_t n,
    uint8_t* x,
    const int64_t incx,
    uint8_t* y,
    const int64_t incy);
```

| 参数 | in/out | 设备 | 类型 | 含义 |
|------|--------|------|------|------|
| handle | in | host | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream |
| n | in | host | const int64_t | 向量中参与交换的元素个数 |
| x | in/out | device | uint8_t* | 指向 float 向量的 device 指针，交换后包含原 y 的元素 |
| incx | in | host | const int64_t | 向量 x 中相邻元素之间的步长 |
| y | in/out | device | uint8_t* | 指向 float 向量的 device 指针，交换后包含原 x 的元素 |
| incy | in | host | const int64_t | 向量 y 中相邻元素之间的步长 |

**注意**：x、y 必须为 device 侧指针，由调用者在调用前通过 `aclrtMalloc` 分配并通过 `aclrtMemcpy` 拷入数据。stream 通过 `aclblasSetStream(handle, stream)` 绑定到 handle。

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (float) |
| 目标芯片 | Ascend950PR |
| 目标架构 | arch35 (DAV_3510) |
| 编程模型 | SIMD membase（TPipe/TQue 流水线） |
| 精度要求 | 位精确（Bitwise Match），MARE=0, MERE=0 |

## 参数约束

| 条件 | 返回值 | 说明 |
|------|--------|------|
| `n <= 0` | `ACLBLAS_STATUS_SUCCESS` | 直接返回成功，不执行任何操作 |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 空指针校验 |
| `x == nullptr \|\| y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 数据指针空指针校验 |
| `incx == 0 \|\| incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` | 步长不能为零 |
| `incx != 1 \|\| incy != 1` | `ACLBLAS_STATUS_INVALID_VALUE` | 首期仅支持 unit stride（连续存储） |

> **说明**：当前 arch35 实现仅支持 `incx = incy = 1`（连续存储），后续可扩展支持非连续步长。

## 编译运行

```bash
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 编译算子及测试
bash build.sh --ops=sswap --soc=ascend950

# 编译并运行测试
bash build.sh --ops=sswap --soc=ascend950 --run

# 直接运行已编译的测试
LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH ./build/test/sswap/sswap_test
```

## 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    // 1. 初始化 ACL
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);

    // 2. 创建 handle 和 stream
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    // 3. 准备数据
    const int64_t n = 1024;
    float* hostX = new float[n];
    float* hostY = new float[n];
    for (int64_t i = 0; i < n; i++) {
        hostX[i] = static_cast<float>(i);
        hostY[i] = static_cast<float>(n - i);
    }

    // 4. 分配 device 内存并拷贝数据
    uint8_t* devX = nullptr;
    uint8_t* devY = nullptr;
    aclrtMalloc((void**)&devX, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&devY, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devX, n * sizeof(float), hostX, n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devY, n * sizeof(float), hostY, n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 调用 sswap 交换两个向量
    aclblasStatus_t status = aclblasSswap(handle, n, devX, 1, devY, 1);

    // 6. 同步并拷贝结果回 host
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostX, n * sizeof(float), devX, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hostY, n * sizeof(float), devY, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 7. 验证结果：hostX 应为原 hostY 的数据，hostY 应为原 hostX 的数据
    // hostX[i] == n - i, hostY[i] == i

    // 8. 释放资源
    aclrtFree(devX);
    aclrtFree(devY);
    delete[] hostX;
    delete[] hostY;
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```

## 实现要点

### 多核切分

- 将 `n` 个元素按 `ELEMENTS_PER_BLOCK` (8) 对齐后均匀分配给所有 Vector Core，最后一个核吸收剩余元素
- 核数通过 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取，不硬编码

### UB 切分与流水线

- 采用 Single Buffer 模式（swap 无 Vector 计算阶段，Double Buffer 无额外收益）
- tileSize = 31744 个 float（248KB / 2 buffers / 4 bytes，256B 对齐）
- 使用 `DataCopyPad` 统一处理完整 tile 和尾部 tile，自动处理非对齐填充
- 同步机制：`SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` 保证 CopyIn 完成后再交叉 CopyOut

### 交叉写回

```
CopyIn:  xGM → inQueueX (UB)    CopyOut:  inQueueX (UB) → yGM  (x 数据写到 y 位置)
         yGM → inQueueY (UB)              inQueueY (UB) → xGM  (y 数据写到 x 位置)
```

## 与 arch22 实现的主要差异

| 项目 | arch22 实现 | arch35 实现 |
|------|-----------|-------------|
| TilingData | `startOffset[40]` / `calNum[40]` 数组 | `totalN` / `perCoreN` / `remainder` / `tileSize` 结构 |
| 核数获取 | 硬编码 `numBlocks = 8` | `aclrtGetDeviceInfo` 动态获取 |
| 编程风格 | 手写 `gm_to_ub_align` / `ub_to_gm_align` | 标准 AscendC `DataCopyPad` + `TPipe`/`TQue` |
| 同步机制 | 手写 `SET_FLAG` / `WAIT_FLAG` 宏 | 标准 `SetFlag<HardEvent::MTE2_MTE3>` / `WaitFlag` API |

## 性能特征

- **算子类型**：memory-bound（4 次 GM 访问：2 读 + 2 写）
- **多核并行**：利用 Ascend950PR 全部 Vector Core 并行处理
- **流水线**：MTE2（GM→UB）与 MTE3（UB→GM）通过硬件事件同步
- **负载均衡**：余数元素均匀分配给前几个核，避免长尾效应
