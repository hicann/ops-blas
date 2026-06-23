# aclblasScopy

## 接口

```c
aclblasStatus_t aclblasScopy(
    aclblasHandle_t handle,
    int n,
    const float *x,
    int incx,
    float *y,
    int incy);
```

## 数学定义

scopy 实现向量拷贝操作：

```
Y_i = X_i,   for i = 0, 1, ..., N-1
```

其中 X 和 Y 为长度为 N 的 float 类型向量。

## 支持规格

| 项目 | 内容 |
|------|------|
| 数据类型 | FP32 (S) |
| 目标芯片 | Ascend950 |
| 目标架构 | arch35 (DAV_3510) |

## 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|---------|------|------|------|
| handle | Host | 输入 | aclblasHandle_t | ops-blas 上下文句柄，包含 stream 和 workspace 管理 |
| n | Host | 输入 | int | 向量长度（元素个数）。n == 0 时直接返回成功（空操作），n < 0 时返回 INVALID_VALUE |
| x | Device | 输入 | const float* | 源向量 X 的 device 侧指针，只读 |
| incx | Host | 输入 | int | X 元素的步长（以 float 元素为单位），不可为 0 |
| y | Device | 输出 | float* | 目标向量 Y 的 device 侧指针，可写 |
| incy | Host | 输入 | int | Y 元素的步长（以 float 元素为单位），不可为 0 |

## 功能特性

| 特性 | 说明 |
|------|------|
| 连续拷贝 (incx=1, incy=1) | DataCopy 整块搬运 + DataCopyPad 尾部，双缓冲 Prime-Pump-Drain |
| 混合路径 (仅一侧离散) | 连续侧 DataCopy 不限量，离散侧 DataCopyPad Compact 分批（每批 ≤4088） |
| 纯离散步长 (两侧均≠1) | 双侧 DataCopyPad Compact，受 blockCount ≤ 4095 硬件约束 |
| 负步长 | Kernel 正向读 + Gather 全局逆序重排（含多 tile/多核场景） |
| 同号负步长 | 两次负方向抵消，无需 Gather |
| 多核并行 | GetAivCoreCount 动态获取核数，均匀分配（extraBlockCores + tailElements） |
| 逆序偏移表 | 优先从 handle workspace 加载（DataCopy 一次搬运），不可用时 Kernel Init 自算 |
| 异步执行 | Host 侧 launch 后直接返回，无 sync |

## 精度标准

scopy 为纯数据搬移操作，不涉及数值计算，精度标准为**位精确（Bit-exact）**，NPU 输出与 CPU 参考输出逐位一致（MARE=0, MERE=0）。

## 错误码

| 错误码 | 触发条件 |
|--------|---------|
| ACLBLAS_STATUS_SUCCESS | 执行成功，或 n == 0 时空操作直接返回 |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr |
| ACLBLAS_STATUS_INVALID_VALUE | x 或 y 为 nullptr；n 为负数；incx 或 incy 为 0 |
| ACLBLAS_STATUS_ALLOC_FAILED | 内部内存分配失败 |
| ACLBLAS_STATUS_INTERNAL_ERROR | 内存传输或 Kernel 执行内部错误 |
| ACLBLAS_STATUS_EXECUTION_FAILED | Kernel 执行失败或 VectorCore 数为 0 |

## 编译与测试

```bash
bash build.sh --ops=scopy                    # 仅编译算子
bash build.sh --ops=scopy --soc=ascend950    # 指定 SOC
bash build.sh --ops=scopy --run              # 编译 + 运行测试
bash build.sh --ops=scopy --pkg              # 编译 + 打包
```

## 调用示例

### 基础示例：连续拷贝

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    // 1. 初始化 ACL
    aclInit(nullptr);
    aclrtSetDevice(0);

    // 2. 创建 handle
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    // 如需绑定自定义 stream，调用 aclblasSetStream(handle, stream)

    // 3. 准备数据
    const int n = 128;
    const int incx = 1;
    const int incy = 1;

    std::vector<float> xHost(n);
    std::vector<float> yHost(n, 0.0f);
    for (int i = 0; i < n; i++) {
        xHost[i] = static_cast<float>(i);
    }

    float *dX = nullptr;
    float *dY = nullptr;
    aclrtMalloc(&dX, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dY, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dX, n * sizeof(float), xHost.data(),
                n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 调用 aclblasScopy
    aclblasStatus_t ret = aclblasScopy(handle, n, dX, incx, dY, incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cerr << "aclblasScopy failed: " << ret << std::endl;
        return 1;
    }

    // 5. 同步并回拷结果
    aclrtSynchronizeDevice();
    aclrtMemcpy(yHost.data(), n * sizeof(float), dY,
                n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 6. 验证
    for (int i = 0; i < n; i++) {
        if (yHost[i] != xHost[i]) {
            std::cerr << "Mismatch at " << i << std::endl;
            return 1;
        }
    }
    std::cout << "PASS" << std::endl;

    // 7. 清理
    aclrtFree(dX);
    aclrtFree(dY);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

### 离散步长示例

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    // 离散步长：incx=2, incy=3
    const int n = 8;
    const int incx = 2;
    const int incy = 3;

    // 内存跨度 = abs(stride) * (n-1) + 1
    int xSpan = 2 * (n - 1) + 1;  // = 15
    int ySpan = 3 * (n - 1) + 1;  // = 22

    std::vector<float> xHost(xSpan, 0.0f);
    std::vector<float> yHost(ySpan, 0.0f);
    for (int i = 0; i < n; i++) {
        xHost[i * incx] = static_cast<float>(i);  // 位置 0, 2, 4, ..., 14
    }

    float *dX = nullptr;
    float *dY = nullptr;
    aclrtMalloc(&dX, xSpan * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dY, ySpan * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dX, xSpan * sizeof(float), xHost.data(),
                xSpan * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasScopy(handle, n, dX, incx, dY, incy);
    aclrtSynchronizeDevice();
    aclrtMemcpy(yHost.data(), ySpan * sizeof(float), dY,
                ySpan * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    for (int i = 0; i < n; i++) {
        if (yHost[i * incy] != static_cast<float>(i)) {
            std::cerr << "Mismatch at i=" << i << std::endl;
            return 1;
        }
    }
    std::cout << "PASS (discrete)" << std::endl;

    aclrtFree(dX);
    aclrtFree(dY);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

## 目录结构

| 路径 | 架构 | 说明 |
|------|------|------|
| arch35 | DAV_3510 | 主线实现：三路径（连续/混合/离散）+ 正负步长 + 异步 + workspace |
| arch22 | DAV_200 | Legacy：仅连续拷贝，接口 `aclblasScopy_legacy` |
