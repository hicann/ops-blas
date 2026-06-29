# rotg算子

## 算子概述

rotg 算子实现了 Givens 旋转参数的构造，根据输入的两个标量 a、b 计算平面旋转矩阵的余弦 c 和正弦 s，使得旋转后 2×1 向量 (a, b)ᵀ 的第二个元素置零。输出时标量 a 被覆写为旋转后的范数 r，b 被覆写为恢复参数 z，常用于 QR 分解、最小二乘求解等数值算法中，其输出 c、s 可直接作为下游 rot 算子的输入。

数学表达式：

```
| c   s |   | a |   | r |
| -s  c | * | b | = | 0 |
```

其中 r = ±√(a² + b²)，算法采用 scale = |a| + |b| 进行缩放以避免溢出。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSrotg | 单精度构造 Givens 旋转参数 |

## 目录结构介绍

```
blas/rotg/
├── README.md                       // 说明文档
├── arch22/
│   ├── srotg_host.cpp              // Host 侧入口：参数校验 + Host/Device 双路径判别 + CPU 计算
│   ├── srotg_kernel.cpp            // Device kernel：标量 Givens 旋转参数生成
│   └── srotg_tiling_data.h         // Tiling 数据结构
└── arch35/
    ├── srotg_host.cpp              // Host 侧入口：参数校验 + Host/Device 双路径判别 + CPU 计算
    ├── srotg_kernel.cpp            // Device SIMT kernel：标量 Givens 旋转参数生成
    └── srotg_kernel.h              // srotg_kernel_do 声明

test/rotg/srotg/
├── CMakeLists.txt
├── srotg_param.h                   // CSV 用例参数结构
├── srotg_golden.h                  // CPU golden（cblas_srotg）
├── arch22/
│   ├── srotg_test.cpp              // GTest：arch22 测试用例
│   ├── srotg_test.csv              // 用例数据
│   └── srotg_npu_wrapper.h         // NPU device 指针路径 wrapper
└── arch35/
    ├── srotg_test.cpp              // GTest：错误路径 + Device/Host 双路径 CSV 驱动
    ├── srotg_test.csv              // 用例数据（含 NaN/Inf/denormal）
    └── srotg_npu_wrapper.h         // NPU device 指针路径 wrapper
```

## 算子执行接口

### aclblasSrotg

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSrotg(aclblasHandle_t handle, float *a, float *b, float *c, float *s);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| a | 输入/输出 | float*（FP32） | 输入标量 a，运算后覆写为 r = ±√(a²+b²)，Device 内存 |
| b | 输入/输出 | float*（FP32） | 输入标量 b，运算后覆写为恢复参数 z，Device 内存 |
| c | 输出 | float*（FP32） | Givens 旋转矩阵余弦元素，Device 内存 |
| s | 输出 | float*（FP32） | Givens 旋转矩阵正弦元素，Device 内存 |

#### 约束说明

- a、b、c、s 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- a、b、c、s 必须全部位于 Host 侧或全部位于 Device 侧；混合 Host/Device 指针将返回 ACLBLAS_STATUS_INVALID_VALUE
- Host 路径：直接 CPU 计算，无 kernel 开销，无数据搬运
- Device 路径：启动 device kernel（1 block SIMT），计算和结果都留在 device 上
- 算子输入 shape 为标量（各 1 个元素），输出 shape 同为标量
- Host 侧不做流同步，调用方需自行管理同步

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle;
    aclblasCreate(&handle);
    aclrtStream stream;
    aclblasGetStream(handle, &stream);

    float aHost = 3.0f;
    float bHost = 4.0f;
    float *aDev = nullptr;
    float *bDev = nullptr;
    float *cDev = nullptr;
    float *sDev = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&aDev), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&bDev), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&cDev), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&sDev), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDev, sizeof(float), &aHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bDev, sizeof(float), &bHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasSrotg(handle, aDev, bDev, cDev, sDev);

    aclrtSynchronizeStream(stream);

    float cHost = 0.0f;
    float sHost = 0.0f;
    aclrtMemcpy(&aHost, sizeof(float), aDev, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&bHost, sizeof(float), bDev, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&cHost, sizeof(float), cDev, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&sHost, sizeof(float), sDev, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(aDev);
    aclrtFree(bDev);
    aclrtFree(cDev);
    aclrtFree(sDev);

    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- Srotg 编译并执行测试
  ```bash
  bash build.sh --ops=srotg --soc=ascend950 --run
  ```

  执行结果如下，说明所有测试用例通过。
  ```
  [PASS] srotg_test

  ========================================
  Test Summary:
    Passed:  1 - srotg
    Skipped: 0 -  (not supported on ascend950)
    Failed:  0 -
  ========================================
  ```
