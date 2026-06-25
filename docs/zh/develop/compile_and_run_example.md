# 编译与运行样例

## 前提说明

- 请确保基础环境已搭建完成，包括 NPU 驱动与固件、CANN 软件包（toolkit + ops）等。环境搭建请参考[环境部署](../install/quick_install.md)。
- ops-blas 库需根据实际硬件 SOC 编译，编译时通过 `--soc` 参数指定目标芯片（如 `bash build.sh --soc=ascend950`）。若编译的 SOC 与运行设备不匹配，kernel 执行时会报错（如内核层 `EZ9999`、ACL 运行时层 `507035`）。

## 步骤一：编译并安装 ops-blas

1. 下载源码。

    ```bash
    git clone https://gitcode.com/cann/ops-blas.git
    cd ops-blas
    ```

2. 安装基础依赖。

    ```bash
    bash install_deps.sh
    pip3 install -r requirements.txt
    ```

3. 配置 CANN 环境变量。

    ```bash
    # 默认路径安装，以 root 用户为例
    source /usr/local/Ascend/cann/set_env.sh
    # 若为指定路径安装，请替换为：
    # source ${install_path}/cann/set_env.sh
    ```

4. 编译生成 run 安装包。

    以 Ascend 950 为例，在 ops-blas 源码根目录下执行：

    ```bash
    bash build.sh --pkg --soc=ascend950
    ```

    编译成功后，安装包生成在 `build_out/` 目录下，文件名类似 `cann-950-ops-blas_<version>_linux-<arch>.run`。

    > **说明**：`--soc` 参数必须与运行设备的芯片型号一致，否则运行时会报错。支持的 SOC 值：`ascend910b`、`ascend910_93`、`ascend950`。

5. 安装 run 包。

    ```bash
    # 赋予可执行权限
    chmod +x build_out/cann-950-ops-blas_*.run
    # 安装（root 用户默认安装到 /usr/local/Ascend）
    ./build_out/cann-950-ops-blas_*.run --install --quiet
    ```

    安装后，ops-blas 的头文件和库文件会安装到 CANN 安装目录下：

    ```
    ${install_path}/cann/
    ├── include/
    │   ├── cann_ops_blas.h
    │   ├── cann_ops_blasLt.h
    │   └── cann_ops_blas_common.h
    └── lib64/
        ├── libops_blas.so
        └── libops_blasLt.so
    ```

## 步骤二：准备示例代码

本章以开发和运行环境合设场景为例，即代码开发和代码运行在同一台机器上。这里以 **Sscal 算子**为例，调用其他算子请根据实际情况自行修改 API 调用脚本（\*.cpp）和编译脚本（CMakeLists.txt）。

1. 新建项目目录。

    ```bash
    mkdir -p ops-blas-example
    cd ops-blas-example
    ```

2. 在项目目录下创建代码文件 `test_sscal.cpp`。

    ```bash
    vi test_sscal.cpp
    ```

    将以下代码粘贴到文件中并保存：

    ```cpp
    #include <iostream>
    #include <vector>
    #include <cmath>

    #include "acl/acl.h"
    #include "cann_ops_blas.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define CHECK_FREE_RET(cond, return_expr) \
        do {                                  \
            if (!(cond)) {                    \
                Finalize(deviceId, stream);   \
                return_expr;                  \
            }                                 \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    int Init(int32_t deviceId, aclrtStream *stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    void Finalize(int32_t deviceId, aclrtStream stream)
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }

    int aclblasSscalTest(int32_t deviceId, aclrtStream &stream)
    {
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 1. 创建 ops-blas 句柄
        aclblasHandle_t handle = nullptr;
        auto blasRet = aclblasCreate(&handle);
        CHECK_FREE_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
                       LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); return blasRet);
        blasRet = aclblasSetStream(handle, stream);
        CHECK_FREE_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
                       LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet); return blasRet);

        // 2. 准备 Host 数据
        int n = 5;
        int incx = 1;
        float alpha = 2.0f;
        std::vector<float> xHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        size_t xBytes = n * sizeof(float);

        // 3. 申请 Device 内存并拷贝数据
        void *xDeviceAddr = nullptr;
        auto aclRet = aclrtMalloc(&xDeviceAddr, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(aclRet == ACL_SUCCESS,
                       LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
        aclRet = aclrtMemcpy(xDeviceAddr, xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_FREE_RET(aclRet == ACL_SUCCESS,
                       LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

        // 4. 调用 aclblasSscal（alpha 为 Host 指针）
        blasRet = aclblasSscal(handle, n, &alpha, static_cast<float*>(xDeviceAddr), incx);
        CHECK_FREE_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
                       LOG_PRINT("aclblasSscal failed. ERROR: %d\n", blasRet); return blasRet);

        // 5. 同步等待任务执行结束
        aclRet = aclrtSynchronizeStream(stream);
        CHECK_FREE_RET(aclRet == ACL_SUCCESS,
                       LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

        // 6. 将结果从 Device 拷贝回 Host 并打印
        std::vector<float> resultData(n, 0);
        aclRet = aclrtMemcpy(resultData.data(), xBytes, xDeviceAddr, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_FREE_RET(aclRet == ACL_SUCCESS,
                       LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet); return aclRet);
        for (int i = 0; i < n; i++) {
            LOG_PRINT("result[%d] is: %f\n", i, resultData[i]);
        }

        // 7. 释放资源
        aclrtFree(xDeviceAddr);
        aclblasDestroy(handle);

        return ACL_SUCCESS;
    }

    int main()
    {
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = aclblasSscalTest(deviceId, stream);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSscalTest failed. ERROR: %d\n", ret); return ret);

        Finalize(deviceId, stream);
        return 0;
    }
    ```

3. 在项目目录下创建编译脚本 `CMakeLists.txt`。

    ```bash
    vi CMakeLists.txt
    ```

    将以下内容粘贴到文件中并保存（无需修改任何内容，路径通过环境变量自动获取）：

    ```cmake
    cmake_minimum_required(VERSION 3.14)

    project(ACLBLAS_EXAMPLE)

    add_compile_options(-std=c++11)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    add_executable(opapi_test
                   test_sscal.cpp)

    # CANN toolkit 路径，优先读取环境变量，若未设置则使用默认路径
    if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()

    # ops-blas 路径，若与 CANN 安装路径不同请通过环境变量 OPS_BLAS_PATH 指定
    if(NOT "$ENV{OPS_BLAS_PATH}" STREQUAL "")
        set(OPS_BLAS_PATH $ENV{OPS_BLAS_PATH})
    else()
        set(OPS_BLAS_PATH "${ASCEND_PATH}")
    endif()

    include_directories(
        ${ASCEND_PATH}/include
        ${ASCEND_PATH}/include/aclnn
        ${OPS_BLAS_PATH}/include
    )

    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${OPS_BLAS_PATH}/lib64/libops_blas.so)

    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

4. 配置路径环境变量。

    CMakeLists.txt 通过环境变量自动获取路径，无需手动编辑。请在当前终端执行以下命令设置环境变量：

    ```bash
    # 配置 CANN 环境变量（source set_env.sh 后 ASCEND_HOME_PATH 会自动设置）
    source /usr/local/Ascend/cann/set_env.sh
    # 若为指定路径安装，请替换为：
    # source ${install_path}/cann/set_env.sh

    # 若 ops-blas 安装路径与 CANN 不同，需额外指定（大多数情况无需设置）
    # export OPS_BLAS_PATH=/your/ops-blas/install/path
    ```

    > **说明**：
    >
    > - `source set_env.sh` 会自动设置 `ASCEND_HOME_PATH` 环境变量，CMakeLists.txt 通过该变量定位 CANN 头文件和库文件。
    > - 若 ops-blas 的 run 包已安装到 CANN 目录下（步骤一第 5 步），则 `OPS_BLAS_PATH` 默认与 `ASCEND_HOME_PATH` 相同，无需额外设置。
    > - 如需使用 BLASLt 接口（如 `aclblasLtMatmul`），还需在 CMakeLists.txt 中额外链接 `libops_blasLt.so`。

## 步骤三：编译与运行

以下步骤均在步骤二创建的项目目录下执行。

1. 新建 build 目录并执行编译。

    ```bash
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    ```

    编译成功后，会在 `build/bin/` 目录下生成可执行文件 `opapi_test`。

2. 运行。

    ```bash
    cd bin
    ./opapi_test
    ```

    以 Sscal 算子（n=5, alpha=2.0, x=[1,2,3,4,5]）为例，预期输出如下：

    ```bash
    result[0] is: 2.000000
    result[1] is: 4.000000
    result[2] is: 6.000000
    result[3] is: 8.000000
    result[4] is: 10.000000
    ```

3. 若执行结果报错，未出现预期结果，可以使用 `aclGetRecentErrMsg` 接口获取报错具体信息。示例如下：

    ```cpp
    ret = aclblasSscal(handle, n, &alpha, x, incx);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        printf("aclblasSscal failed. ERROR: %d\n[ERROR msg]%s\n", ret, aclGetRecentErrMsg());
        return ret;
    }
    ```

## 常见问题

- 若运行时出现内核层 `EZ9999` 或 ACL 运行时层 `507035` 错误，通常是 ops-blas 编译时的 `--soc` 参数与运行设备不匹配，请确认编译目标 SOC 与硬件一致后重新编译 ops-blas。
- 若链接时报 `undefined reference to aclblasXxx`，请确认链接的是正确编译产物 `libops_blas.so`（`--ops` 参数仅控制测试程序的编译，不影响库中包含的算子，正常编译的 `libops_blas.so` 包含全部算子符号）。可通过 `nm -D libops_blas.so | grep aclblasXxx` 确认目标符号是否存在。若符号缺失，请检查是否误链接了旧版本或不完整的库文件。
