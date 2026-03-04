#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

BUILD_DIR=build
BUILD_OP=""
RUN_TEST=OFF
ENABLE_PACKAGE=FALSE

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
ARCH_INFO=$(uname -m)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export EXTERNAL_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/external"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_OPP_PATH="${ASCEND_OPP_PATH}/lib64"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
CANN_3RD_LIB_PATH="${BUILD_PATH}/third_party"

# ==========================
# 解析参数
# ==========================
for arg in "$@"; do
    case $arg in
        --op=*)
            BUILD_OP="${arg#*=}"
            ;;
        --run)
            RUN_TEST=ON
            ;;
        --soc=*)
            SOC_VERSION="${arg#*=}"
            ;;
        --pkg)
            ENABLE_PACKAGE=TRUE
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage:"
            echo "  bash build.sh                      # 只编译库"
            echo "  bash build.sh --op=scopy           # 编译指定算子"
            echo "  bash build.sh --op=scopy --run     # 编译并运行算子"
            echo "  bash build.sh --pkg                # 编译并打包run包"
            echo "  bash build.sh --pkg --soc=ascend910b3"
            exit 1
            ;;
    esac
done

echo "BUILD_OP=${BUILD_OP}, RUN_TEST=${RUN_TEST}, ENABLE_PACKAGE=${ENABLE_PACKAGE}"

# ==========================
# 构建
# ==========================
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

CMAKE_OPTIONS="-DSOC_VERSION=${SOC_VERSION} -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}"

if [ -n "${BUILD_OP}" ]; then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBUILD_TEST=ON -DTEST_NAME=${BUILD_OP}"
fi

cmake -B ${BUILD_DIR} ${CMAKE_OPTIONS}
cmake --build ${BUILD_DIR} -j
if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    cmake --build ${BUILD_DIR} --target package
else
    cmake --install ${BUILD_DIR}
fi

# ==========================
# 运行算子测试
# ==========================
if [ "${RUN_TEST}" == "ON" ]; then
    if [ -z "${BUILD_OP}" ]; then
        echo "Error: --run requires --op=<算子名>"
        exit 1
    fi

    TEST_BIN="${BUILD_DIR}/test/${BUILD_OP}/${BUILD_OP}_test"
    if [ ! -f "${TEST_BIN}" ]; then
        echo "Error: test binary not found: ${TEST_BIN}"
        exit 1
    fi

    echo "Running ${BUILD_OP}_test..."
    "$TEST_BIN"
fi
