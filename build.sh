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

# 支持的 SOC 版本
# 按字符串长度从长到短排序，避免前缀匹配时出错
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910_93" "ascend910b" "ascend950" "ascend310p")
SUPPORT_COMPUTE_UNIT_SHORT=($(printf '%s\n' "${SUPPORT_COMPUTE_UNIT_SHORT[@]}" | awk '{print length($0) " " $0}' | sort -rn | cut -d ' ' -f2-))

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
            echo "  bash build.sh --pkg --soc=ascend950 # 打包指定SOC的run包"
            exit 1
            ;;
    esac
done

echo "BUILD_OP=${BUILD_OP}, RUN_TEST=${RUN_TEST}, ENABLE_PACKAGE=${ENABLE_PACKAGE}"

# ==========================
# 环境检查（Ascend/CANN）
# ==========================
ASCEND_HOME="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME}}"
if [ -z "${ASCEND_HOME}" ]; then
    echo "Error: Ascend/CANN environment is not configured."
    echo "Please source the Ascend environment script first, e.g.:"
    echo "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    echo "  or source \${HOME}/Ascend/ascend-toolkit/latest/set_env.sh"
    exit 1
fi
# 确保后续使用的 ASCEND_HOME_PATH 有值
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_HOME}}"

# ==========================
# 构建
# ==========================
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

# 默认 SOC_VERSION，ASCEND_CANN_PACKAGE_PATH 使用环境变量
if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION="ascend910b3"
fi

# 校验 SOC 是否在支持列表中（前缀匹配）
soc_lower=$(echo "${SOC_VERSION}" | tr '[:upper:]' '[:lower:]')
matched=""
for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
    if [[ "${soc_lower}" == "${support_unit}"* ]]; then
        matched="${support_unit}"
        break
    fi
done
if [ -z "${matched}" ]; then
    echo "Error: The soc [${SOC_VERSION}] is not supported."
    echo "Supported SOC: ${SUPPORT_COMPUTE_UNIT_SHORT[*]}"
    exit 1
fi

CMAKE_OPTIONS="-DSOC_VERSION=${SOC_VERSION} -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME}"

# 帮助 find_package(ASC) 定位 ASCConfig.cmake（CMake 查找 ASC/ASCConfig.cmake 或 ASC/asc-config.cmake）
if [ -z "${ASC_DIR}" ]; then
    for p in "${ASCEND_HOME}/lib64/cmake" "${ASCEND_HOME}/compiler/latest/lib64/cmake" \
             "${ASCEND_HOME}/ascendc/lib64/cmake"; do
        if [ -f "${p}/ASC/ASCConfig.cmake" ] || [ -f "${p}/ASC/asc-config.cmake" ]; then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_PREFIX_PATH=${p}"
            break
        fi
    done
fi
[ -n "${ASC_DIR}" ] && CMAKE_OPTIONS="${CMAKE_OPTIONS} -DASC_DIR=${ASC_DIR}"

if [ -n "${BUILD_OP}" ]; then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBUILD_TEST=ON -DTEST_NAME=${BUILD_OP}"
fi

if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DENABLE_PACKAGE=ON"
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
