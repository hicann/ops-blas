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

BUILD_OPS=""
RUN_TEST=OFF
ENABLE_PACKAGE=FALSE
TEST_DEVICE_ID=0

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
BUILD_DIR="${BASE_PATH}/build"
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

# 分隔线和错误打印函数（参考ops-nn/build.sh格式）
dotted_line="----------------------------------------------------------------"
print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "\033[31m[ERROR] ${msg}\033[0m"
  echo $dotted_line
  echo
}

print_skip() {
  echo -e "\033[33m[SKIP] ${1}\033[0m"
}

# 支持的 SOC 版本
# 按字符串长度从长到短排序，避免前缀匹配时出错
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910_93" "ascend910b" "ascend950" "ascend310p")
SUPPORT_COMPUTE_UNIT_SHORT=($(printf '%s\n' "${SUPPORT_COMPUTE_UNIT_SHORT[@]}" | awk '{print length($0) " " $0}' | sort -rn | cut -d ' ' -f2-))

# ==========================
# 解析参数
# ==========================
for arg in "$@"; do
    case $arg in
        --ops=*)
            BUILD_OPS="${arg#*=}"
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
        --device=*)
            TEST_DEVICE_ID="${arg#*=}"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage:"
            echo "  bash build.sh                                 # 只编译库"
            echo "  bash build.sh --ops=scopy,blasLtMatmul        # 编译指定算子(多算子，逗号分隔)"
            echo "  bash build.sh --run                           # 编译并运行所有算子测试"
            echo "  bash build.sh --ops=scopy,blasLtMatmul --run  # 编译并运行多个算子"
            echo "  bash build.sh --pkg                           # 编译并打包run包"
            echo "  bash build.sh --pkg --soc=ascend950           # 打包指定SOC的run包"
            echo "  bash build.sh --pkg --soc=ascend950 --ops=scopy --run  # 编译指定算子打包并运行测试"
            echo "  bash build.sh --ops=scopy --run --device=1     # 指定测试运行设备(默认0)"
            exit 1
            ;;
    esac
done

# 默认 SOC_VERSION
if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION="ascend910b3"
fi

# SOC_VERSION -> SOC_ARCH_DIRS 映射（与 CMakeLists.txt 保持一致）
soc_lower=$(echo "${SOC_VERSION}" | tr '[:upper:]' '[:lower:]')
if [[ "${soc_lower}" == ascend910b* ]] || [[ "${soc_lower}" == ascend910_93* ]]; then
    SOC_ARCH_DIRS=("arch22")
elif [[ "${soc_lower}" == ascend950* ]]; then
    SOC_ARCH_DIRS=("arch35")
elif [[ "${soc_lower}" == ascend310p* ]]; then
    SOC_ARCH_DIRS=("arch20")
else
    SOC_ARCH_DIRS=()
fi


# 展开家族名为具体算子列表
# 例如: --ops=gbmv → --ops=sgbmv (如果 test/gbmv/sgbmv/CMakeLists.txt 存在)
# 搜索逻辑: 在家族目录下遍历各算子子目录,检查是否有目标 archXX 目录
expand_family_ops() {
  local ops_str="$1"
  local expanded=""
  IFS=',' read -ra OPS_ARRAY <<< "${ops_str}"
  for op in "${OPS_ARRAY[@]}"; do
    if [ -f "${BASE_PATH}/test/${op}/CMakeLists.txt" ]; then
      # 直接算子名 (如 scopy, sgbmv)
      if [ -z "${expanded}" ]; then
        expanded="${op}"
      else
        expanded="${expanded},${op}"
      fi
    elif [ -d "${BASE_PATH}/test/${op}" ]; then
      # 可能是家族名,遍历子目录查找具体算子
      local found_any=FALSE
      for sub_dir in "${BASE_PATH}/test/${op}"/*/; do
        [ -d "${sub_dir}" ] || continue
        [ -f "${sub_dir}/CMakeLists.txt" ] || continue
        local sub_name=$(basename "${sub_dir}")
        # 检查该算子是否有当前 SOC 对应的 archXX 目录
        local has_arch=FALSE
        for arch_dir in "${SOC_ARCH_DIRS[@]}"; do
          if [ -d "${sub_dir}/${arch_dir}" ]; then
            has_arch=TRUE
            break
          fi
        done
        if [ "${has_arch}" = "TRUE" ]; then
          if [ -z "${expanded}" ]; then
            expanded="${sub_name}"
          else
            expanded="${expanded},${sub_name}"
          fi
          found_any=TRUE
        else
          echo "[INFO] ${sub_name}: skipped (no ${SOC_ARCH_DIRS[*]} implementation)" >&2
        fi
      done
      if [ "${found_any}" = "FALSE" ]; then
        echo "[WARN] Family '${op}' has no operators with ${SOC_ARCH_DIRS[*]} implementation" >&2
      fi
    else
      # 未知算子名,保留原样让 CMake 报错
      if [ -z "${expanded}" ]; then
        expanded="${op}"
      else
        expanded="${expanded},${op}"
      fi
    fi
  done
  echo "${expanded}"
}

# 展开家族名
if [ -n "${BUILD_OPS}" ]; then
  BUILD_OPS=$(expand_family_ops "${BUILD_OPS}")
fi

# 如果 --run 单独使用（没有指定算子），自动发现所有测试目录
if [ "${RUN_TEST}" == "ON" ] && [ -z "${BUILD_OPS}" ]; then
  # 从 test 目录下查找所有包含 CMakeLists.txt 的子目录（支持家族嵌套）
  AUTO_DISCOVER_OPS=""
  for dir in ${BASE_PATH}/test/*/; do
    [ -d "${dir}" ] || continue
    dir_name=$(basename "${dir}")
    # 排除非算子目录
    if [ "${dir_name}" = "frame" ] || [ "${dir_name}" = "utils" ] || [ "${dir_name}" = "common" ]; then
      continue
    fi
    if [ -f "${dir}/CMakeLists.txt" ]; then
      # 直接算子目录 (如 test/scopy/)
      if [ -z "${AUTO_DISCOVER_OPS}" ]; then
        AUTO_DISCOVER_OPS="${dir_name}"
      else
        AUTO_DISCOVER_OPS="${AUTO_DISCOVER_OPS},${dir_name}"
      fi
    else
      # 可能是家族目录 (如 test/gbmv/sgbmv/)
      for sub_dir in "${dir}"/*/; do
        [ -d "${sub_dir}" ] || continue
        [ -f "${sub_dir}/CMakeLists.txt" ] || continue
        sub_name=$(basename "${sub_dir}")
        if [ -z "${AUTO_DISCOVER_OPS}" ]; then
          AUTO_DISCOVER_OPS="${sub_name}"
        else
          AUTO_DISCOVER_OPS="${AUTO_DISCOVER_OPS},${sub_name}"
        fi
      done
    fi
  done
  BUILD_OPS="${AUTO_DISCOVER_OPS}"
  echo "Auto-discovered tests: ${BUILD_OPS}"
fi

echo "BUILD_OPS=${BUILD_OPS}, RUN_TEST=${RUN_TEST}, ENABLE_PACKAGE=${ENABLE_PACKAGE}"

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

if [ -n "${BUILD_OPS}" ]; then
    # 将逗号分隔转换为 CMake 列表格式（分号分隔）
    TEST_NAMES_CMAKE="${BUILD_OPS//,/;}"
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBUILD_TEST=ON -DTEST_NAMES=${TEST_NAMES_CMAKE}"
fi

CMAKE_OPTIONS="${CMAKE_OPTIONS} -DTEST_DEVICE_ID=${TEST_DEVICE_ID}"

if [ "${ENABLE_PACKAGE}" == "TRUE" ]; then
    CMAKE_OPTIONS="${CMAKE_OPTIONS} -DENABLE_PACKAGE=ON"
fi

cmake -S "${BASE_PATH}" -B "${BUILD_DIR}" ${CMAKE_OPTIONS}
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
    if [ -z "${BUILD_OPS}" ]; then
        echo "Error: No tests found. Please check test directories or specify --ops=<算子名1,算子名2,...>"
        exit 1
    fi

    # 将逗号分隔的算子名转换为数组
    IFS=',' read -ra OP_ARRAY <<< "${BUILD_OPS}"

    FAILED_OPS=()
    PASSED_OPS=()
    SKIPPED_OPS=()

    # 读取 CMake 配置阶段生成的 skip 清单
    declare -A SKIP_REASON_MAP=()
    SKIPPED_FILE="${BUILD_DIR}/test/skipped_tests.list"
    if [ -f "${SKIPPED_FILE}" ]; then
        while IFS='|' read -r skip_op skip_reason || [ -n "${skip_op}" ]; do
            [ -z "${skip_op}" ] && continue
            SKIP_REASON_MAP["${skip_op}"]="${skip_reason}"
        done < "${SKIPPED_FILE}"
    fi

    for op in "${OP_ARRAY[@]}"; do
        # 解析测试二进制路径：支持直接目录 (test/scopy/) 和家族嵌套 (test/gbmv/sgbmv/)
        TEST_BIN="${BUILD_DIR}/test/${op}/${op}_test"
        if [ ! -f "${TEST_BIN}" ]; then
            # 搜索家族子目录
            for family_dir in "${BUILD_DIR}/test"/*/; do
                if [ -f "${family_dir}${op}/${op}_test" ]; then
                    TEST_BIN="${family_dir}${op}/${op}_test"
                    break
                fi
            done
        fi

        # 当前 SOC 不支持该算子时，直接标记为 skip，避免误报 fail/error
        if [ -n "${SKIP_REASON_MAP[${op}]+x}" ]; then
            echo ""
            echo "========== Skipping ${op}_test =========="
            print_skip "${op}_test: not supported on SOC '${SOC_VERSION}' (${SKIP_REASON_MAP[${op}]})"
            SKIPPED_OPS+=("${op}")
            continue
        fi

        echo ""
        echo "========== Running ${op}_test =========="
        if [ ! -f "${TEST_BIN}" ]; then
            print_error "Test binary not found: ${TEST_BIN} (build may have failed)"
            FAILED_OPS+=("${op}")
            continue
        fi

        TEST_CFG_DIR="$(dirname "${TEST_BIN}")"

        # 临时禁用 errexit 以捕获测试退出码
        set +e
        "${TEST_BIN}" "${TEST_CFG_DIR}"
        exit_code=$?
        set -e
        if [ $exit_code -eq 0 ]; then
            echo "[PASS] ${op}_test"
            PASSED_OPS+=("${op}")
        else
            echo "[FAIL] ${op}_test (exit code: $exit_code)"
            FAILED_OPS+=("${op}")
        fi
    done

    # 汇总测试结果
    echo ""
    echo "========================================"
    echo "Test Summary:"
    echo "  Passed:  ${#PASSED_OPS[@]} - ${PASSED_OPS[*]}"
    echo "  Skipped: ${#SKIPPED_OPS[@]} - ${SKIPPED_OPS[*]} (not supported on ${SOC_VERSION})"
    echo "  Failed:  ${#FAILED_OPS[@]} - ${FAILED_OPS[*]}"
    echo "========================================"

    if [ ${#FAILED_OPS[@]} -gt 0 ]; then
        exit 1
    fi
fi
