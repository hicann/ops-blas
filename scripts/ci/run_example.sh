#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ops-blas 用例执行脚本，根据修改文件触发对应算子的测试用例
ROOT_PATH=$(
    cd "$(dirname $0)"/../..
    pwd
)
cd ${ROOT_PATH}
if [ $# -lt 1 ]; then
    echo "Usage: $0 <pr_filelist.txt> [soc_version]"
    exit 1
fi

pr_filelist=$1
soc_version=$2

if [ ! -f "$pr_filelist" ]; then
    echo "Error: File $pr_filelist not found"
    exit 1
fi

ops=()
run_all=false
ignore_dirs="common include utils frame"

while IFS= read -r filepath; do
    if [[ "$filepath" == *.md ]]; then
        continue
    fi
    if [[ "$filepath" == include* ]]; then
        run_all=true
        break
    fi
    if [[ !"$filepath" =~ blas* ]] && [[ !"$filepath" =~ test* ]];then
        continue
    fi
    paths=($(echo "$filepath" | sed 's/\// /g'))
    second_dir=${paths[2]}
    first_dir=${paths[1]}
    root_dir=${paths[0]}

    if [[ "$root_dir" == "test" ]] && ([[ ! -d "$root_dir/$first_dir" ]] || [[ " $ignore_dirs " =~ " $first_dir " ]]); then
        run_all=true
        break
    fi
        
    if [[ -n "$second_dir" ]] && [[ -d "./test/$first_dir/$second_dir" ]]; then
        ops+=($second_dir)
    elif [[ -n "$first_dir" ]] && [[ -d "./test/$first_dir" ]]; then
        ops+=($first_dir)
    fi
done < "$pr_filelist"

ops+=("blasLtMatmul")

declare -A _seen
_unique=()
for _op in "${ops[@]}"; do
    if [[ -z "${_seen[$_op]}" ]]; then
        _seen[$_op]=1
        _unique+=("$_op")
    fi
done
ops=("${_unique[@]}")

echo "Trigger Ops: ${ops[@]}."
echo "Need run all: ${run_all}."

if [ "$run_all" = true ]; then
    if [ -n "$soc_version" ]; then
        cmd="bash build.sh --run --soc=$soc_version"
    else
        cmd="bash build.sh --run"
    fi
else
    ops_str=""
    for op in "${ops[@]}"; do
        if [ -z "$ops_str" ]; then
            ops_str="$op"
        else
            ops_str="$ops_str,$op"
        fi
    done
    if [ -n "$soc_version" ]; then
        cmd="bash build.sh --run --ops=$ops_str --soc=$soc_version"
    else
        cmd="bash build.sh --run --ops=$ops_str"
    fi
fi

echo "Command: $cmd."
${cmd}