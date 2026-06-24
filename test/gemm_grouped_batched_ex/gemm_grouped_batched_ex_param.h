/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_EX_PARAM_H
#define GEMM_GROUPED_BATCHED_EX_PARAM_H

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

inline std::vector<std::string> SplitSemicolon(const std::string& s)
{
    std::vector<std::string> parts;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ';'))
        if (!token.empty())
            parts.push_back(token);
    return parts;
}

inline std::vector<int> ParseIntArray(const std::string& s)
{
    auto parts = SplitSemicolon(s);
    std::vector<int> result;
    for (const auto& p : parts)
        result.push_back(parseInt(p));
    return result;
}

inline std::vector<float> ParseFloatArray(const std::string& s)
{
    auto parts = SplitSemicolon(s);
    std::vector<float> result;
    for (const auto& p : parts)
        result.push_back(parseFloat(p));
    return result;
}

inline std::vector<aclblasOperation_t> ParseOpArray(const std::string& s)
{
    auto parts = SplitSemicolon(s);
    std::vector<aclblasOperation_t> result;
    for (const auto& p : parts)
        result.push_back(parseOpTrans(p));
    return result;
}

template <typename T>
inline void ExpandArrayToGroupCount(std::vector<T>& values, int groupCount)
{
    if (groupCount <= 0 || values.empty())
        return;
    if (values.size() == 1 && groupCount > 1) {
        values.resize(static_cast<size_t>(groupCount), values[0]);
    }
}

inline aclDataType ParseAclDataTypeForGemm(const std::string& s)
{
    if (s == "FP16" || s == "ACL_FLOAT16")
        return ACL_FLOAT16;
    if (s == "BF16" || s == "ACL_BF16")
        return ACL_BF16;
    if (s == "FP8" || s == "FP8_E4M3FN" || s == "ACL_FLOAT8_E4M3FN")
        return ACL_FLOAT8_E4M3FN;
    if (s == "FP8_E5M2" || s == "ACL_FLOAT8_E5M2" || s == "35")
        return ACL_FLOAT8_E5M2;
    return ACL_FLOAT16;
}

inline size_t DtypeElementSize(aclDataType dt)
{
    if (dt == ACL_FLOAT16) return 2;
    if (dt == ACL_BF16) return 2;
    if (dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2) return 1;
    return 4;
}

struct GemmGroupedBatchedExParam : public BlasTestParamBase {
    std::vector<aclblasOperation_t> transaArray;
    std::vector<aclblasOperation_t> transbArray;
    std::vector<int> mArray;
    std::vector<int> nArray;
    std::vector<int> kArray;
    std::vector<float> alphaArray;
    std::vector<float> betaArray;
    std::vector<int> ldaArray;
    std::vector<int> ldbArray;
    std::vector<int> ldcArray;
    int groupCount = 1;
    std::vector<int> groupSizeArray;
    aclDataType Atype = ACL_FLOAT16;
    aclDataType Btype = ACL_FLOAT16;
    aclDataType Ctype = ACL_FLOAT16;
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
    BlasFillMode aFill = parseFill("RANDOM");
    BlasFillMode bFill = parseFill("RANDOM");
    BlasFillMode cFill = parseFill("VALUE_NORM_0");

    void ParseGroupArrays(const csv_map& values)
    {
        transaArray = ParseOpArray(ReadMap(values, "transa_array", "N"));
        transbArray = ParseOpArray(ReadMap(values, "transb_array", "N"));
        mArray = ParseIntArray(ReadMap(values, "m_array", "32"));
        nArray = ParseIntArray(ReadMap(values, "n_array", "32"));
        kArray = ParseIntArray(ReadMap(values, "k_array", "32"));
        alphaArray = ParseFloatArray(ReadMap(values, "alpha_array", "1.0"));
        betaArray = ParseFloatArray(ReadMap(values, "beta_array", "0.0"));
        groupCount = parseInt(ReadMap(values, "group_count", "1"));
        groupSizeArray = ParseIntArray(ReadMap(values, "group_size", "1"));
        ExpandArrayToGroupCount(transaArray, groupCount);
        ExpandArrayToGroupCount(transbArray, groupCount);
        ExpandArrayToGroupCount(mArray, groupCount);
        ExpandArrayToGroupCount(nArray, groupCount);
        ExpandArrayToGroupCount(kArray, groupCount);
        ExpandArrayToGroupCount(alphaArray, groupCount);
        ExpandArrayToGroupCount(betaArray, groupCount);
        ExpandArrayToGroupCount(groupSizeArray, groupCount);
    }

    void ParseTypes(const csv_map& values)
    {
        Atype = ParseAclDataTypeForGemm(ReadMap(values, "Atype", "FP16"));
        Btype = ParseAclDataTypeForGemm(ReadMap(values, "Btype", "FP16"));
        Ctype = ParseAclDataTypeForGemm(ReadMap(values, "Ctype", "FP16"));
        computeType = parseComputeType(ReadMap(values, "computeType", "COMPUTE_32F"));
    }

    void ParseLeadingDimensions(const csv_map& values)
    {
        std::vector<int> ldaDefault;
        for (size_t g = 0; g < mArray.size(); g++) {
            int minLda = (transaArray[g] == ACLBLAS_OP_N) ? std::max(1, mArray[g]) : std::max(1, kArray[g]);
            ldaDefault.push_back(minLda);
        }
        std::string ldaStr = ReadMap(values, "lda_array", "");
        if (!ldaStr.empty())
            ldaArray = ParseIntArray(ldaStr);
        else
            ldaArray = ldaDefault;
        ExpandArrayToGroupCount(ldaArray, groupCount);

        std::vector<int> ldbDefault;
        for (size_t g = 0; g < mArray.size(); g++) {
            int minLdb = (transbArray[g] == ACLBLAS_OP_N) ? std::max(1, kArray[g]) : std::max(1, nArray[g]);
            ldbDefault.push_back(minLdb);
        }
        std::string ldbStr = ReadMap(values, "ldb_array", "");
        if (!ldbStr.empty())
            ldbArray = ParseIntArray(ldbStr);
        else
            ldbArray = ldbDefault;
        ExpandArrayToGroupCount(ldbArray, groupCount);

        std::vector<int> ldcDefault;
        for (size_t g = 0; g < mArray.size(); g++) {
            ldcDefault.push_back(std::max(1, mArray[g]));
        }
        std::string ldcStr = ReadMap(values, "ldc_array", "");
        if (!ldcStr.empty())
            ldcArray = ParseIntArray(ldcStr);
        else
            ldcArray = ldcDefault;
        ExpandArrayToGroupCount(ldcArray, groupCount);
    }

    GemmGroupedBatchedExParam(const csv_map& values) : BlasTestParamBase(values)
    {
        ParseGroupArrays(values);
        ParseTypes(values);
        ParseLeadingDimensions(values);
        aFill = parseFill(ReadMap(values, "a_fill", "RANDOM"));
        bFill = parseFill(ReadMap(values, "b_fill", "RANDOM"));
        cFill = parseFill(ReadMap(values, "c_fill", "VALUE_NORM_0"));
    }

    int TotalInstanceCount() const
    {
        int total = 0;
        for (int i = 0; i < groupCount; i++)
            total += groupSizeArray[i];
        return total;
    }

    int GroupIndexOf(int instanceIdx) const
    {
        int offset = 0;
        for (int g = 0; g < groupCount; g++) {
            if (instanceIdx < offset + groupSizeArray[g])
                return g;
            offset += groupSizeArray[g];
        }
        return groupCount - 1;
    }
};

#endif // GEMM_GROUPED_BATCHED_EX_PARAM_H
