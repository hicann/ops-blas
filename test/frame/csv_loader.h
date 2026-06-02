/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "fill.h"
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ── csv_map: each CSV row is parsed into a string→string map ──
using csv_map = std::unordered_map<std::string, std::string>;

// ── ReadMap: safe lookup with default ──
template <typename Map>
inline typename Map::mapped_type ReadMap(
    const Map& m, const typename Map::key_type& key,
    const typename Map::mapped_type& defaultValue = typename Map::mapped_type{})
{
    auto it = m.find(key);
    return it != m.end() ? it->second : defaultValue;
}

// ── CSV line splitting ──
inline std::vector<std::string> CsvSplitLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            // trim
            size_t s = field.find_first_not_of(" \t\r\n");
            size_t e = field.find_last_not_of(" \t\r\n");
            fields.push_back((s == std::string::npos) ? "" : field.substr(s, e - s + 1));
            field.clear();
        } else {
            field += c;
        }
    }
    size_t s = field.find_first_not_of(" \t\r\n");
    size_t e = field.find_last_not_of(" \t\r\n");
    fields.push_back((s == std::string::npos) ? "" : field.substr(s, e - s + 1));
    return fields;
}

// ── GetCasesFromCsv<T>: parse CSV, construct one T(csv_map) per data row ──
template <typename T>
inline std::vector<T> GetCasesFromCsv(const std::string& csvPath)
{
    std::ifstream ifs(csvPath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csvPath);
    }

    std::string headerLine;
    if (!std::getline(ifs, headerLine)) {
        throw std::runtime_error("CSV file is empty: " + csvPath);
    }
    auto keys = CsvSplitLine(headerLine);

    std::vector<T> cases;
    std::string line;
    while (std::getline(ifs, line)) {
        // skip empty and comment lines
        auto trimmed = line;
        size_t ns = trimmed.find_first_not_of(" \t\r\n");
        if (ns == std::string::npos)
            continue;
        if (trimmed[ns] == '#')
            continue;

        auto values = CsvSplitLine(line);
        if (values.size() < keys.size())
            continue;

        csv_map row;
        for (size_t i = 0; i < keys.size(); i++) {
            if (!values[i].empty()) {
                row[keys[i]] = values[i];
            }
        }
        cases.emplace_back(row);
    }
    return cases;
}

// ── ReplaceFileExtension2Csv: derive CSV path from __FILE__ ──
inline std::string ReplaceFileExtension2Csv(const std::string& filePath)
{
    std::string p = filePath;
    size_t dot = p.rfind('.');
    if (dot != std::string::npos)
        p = p.substr(0, dot);
    return p + ".csv";
}

// ── Safe numeric parsers ──
inline int parseInt(const std::string& s, int def = 0)
{
    try {
        return std::stoi(s);
    } catch (...) {
        return def;
    }
}
inline int64_t parseInt64(const std::string& s, int64_t def = 0)
{
    try {
        return std::stoll(s);
    } catch (...) {
        return def;
    }
}
inline float parseFloat(const std::string& s, float def = 0.0f)
{
    try {
        return std::stof(s);
    } catch (...) {
        return def;
    }
}
inline double parseDouble(const std::string& s, double def = 0.0)
{
    try {
        return std::stod(s);
    } catch (...) {
        return def;
    }
}
inline uint32_t parseUint(const std::string& s, uint32_t def = 0)
{
    try {
        unsigned long val = std::stoul(s);
        if (val > UINT32_MAX)
            return def;
        return static_cast<uint32_t>(val);
    } catch (...) {
        return def;
    }
}

// aclblasStatus_t parser (must be defined before BlasTestParamBase)
inline aclblasStatus_t parseStatus(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasStatus_t> table = {
        {"ACLBLAS_STATUS_SUCCESS", ACLBLAS_STATUS_SUCCESS},
        {"SUCCESS", ACLBLAS_STATUS_SUCCESS},
        {"ACLBLAS_STATUS_NOT_INITIALIZED", ACLBLAS_STATUS_NOT_INITIALIZED},
        {"NOT_INITIALIZED", ACLBLAS_STATUS_NOT_INITIALIZED},
        {"ACLBLAS_STATUS_ALLOC_FAILED", ACLBLAS_STATUS_ALLOC_FAILED},
        {"ALLOC_FAILED", ACLBLAS_STATUS_ALLOC_FAILED},
        {"ACLBLAS_STATUS_INVALID_VALUE", ACLBLAS_STATUS_INVALID_VALUE},
        {"INVALID_VALUE", ACLBLAS_STATUS_INVALID_VALUE},
        {"ACLBLAS_STATUS_MAPPING_ERROR", ACLBLAS_STATUS_MAPPING_ERROR},
        {"MAPPING_ERROR", ACLBLAS_STATUS_MAPPING_ERROR},
        {"ACLBLAS_STATUS_EXECUTION_FAILED", ACLBLAS_STATUS_EXECUTION_FAILED},
        {"EXECUTION_FAILED", ACLBLAS_STATUS_EXECUTION_FAILED},
        {"ACLBLAS_STATUS_INTERNAL_ERROR", ACLBLAS_STATUS_INTERNAL_ERROR},
        {"INTERNAL_ERROR", ACLBLAS_STATUS_INTERNAL_ERROR},
        {"ACLBLAS_STATUS_NOT_SUPPORTED", ACLBLAS_STATUS_NOT_SUPPORTED},
        {"NOT_SUPPORTED", ACLBLAS_STATUS_NOT_SUPPORTED},
        {"ACLBLAS_STATUS_ARCH_MISMATCH", ACLBLAS_STATUS_ARCH_MISMATCH},
        {"ARCH_MISMATCH", ACLBLAS_STATUS_ARCH_MISMATCH},
        {"ACLBLAS_STATUS_HANDLE_IS_NULLPTR", ACLBLAS_STATUS_HANDLE_IS_NULLPTR},
        {"HANDLE_IS_NULLPTR", ACLBLAS_STATUS_HANDLE_IS_NULLPTR},
        {"ACLBLAS_STATUS_INVALID_ENUM", ACLBLAS_STATUS_INVALID_ENUM},
        {"INVALID_ENUM", ACLBLAS_STATUS_INVALID_ENUM},
        {"ACLBLAS_STATUS_UNKNOWN", ACLBLAS_STATUS_UNKNOWN},
        {"UNKNOWN", ACLBLAS_STATUS_UNKNOWN},
    };
    if (s.empty())
        return ACLBLAS_STATUS_SUCCESS;
    auto it = table.find(s);
    if (it != table.end())
        return it->second;
    try {
        return static_cast<aclblasStatus_t>(std::stoi(s));
    } catch (...) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
}

// ── BlasTestParamBase: common fields loaded from every CSV row ──
struct BlasTestParamBase {
    std::string caseName;
    std::string description;
    aclblasStatus_t expectResult = ACLBLAS_STATUS_SUCCESS;
    double mereThreshold = 0.0;
    double mareMultiplier = 0.0;
    uint32_t randomSeed = 0;

    BlasTestParamBase(const csv_map& m)
    {
        caseName = ReadMap(m, "case_name");
        description = ReadMap(m, "description");
        expectResult = parseStatus(ReadMap(m, "expect_result", "SUCCESS"));
        mereThreshold = parseDouble(ReadMap(m, "mere_threshold", "0"));
        mareMultiplier = parseDouble(ReadMap(m, "mare_multiplier", "0"));
        randomSeed = parseUint(ReadMap(m, "random_seed", "0"));
    }
};

// ── operator<< for GTest parameter name printing ──
inline std::ostream& operator<<(std::ostream& os, const BlasTestParamBase& p) { return os << p.caseName; }

// ── PrintCaseInfoString: GTest name generator ──
template <typename T>
inline std::string PrintCaseInfoString(const ::testing::TestParamInfo<T>& info)
{
    return info.param.caseName;
}

// ── Common BLAS enum parsers (values match cann_ops_blas_common.h) ──

// aclblasFillMode_t  (121=UPPER, 122=LOWER)
inline aclblasFillMode_t parseFillMode(const std::string& s)
{
    if (s == "ACLBLAS_UPPER" || s == "UPPER" || s == "121")
        return ACLBLAS_UPPER;
    if (s == "ACLBLAS_LOWER" || s == "LOWER" || s == "122")
        return ACLBLAS_LOWER;
    if (s == "INVALID" || s == "0xFF")
        return static_cast<aclblasFillMode_t>(0xFF);
    try {
        return static_cast<aclblasFillMode_t>(std::stoi(s));
    } catch (...) {
        return static_cast<aclblasFillMode_t>(0xFF);
    }
}

// aclblasDiagType_t  (131=NON_UNIT, 132=UNIT)
inline aclblasDiagType_t parseDiagType(const std::string& s)
{
    if (s == "ACLBLAS_NON_UNIT" || s == "NON_UNIT" || s == "131")
        return ACLBLAS_NON_UNIT;
    if (s == "ACLBLAS_UNIT" || s == "UNIT" || s == "132")
        return ACLBLAS_UNIT;
    if (s == "INVALID" || s == "0xFF")
        return static_cast<aclblasDiagType_t>(0xFF);
    try {
        return static_cast<aclblasDiagType_t>(std::stoi(s));
    } catch (...) {
        return ACLBLAS_NON_UNIT;
    }
}

// aclblasSideMode_t  (141=LEFT, 142=RIGHT)
inline aclblasSideMode_t parseSideMode(const std::string& s)
{
    if (s == "ACLBLAS_SIDE_LEFT" || s == "LEFT" || s == "141")
        return ACLBLAS_SIDE_LEFT;
    if (s == "ACLBLAS_SIDE_RIGHT" || s == "RIGHT" || s == "142")
        return ACLBLAS_SIDE_RIGHT;
    try {
        return static_cast<aclblasSideMode_t>(std::stoi(s));
    } catch (...) {
        return ACLBLAS_SIDE_LEFT;
    }
}

// aclblasOperation_t  (111=N, 112=T, 113=C)
inline aclblasOperation_t parseOpTrans(const std::string& s)
{
    if (s == "ACLBLAS_OP_N" || s == "N" || s == "111")
        return ACLBLAS_OP_N;
    if (s == "ACLBLAS_OP_T" || s == "T" || s == "112")
        return ACLBLAS_OP_T;
    if (s == "ACLBLAS_OP_C" || s == "C" || s == "113")
        return ACLBLAS_OP_C;
    if (s == "ACLBLAS_OP_H" || s == "H" || s == "113")
        return ACLBLAS_OP_C;
    if (s == "INVALID" || s == "0xFF")
        return static_cast<aclblasOperation_t>(0xFF);
    try {
        return static_cast<aclblasOperation_t>(std::stoi(s));
    } catch (...) {
        return ACLBLAS_OP_N;
    }
}

// aclblasComputeType_t  (0..10)
inline aclblasComputeType_t parseComputeType(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasComputeType_t> table = {
        {"ACLBLAS_COMPUTE_16F", ACLBLAS_COMPUTE_16F},
        {"COMPUTE_16F", ACLBLAS_COMPUTE_16F},
        {"0", ACLBLAS_COMPUTE_16F},
        {"ACLBLAS_COMPUTE_16F_PEDANTIC", ACLBLAS_COMPUTE_16F_PEDANTIC},
        {"COMPUTE_16F_PEDANTIC", ACLBLAS_COMPUTE_16F_PEDANTIC},
        {"1", ACLBLAS_COMPUTE_16F_PEDANTIC},
        {"ACLBLAS_COMPUTE_32F", ACLBLAS_COMPUTE_32F},
        {"COMPUTE_32F", ACLBLAS_COMPUTE_32F},
        {"2", ACLBLAS_COMPUTE_32F},
        {"ACLBLAS_COMPUTE_32F_PEDANTIC", ACLBLAS_COMPUTE_32F_PEDANTIC},
        {"COMPUTE_32F_PEDANTIC", ACLBLAS_COMPUTE_32F_PEDANTIC},
        {"3", ACLBLAS_COMPUTE_32F_PEDANTIC},
        {"ACLBLAS_COMPUTE_32F_FAST_16F", ACLBLAS_COMPUTE_32F_FAST_16F},
        {"COMPUTE_32F_FAST_16F", ACLBLAS_COMPUTE_32F_FAST_16F},
        {"4", ACLBLAS_COMPUTE_32F_FAST_16F},
        {"ACLBLAS_COMPUTE_32F_FAST_16BF", ACLBLAS_COMPUTE_32F_FAST_16BF},
        {"COMPUTE_32F_FAST_16BF", ACLBLAS_COMPUTE_32F_FAST_16BF},
        {"5", ACLBLAS_COMPUTE_32F_FAST_16BF},
        {"ACLBLAS_COMPUTE_32F_FAST_TF32", ACLBLAS_COMPUTE_32F_FAST_TF32},
        {"COMPUTE_32F_FAST_TF32", ACLBLAS_COMPUTE_32F_FAST_TF32},
        {"6", ACLBLAS_COMPUTE_32F_FAST_TF32},
        {"ACLBLAS_COMPUTE_64F", ACLBLAS_COMPUTE_64F},
        {"COMPUTE_64F", ACLBLAS_COMPUTE_64F},
        {"7", ACLBLAS_COMPUTE_64F},
        {"ACLBLAS_COMPUTE_64F_PEDANTIC", ACLBLAS_COMPUTE_64F_PEDANTIC},
        {"COMPUTE_64F_PEDANTIC", ACLBLAS_COMPUTE_64F_PEDANTIC},
        {"8", ACLBLAS_COMPUTE_64F_PEDANTIC},
        {"ACLBLAS_COMPUTE_32I", ACLBLAS_COMPUTE_32I},
        {"COMPUTE_32I", ACLBLAS_COMPUTE_32I},
        {"9", ACLBLAS_COMPUTE_32I},
        {"ACLBLAS_COMPUTE_32I_PEDANTIC", ACLBLAS_COMPUTE_32I_PEDANTIC},
        {"COMPUTE_32I_PEDANTIC", ACLBLAS_COMPUTE_32I_PEDANTIC},
        {"10", ACLBLAS_COMPUTE_32I_PEDANTIC},
    };
    if (s.empty())
        return ACLBLAS_COMPUTE_32F;
    auto it = table.find(s);
    if (it != table.end())
        return it->second;
    try {
        return static_cast<aclblasComputeType_t>(std::stoi(s));
    } catch (...) {
        return ACLBLAS_COMPUTE_32F;
    }
}

#endif // CSV_LOADER_H
