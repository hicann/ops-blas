/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.h"
#include "verify.h"

class ConfigLoader {
    static std::string readWholeFile(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        return oss.str();
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return s.substr(start, end - start + 1);
    }

    static PrecisionMode parsePrecisionMode(const std::string& modeStr) {
        if (modeStr == "ABS") return PrecisionMode::ABS;
        if (modeStr == "REL") return PrecisionMode::REL;
        if (modeStr == "COMBINED") return PrecisionMode::COMBINED;
        if (modeStr == "MERE_MARE") return PrecisionMode::MERE_MARE;
        if (modeStr == "EXACT") return PrecisionMode::EXACT;
        if (modeStr == "INTEGER") return PrecisionMode::INTEGER;
        return PrecisionMode::ABS;
    }

    static VerifyConfig buildVerifyConfig(const TestCaseConfig& tc, const OpConfig& opCfg) {
        VerifyConfig cfg;
        cfg.mode = parsePrecisionMode(tc.level == "L0" || tc.level == "L1"
            ? (tc.absTol.has_value() || tc.mereThreshold.has_value()
                ? (tc.mereThreshold.has_value() ? "MERE_MARE" : "ABS")
                : opCfg.defaultPrecisionMode)
            : opCfg.defaultPrecisionMode);

        if (tc.mereThreshold.has_value()) {
            cfg.mode = PrecisionMode::MERE_MARE;
            cfg.mereThreshold = tc.mereThreshold.value();
        }
        if (tc.mareMultiplier.has_value()) {
            cfg.mareMultiplier = tc.mareMultiplier.value();
        }
        if (tc.absTol.has_value()) {
            cfg.absTol = tc.absTol.value();
            if (cfg.mode != PrecisionMode::MERE_MARE && cfg.mode != PrecisionMode::REL && cfg.mode != PrecisionMode::COMBINED) {
                cfg.mode = PrecisionMode::ABS;
            }
        }
        if (tc.relTol.has_value()) {
            cfg.relTol = tc.relTol.value();
        }
        return cfg;
    }

    static std::vector<std::string> splitCsvLine(const std::string& line) {
        std::vector<std::string> fields;
        std::string field;
        bool inQuotes = false;
        for (size_t i = 0; i < line.size(); i++) {
            char c = line[i];
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.push_back(trim(field));
                field.clear();
            } else {
                field += c;
            }
        }
        fields.push_back(trim(field));
        return fields;
    }

    static OpConfig loadJsonOpConfig(const std::string& jsonPath) {
        std::string json = readWholeFile(jsonPath);
        OpConfig cfg;

        auto extractStr = [&](const std::string& key) {
            std::string sk = "\"" + key + "\"";
            size_t pos = json.find(sk);
            if (pos == std::string::npos) return std::string();
            size_t colon = json.find(':', pos + sk.size());
            if (colon == std::string::npos) return std::string();
            size_t q1 = json.find('"', colon + 1);
            if (q1 == std::string::npos) return std::string();
            size_t q2 = json.find('"', q1 + 1);
            if (q2 == std::string::npos) return std::string();
            return json.substr(q1 + 1, q2 - q1 - 1);
        };

        cfg.opName = extractStr("op_name");
        cfg.apiFunction = extractStr("api_function");
        cfg.inputBuffers = extractStr("input_buffers");
        cfg.outputBuffers = extractStr("output_buffers");
        cfg.resultType = extractStr("result_type");
        cfg.dataFormat = extractStr("data_format");
        cfg.goldenFunction = extractStr("golden_function");
        cfg.defaultPrecisionMode = extractStr("default_precision_mode");
        return cfg;
    }

    static std::optional<int64_t> parseOptInt(const std::string& val) {
        if (val.empty()) return std::nullopt;
        try { return static_cast<int64_t>(std::stoll(val)); }
        catch (...) { return std::nullopt; }
    }

    static std::optional<float> parseOptFloat(const std::string& val) {
        if (val.empty()) return std::nullopt;
        try {
            if (val.find("2^-") != std::string::npos || val.find("2^(-") != std::string::npos) {
                size_t expStart = val.find('-', val.find('^'));
                int exp = std::stoi(val.substr(expStart + 1));
                return static_cast<float>(std::pow(2.0, -exp));
            }
            return std::stof(val);
        }
        catch (...) { return std::nullopt; }
    }

    static std::optional<bool> parseOptBool(const std::string& val) {
        if (val.empty()) return std::nullopt;
        std::string lower = val;
        for (auto& c : lower) c = std::tolower(c);
        if (lower == "true" || lower == "1") return true;
        if (lower == "false" || lower == "0") return false;
        return std::nullopt;
    }

    static std::optional<std::string> parseOptStr(const std::string& val) {
        if (val.empty()) return std::nullopt;
        return val;
    }

    static std::vector<TestCaseConfig> loadCsvTestCases(const std::string& csvPath) {
        std::ifstream ifs(csvPath);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + csvPath);
        }

        std::string headerLine;
        std::getline(ifs, headerLine);
        auto headers = splitCsvLine(headerLine);

        std::vector<TestCaseConfig> cases;
        std::string line;
        while (std::getline(ifs, line)) {
            if (trim(line).empty() || trim(line)[0] == '#') continue;
            auto fields = splitCsvLine(line);
            if (fields.size() < headers.size()) continue;

            TestCaseConfig tc;
            for (size_t i = 0; i < headers.size(); i++) {
                const std::string& key = headers[i];
                const std::string& val = fields[i];

                if (key == "case_id") tc.caseId = val;
                else if (key == "level") { tc.level = val.empty() ? "L0" : val; }
                else if (key == "description") tc.description = val;
                else if (key == "data_type") { tc.dataType = val.empty() ? "fp32" : val; }
                else if (key == "trans") tc.trans = parseOptStr(val);
                else if (key == "m") tc.m = parseOptInt(val);
                else if (key == "n") tc.n = parseOptInt(val);
                else if (key == "k") tc.k = parseOptInt(val);
                else if (key == "kl") tc.kl = parseOptInt(val);
                else if (key == "ku") tc.ku = parseOptInt(val);
                else if (key == "lda") tc.lda = parseOptInt(val);
                else if (key == "ldb") tc.ldb = parseOptInt(val);
                else if (key == "ldc") tc.ldc = parseOptInt(val);
                else if (key == "alpha_real") tc.alphaReal = parseOptFloat(val);
                else if (key == "alpha_imag") tc.alphaImag = parseOptFloat(val);
                else if (key == "beta_real") tc.betaReal = parseOptFloat(val);
                else if (key == "beta_imag") tc.betaImag = parseOptFloat(val);
                else if (key == "incx") tc.incx = parseOptInt(val);
                else if (key == "incy") tc.incy = parseOptInt(val);
                else if (key == "uplo") tc.uplo = parseOptStr(val);
                else if (key == "side") tc.side = parseOptStr(val);
                else if (key == "diag") tc.diag = parseOptStr(val);
                else if (key == "batch_count") tc.batchCount = parseOptInt(val);
                else if (key == "total_length") tc.totalLength = parseOptInt(val);
                else if (key == "value_x") tc.valueX = parseOptFloat(val);
                else if (key == "seed") {
                    auto v = parseOptInt(val);
                    tc.seed = v.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(v.value())) : std::nullopt;
                }
                else if (key == "expect_success") {
                    auto v = parseOptBool(val);
                    tc.expectSuccess = v.has_value() ? v.value() : true;
                }
                else if (key == "abs_tol") tc.absTol = parseOptFloat(val);
                else if (key == "rel_tol") tc.relTol = parseOptFloat(val);
                else if (key == "mere_threshold") tc.mereThreshold = parseOptFloat(val);
                else if (key == "mare_multiplier") tc.mareMultiplier = parseOptFloat(val);
            }

            cases.push_back(tc);
        }
        return cases;
    }

    static std::pair<std::vector<TestCaseConfig>, OpConfig> loadLegacyJson(const std::string& jsonPath) {
        std::string json = readWholeFile(jsonPath);

        OpConfig opCfg;
        auto extractStr = [&](const std::string& key, size_t start) {
            std::string sk = "\"" + key + "\"";
            size_t pos = json.find(sk, start);
            if (pos == std::string::npos) return std::string();
            size_t colon = json.find(':', pos + sk.size());
            if (colon == std::string::npos) return std::string();
            size_t q1 = json.find('"', colon + 1);
            if (q1 == std::string::npos) return std::string();
            size_t q2 = json.find('"', q1 + 1);
            if (q2 == std::string::npos) return std::string();
            return json.substr(q1 + 1, q2 - q1 - 1);
        };
        auto extractInt = [&](const std::string& key, size_t start) -> std::optional<int64_t> {
            std::string sk = "\"" + key + "\"";
            size_t pos = json.find(sk, start);
            if (pos == std::string::npos) return std::nullopt;
            size_t colon = json.find(':', pos + sk.size());
            if (colon == std::string::npos) return std::nullopt;
            size_t vs = colon + 1;
            while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\t' || json[vs] == '\n')) vs++;
            if (vs >= json.size() || json[vs] == '"') return std::nullopt;
            size_t ve = vs;
            while (ve < json.size() && json[ve] != ',' && json[ve] != '}' && json[ve] != '\n' && json[ve] != ' ') ve++;
            try { return static_cast<int64_t>(std::stoll(json.substr(vs, ve - vs))); }
            catch (...) { return std::nullopt; }
        };
        auto extractFloat = [&](const std::string& key, size_t start) -> std::optional<float> {
            std::string sk = "\"" + key + "\"";
            size_t pos = json.find(sk, start);
            if (pos == std::string::npos) return std::nullopt;
            size_t colon = json.find(':', pos + sk.size());
            if (colon == std::string::npos) return std::nullopt;
            size_t vs = colon + 1;
            while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\t' || json[vs] == '\n')) vs++;
            if (vs >= json.size()) return std::nullopt;
            if (json[vs] == '"') {
                size_t q2 = json.find('"', vs + 1);
                std::string sv = json.substr(vs + 1, q2 - vs - 1);
                if (sv.find("2^-") != std::string::npos || sv.find("2^(-") != std::string::npos) {
                    size_t es = sv.find('-', sv.find('^'));
                    int exp = std::stoi(sv.substr(es + 1));
                    return static_cast<float>(std::pow(2.0, -exp));
                }
                try { return std::stof(sv); } catch (...) { return std::nullopt; }
            }
            size_t ve = vs;
            while (ve < json.size() && json[ve] != ',' && json[ve] != '}' && json[ve] != '\n' && json[ve] != ' ') ve++;
            try { return std::stof(json.substr(vs, ve - vs)); } catch (...) { return std::nullopt; }
        };
        auto extractBool = [&](const std::string& key, size_t start) -> std::optional<bool> {
            std::string sk = "\"" + key + "\"";
            size_t pos = json.find(sk, start);
            if (pos == std::string::npos) return std::nullopt;
            size_t colon = json.find(':', pos + sk.size());
            if (colon == std::string::npos) return std::nullopt;
            size_t vs = colon + 1;
            while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\t')) vs++;
            if (vs >= json.size()) return std::nullopt;
            if (json.substr(vs, 4) == "true") return true;
            if (json.substr(vs, 5) == "false") return false;
            return std::nullopt;
        };

        opCfg.opName = extractStr("op_name", 0);
        opCfg.apiFunction = extractStr("api_function", 0);
        opCfg.inputBuffers = extractStr("input_buffers", 0);
        opCfg.outputBuffers = extractStr("output_buffers", 0);
        opCfg.resultType = extractStr("result_type", 0);
        opCfg.dataFormat = extractStr("data_format", 0);
        opCfg.goldenFunction = extractStr("golden_function", 0);
        opCfg.defaultPrecisionMode = extractStr("default_precision_mode", 0);

        std::string arrKey = "\"test_cases\"";
        size_t arrPos = json.find(arrKey);
        if (arrPos == std::string::npos) return {{}, opCfg};
        size_t bOpen = json.find('[', arrPos + arrKey.size());
        if (bOpen == std::string::npos) return {{}, opCfg};

        std::vector<size_t> objPositions;
        int depth = 1;
        size_t p = bOpen + 1;
        size_t objStart = std::string::npos;
        while (p < json.size() && depth > 0) {
            if (json[p] == '{') { if (depth == 1) objStart = p; depth++; }
            else if (json[p] == '}') { depth--; if (depth == 1 && objStart != std::string::npos) { objPositions.push_back(objStart); objStart = std::string::npos; } }
            else if (json[p] == '[') depth++;
            else if (json[p] == ']') depth--;
            p++;
        }

        std::vector<TestCaseConfig> cases;
        for (size_t pos : objPositions) {
            TestCaseConfig tc;
            tc.caseId = extractStr("case_id", pos);
            tc.level = extractStr("level", pos); if (tc.level.empty()) tc.level = "L0";
            tc.description = extractStr("description", pos);
            tc.dataType = extractStr("data_type", pos); if (tc.dataType.empty()) tc.dataType = "fp32";
            std::string ts = extractStr("trans", pos); tc.trans = ts.empty() ? std::nullopt : std::optional<std::string>(ts);
            tc.m = extractInt("m", pos); tc.n = extractInt("n", pos); tc.k = extractInt("k", pos);
            tc.kl = extractInt("kl", pos); tc.ku = extractInt("ku", pos);
            tc.lda = extractInt("lda", pos); tc.ldb = extractInt("ldb", pos); tc.ldc = extractInt("ldc", pos);
            tc.alphaReal = extractFloat("alpha_real", pos); tc.alphaImag = extractFloat("alpha_imag", pos);
            tc.betaReal = extractFloat("beta_real", pos); tc.betaImag = extractFloat("beta_imag", pos);
            tc.incx = extractInt("incx", pos); tc.incy = extractInt("incy", pos);
            std::string us = extractStr("uplo", pos); tc.uplo = us.empty() ? std::nullopt : std::optional<std::string>(us);
            std::string ss = extractStr("side", pos); tc.side = ss.empty() ? std::nullopt : std::optional<std::string>(ss);
            std::string ds = extractStr("diag", pos); tc.diag = ds.empty() ? std::nullopt : std::optional<std::string>(ds);
            tc.batchCount = extractInt("batch_count", pos);
            tc.totalLength = extractInt("total_length", pos);
            tc.valueX = extractFloat("value_x", pos);
            auto sv = extractInt("seed", pos); tc.seed = sv.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(sv.value())) : std::nullopt;
            auto ev = extractBool("expect_success", pos); tc.expectSuccess = ev.has_value() ? ev.value() : true;
            tc.absTol = extractFloat("abs_tol", pos); tc.relTol = extractFloat("rel_tol", pos);
            tc.mereThreshold = extractFloat("mere_threshold", pos); tc.mareMultiplier = extractFloat("mare_multiplier", pos);
            tc.verifyCfg = buildVerifyConfig(tc, opCfg);
            cases.push_back(tc);
        }
        return {cases, opCfg};
    }

public:
    static std::pair<std::vector<TestCaseConfig>, OpConfig> loadAll(
        const std::string& configDir) {

        std::string csvPath = configDir + "/" + "gbmv_testcases.csv";
        std::string jsonCfgPath = configDir + "/" + "gbmv_config.json";
        std::string legacyJsonPath = configDir + "/" + "gbmv_testcases.json";

        if (std::ifstream(csvPath).is_open()) {
            auto cases = loadCsvTestCases(csvPath);
            OpConfig opCfg;
            if (std::ifstream(jsonCfgPath).is_open()) {
                opCfg = loadJsonOpConfig(jsonCfgPath);
            }
            for (auto& tc : cases) {
                tc.verifyCfg = buildVerifyConfig(tc, opCfg);
            }
            return {cases, opCfg};
        }

        return loadLegacyJson(legacyJsonPath);
    }

    static std::pair<std::vector<TestCaseConfig>, OpConfig> loadAllForOp(
        const std::string& configDir, const std::string& opName) {

        std::string csvPath = configDir + "/" + opName + "_testcases.csv";
        std::string jsonCfgPath = configDir + "/" + opName + "_config.json";
        std::string legacyJsonPath = configDir + "/" + opName + "_testcases.json";

        if (std::ifstream(csvPath).is_open()) {
            auto cases = loadCsvTestCases(csvPath);
            OpConfig opCfg;
            if (std::ifstream(jsonCfgPath).is_open()) {
                opCfg = loadJsonOpConfig(jsonCfgPath);
            }
            for (auto& tc : cases) {
                tc.verifyCfg = buildVerifyConfig(tc, opCfg);
            }
            return {cases, opCfg};
        }

        return loadLegacyJson(legacyJsonPath);
    }
};
