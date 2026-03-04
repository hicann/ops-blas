#ifndef OPS_BLAS_TEST_COMMON_UTIL_H
#define OPS_BLAS_TEST_COMMON_UTIL_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "acl/acl.h"

template <typename T>
void FillRandomData(std::vector<T>& data, T min, T max)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<T> dist(min, max);
    for (auto& elem : data) elem = dist(gen);
  } else if constexpr (std::is_floating_point<T>::value) {
    std::uniform_real_distribution<T> dist(min, max);
    for (auto& elem : data) elem = dist(gen);
  }
}

template <typename T>
void ComputeGolden(int m, int k, int n, std::vector<T>& hostInput, std::vector<T>& hostWeight,
                   std::vector<T>& goldenOutput)
{
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t col = 0; col < n; ++col) {
      size_t offsetGolden = row * n + col;
      T sum = 0;
      for (uint32_t iter = 0; iter < k; ++iter) {
        size_t offsetInput = row * k + iter;
        size_t offsetWeight = iter * n + col;
        sum += hostInput[offsetInput] * hostWeight[offsetWeight];
      }
      goldenOutput[offsetGolden] = sum;
    }
  }
}

template <typename T>
std::vector<uint64_t> Compare(std::vector<T>& hostOutput, std::vector<T>& goldenOutput)
{
  std::vector<uint64_t> errorIndices;
  const float rtol = 1.0f / 256;
  for (uint64_t i = 0; i < hostOutput.size(); ++i) {
    T actualValue = hostOutput[i];
    T expectValue = goldenOutput[i];
    T diff = std::fabs(actualValue - expectValue);
    if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
      errorIndices.push_back(i);
    }
  }
  return errorIndices;
}

// 打印使用说明
inline void printUsage(const std::string& programName)
{
  std::cerr << "Usage: " << programName << " m k n" << std::endl;
  std::cerr << "Args: " << std::endl;
  std::cerr << "  m: row of matrix A" << std::endl;
  std::cerr << "  k: col of matrix A" << std::endl;
  std::cerr << "  n: col of matrix B" << std::endl;
  std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
}

// 解析命令行参数
inline void parseArguments(int argc, char* argv[], int& m, int& k, int& n)
{
  if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    printUsage(argv[0]);
    exit(1);
  }
  if (argc < 4) {
    throw std::invalid_argument("ERROR: Lacks Arguments");
  }
  try {
    m = std::stoi(argv[1]);
    k = std::stoi(argv[2]);
    n = std::stoi(argv[3]);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument("ERROR: m k n must be Integer");
  }

  if (m <= 0 || k <= 0 || n <= 0) {
    throw std::invalid_argument("ERROR: m k n must be positive");
  }
}

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

#define CHECK_ACLRT(func)                                                               \
  {                                                                                     \
    aclError status = (func);                                                           \
    if (status != ACL_SUCCESS) {                                                        \
      std::cerr << "ACL Runtime Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

#define CHECK_ACLBLASLT(func)                                                      \
  {                                                                                \
    aclblasStatus_t status = (func);                                               \
    if (status != ACLBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "BLASLT Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  }

#endif // OPS_BLAS_TEST_COMMON_UTIL_H
