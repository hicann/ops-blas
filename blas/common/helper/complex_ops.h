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

#include <math.h>
#include "cann_ops_blas_common.h"

#ifdef __cplusplus

inline aclblasComplex operator+(aclblasComplex a, aclblasComplex b) {
    return {a.real + b.real, a.imag + b.imag};
}
inline aclblasComplex operator-(aclblasComplex a, aclblasComplex b) {
    return {a.real - b.real, a.imag - b.imag};
}
inline aclblasComplex operator*(aclblasComplex a, aclblasComplex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
inline aclblasComplex operator/(aclblasComplex a, aclblasComplex b) {
    float denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0f) {
        return {0.0f, 0.0f};
    }
    return {(a.real * b.real + a.imag * b.imag) / denom,
            (a.imag * b.real - a.real * b.imag) / denom};
}
inline bool operator==(aclblasComplex a, aclblasComplex b) {
    return a.real == b.real && a.imag == b.imag;
}
inline bool operator!=(aclblasComplex a, aclblasComplex b) { return !(a == b); }
inline float aclblasAbs(aclblasComplex z) { return sqrtf(z.real * z.real + z.imag * z.imag); }

inline aclblasDoubleComplex operator+(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return {a.real + b.real, a.imag + b.imag};
}
inline aclblasDoubleComplex operator-(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return {a.real - b.real, a.imag - b.imag};
}
inline aclblasDoubleComplex operator*(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
inline aclblasDoubleComplex operator/(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    double denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0) {
        return {0.0, 0.0};
    }
    return {(a.real * b.real + a.imag * b.imag) / denom,
            (a.imag * b.real - a.real * b.imag) / denom};
}
inline bool operator==(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return a.real == b.real && a.imag == b.imag;
}
inline bool operator!=(aclblasDoubleComplex a, aclblasDoubleComplex b) { return !(a == b); }
inline double aclblasAbs(aclblasDoubleComplex z) { return sqrt(z.real * z.real + z.imag * z.imag); }

#endif
