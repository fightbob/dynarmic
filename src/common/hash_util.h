/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * This software may be used and distributed according to the terms of the GNU
 * General Public License version 2 or any later version.
 */

#pragma once

#include "common/common_types.h"

namespace Dynarmic {
namespace Common {

inline u64 u64_mixer(u64 a) {
    a ^= a >> 33;
    a *= 0xff51afd7ed558ccd;
    a ^= a >> 33;
    return a;
}

struct U64Mixer {
    size_t operator()(u64 x) const {
        return static_cast<size_t>(u64_mixer(x));
    }
};

} // namespace Common
} // namespace Dynarmic
