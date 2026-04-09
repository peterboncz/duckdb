//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/vector_size.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"

namespace duckdb {

constexpr inline idx_t ComputePowerOfTwoBits(idx_t value) {
	return value <= 1 ? 0 : 1 + ComputePowerOfTwoBits(value >> 1);
}

//! The default standard vector size
#define DEFAULT_STANDARD_VECTOR_SIZE 2048U

//! The vector size used in the execution engine
#ifndef STANDARD_VECTOR_SIZE
#define STANDARD_VECTOR_SIZE DEFAULT_STANDARD_VECTOR_SIZE
#endif

#if (STANDARD_VECTOR_SIZE & (STANDARD_VECTOR_SIZE - 1) != 0)
#error The vector size must be a power of two
#endif

static constexpr const idx_t STANDARD_VECTOR_BITS = ComputePowerOfTwoBits(STANDARD_VECTOR_SIZE);
static_assert((idx_t(1) << STANDARD_VECTOR_BITS) == STANDARD_VECTOR_SIZE, "STANDARD_VECTOR_BITS mismatch");

} // namespace duckdb
