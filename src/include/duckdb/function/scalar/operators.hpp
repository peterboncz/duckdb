//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/operators.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/function_set.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/built_in_functions.hpp"

namespace duckdb {

struct AddFunction {
	static ScalarFunction GetFunction(const LogicalType &type);
	static ScalarFunction GetFunction(const LogicalType &left_type, const LogicalType &right_type);
};

struct SubtractFunction {
	static ScalarFunction GetFunction(const LogicalType &type);
	static ScalarFunction GetFunction(const LogicalType &left_type, const LogicalType &right_type);
};

//! The plain integer +,-,* kernel at a narrow physical type, for arithmetic over FOR payloads
scalar_function_t GetForArithmeticFunction(char op, PhysicalType type);

} // namespace duckdb
