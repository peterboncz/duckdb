#include "catch.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/scalar/compressed_materialization_utils.hpp"
#include "duckdb/function/scalar/operators.hpp"
#include "duckdb/function/scalar/operator_functions.hpp"

using namespace duckdb;

TEST_CASE("FOR Vector - basic creation and flatten", "[for_vector]") {
	// Create a FOR vector: int64 values [1000, 1001, 1002, 1003, 1004]
	// stored directly as uint16 with max 1004
	idx_t count = 5;
	Vector vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, 1004);

	auto stored_data = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint16_t>(1000 + i);
	}

	REQUIRE(vec.GetVectorType() == VectorType::FOR_VECTOR);
	REQUIRE(FORVector::GetStoredType(vec) == PhysicalType::UINT16);
	REQUIRE(FORVector::GetMax<int64_t>(vec) == 1004);

	// Test GetValue
	for (idx_t i = 0; i < count; i++) {
		auto val = vec.GetValue(i);
		REQUIRE(val.GetValue<int64_t>() == static_cast<int64_t>(1000 + i));
	}

	// Test Flatten
	vec.Flatten(count);
	REQUIRE(vec.GetVectorType() == VectorType::FLAT_VECTOR);
	auto flat_data = FlatVector::GetData<int64_t>(vec);
	for (idx_t i = 0; i < count; i++) {
		REQUIRE(flat_data[i] == static_cast<int64_t>(1000 + i));
	}
}

TEST_CASE("FOR Vector - NULL handling", "[for_vector]") {
	Vector vec(LogicalType::INTEGER);
	FORVector::Create<int32_t>(vec, PhysicalType::UINT8, 130);

	auto stored_data = reinterpret_cast<uint8_t *>(FORVector::GetData(vec));
	stored_data[0] = 100;
	stored_data[1] = 110;
	stored_data[2] = 120;
	stored_data[3] = 130;

	// Set element 1 and 3 as NULL
	FORVector::Validity(vec).SetInvalid(1);
	FORVector::Validity(vec).SetInvalid(3);

	REQUIRE(vec.GetValue(0).GetValue<int32_t>() == 100);
	REQUIRE(vec.GetValue(1).IsNull());
	REQUIRE(vec.GetValue(2).GetValue<int32_t>() == 120);
	REQUIRE(vec.GetValue(3).IsNull());
}

TEST_CASE("FOR Vector - comparison with constant", "[for_vector]") {
	idx_t count = 5;

	// Create FOR vector with values [100, 110, 120, 130, 140]
	Vector for_vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(for_vec, PhysicalType::UINT8, 140);
	auto stored_data = reinterpret_cast<uint8_t *>(FORVector::GetData(for_vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint8_t>(100 + i * 10);
	}

	// Test: for_vec == 120
	Vector const_vec(Value::BIGINT(120));
	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	auto match_count = VectorOperations::Equals(for_vec, const_vec, nullptr, count, &true_sel, &false_sel);
	REQUIRE(match_count == 1);
	REQUIRE(true_sel.get_index(0) == 2); // index 2 has value 120

	// Test: for_vec > 120
	match_count = VectorOperations::GreaterThan(for_vec, const_vec, nullptr, count, &true_sel, &false_sel);
	REQUIRE(match_count == 2); // indices 3 (130) and 4 (140)

	// Test: for_vec < 120
	match_count = VectorOperations::LessThan(for_vec, const_vec, nullptr, count, &true_sel, &false_sel);
	REQUIRE(match_count == 2); // indices 0 (100) and 1 (110)

	// Test: for_vec >= 120
	match_count = VectorOperations::GreaterThanEquals(for_vec, const_vec, nullptr, count, &true_sel, &false_sel);
	REQUIRE(match_count == 3); // indices 2, 3, 4
}

TEST_CASE("FOR Vector - comparison with out-of-range constant", "[for_vector]") {
	idx_t count = 3;

	// Create FOR vector with values [100, 101, 102]
	Vector for_vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(for_vec, PhysicalType::UINT8, 102);
	auto stored_data = reinterpret_cast<uint8_t *>(FORVector::GetData(for_vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint8_t>(100 + i);
	}

	// Constant below base: all values > 50
	{
		Vector const_vec(Value::BIGINT(50));
		SelectionVector true_sel(count);
		auto match_count = VectorOperations::GreaterThan(for_vec, const_vec, nullptr, count, &true_sel, nullptr);
		REQUIRE(match_count == 3); // all values > 50
	}

	// Constant above max stored range: all values < 500
	{
		Vector const_vec(Value::BIGINT(500));
		SelectionVector true_sel(count);
		auto match_count = VectorOperations::LessThan(for_vec, const_vec, nullptr, count, &true_sel, nullptr);
		REQUIRE(match_count == 3); // all values < 500
	}
}

//===--------------------------------------------------------------------===//
// Regression tests for specific bugs
//===--------------------------------------------------------------------===//

TEST_CASE("FOR Vector - Slice with non-monotonic selection", "[for_vector]") {
	// Regression: Slice compaction was done in-place, which corrupted data when
	// the selection vector was non-monotonic (e.g. [2, 0] overwrites row 0 before reading it).
	Vector vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, 1040);
	auto stored = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	stored[0] = 1010;
	stored[1] = 1020;
	stored[2] = 1030;
	stored[3] = 1040;

	// Non-monotonic selection: pick rows [2, 0, 3]
	SelectionVector sel(3);
	sel.set_index(0, 2);
	sel.set_index(1, 0);
	sel.set_index(2, 3);
	vec.Slice(sel, 3);

	// After Slice, vec should have 3 values: [1030, 1010, 1040]
	vec.Flatten(3);
	auto data = FlatVector::GetData<int64_t>(vec);
	REQUIRE(data[0] == 1030);
	REQUIRE(data[1] == 1010);
	REQUIRE(data[2] == 1040);
}

TEST_CASE("FOR Vector - Flatten with selection preserves validity", "[for_vector]") {
	// Regression: Flatten(sel, count) copied the original validity mask without
	// remapping through the selection vector, causing wrong NULL positions.
	Vector vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(vec, PhysicalType::UINT8, 100);
	auto stored = reinterpret_cast<uint8_t *>(FORVector::GetData(vec));
	stored[0] = 0;
	stored[1] = 1;
	stored[2] = 2;
	stored[3] = 3;
	// Row 1 is NULL, others valid
	FORVector::Validity(vec).SetInvalid(1);

	// Selection: pick rows [2, 1, 0] → output should be [valid, NULL, valid]
	SelectionVector sel(3);
	sel.set_index(0, 2);
	sel.set_index(1, 1);
	sel.set_index(2, 0);
	vec.Flatten(sel, 3);

	REQUIRE(vec.GetVectorType() == VectorType::FLAT_VECTOR);
	auto data = FlatVector::GetData<int64_t>(vec);
	auto &validity = FlatVector::Validity(vec);
	REQUIRE(validity.RowIsValid(0));  // row 2 was valid
	REQUIRE(!validity.RowIsValid(1)); // row 1 was NULL
	REQUIRE(validity.RowIsValid(2));  // row 0 was valid
	REQUIRE(data[0] == 2);
	REQUIRE(data[2] == 0);
}

TEST_CASE("FOR Vector - Copy with selection beyond source_count", "[for_vector]") {
	// Regression: VectorOperations::Copy for FOR vectors flattened with source_count
	// (which is sel_count, not the vector's actual element count), then used sel to
	// index beyond the partially-decompressed data.
	idx_t full_count = 100;
	Vector vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, 99);
	auto stored = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	for (idx_t i = 0; i < full_count; i++) {
		stored[i] = static_cast<uint16_t>(i);
	}

	// Selection picks 3 rows from high indices
	SelectionVector sel(3);
	sel.set_index(0, 90);
	sel.set_index(1, 50);
	sel.set_index(2, 99);

	Vector target(LogicalType::BIGINT);
	VectorOperations::Copy(vec, target, sel, 3, 0, 0);

	auto data = FlatVector::GetData<int64_t>(target);
	REQUIRE(data[0] == 90);
	REQUIRE(data[1] == 50);
	REQUIRE(data[2] == 99);
}

TEST_CASE("FOR Vector - hugeint with UINT64 stored type", "[for_vector]") {
	// Regression: RangeAnalysis skipped the all_lt check for UINT64 stored type,
	// which was only safe when the logical type was also 64-bit. With HUGEINT FOR
	// vectors backed by UINT64, constants above the FOR max were not detected.
	idx_t count = 3;
	Vector vec(LogicalType::HUGEINT);
	hugeint_t max_value(0, 1200);
	FORVector::Create<hugeint_t>(vec, PhysicalType::UINT64, max_value);
	auto stored = reinterpret_cast<uint64_t *>(FORVector::GetData(vec));
	stored[0] = 1000;
	stored[1] = 1100;
	stored[2] = 1200;

	// Constant far above max: all values should be less than it
	hugeint_t big_constant(10, 0); // way above base + UINT64_MAX
	Vector const_vec(Value::HUGEINT(big_constant));
	SelectionVector true_sel(count);
	auto match_count = VectorOperations::LessThan(vec, const_vec, nullptr, count, &true_sel, nullptr);
	REQUIRE(match_count == 3); // all values < big_constant

	// Also verify Equals returns 0 for out-of-range constant
	match_count = VectorOperations::Equals(vec, const_vec, nullptr, count, &true_sel, nullptr);
	REQUIRE(match_count == 0);
}

//===--------------------------------------------------------------------===//
// Arithmetic helpers
//===--------------------------------------------------------------------===//

// Helper to create a FOR vector with int64 values [start, start+1, ..., start+(count-1)].
static Vector CreateSimpleFOR(int64_t start, idx_t count) {
	Vector vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, start + NumericCast<int64_t>(count - 1));
	auto stored_data = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint16_t>(start + NumericCast<int64_t>(i));
	}
	return vec;
}

// Helper to execute a binary operation using ScalarFunction::BinaryFunction (goes through ExecuteStandard)
template <class T, class OP>
static void ExecuteArithmetic(Vector &left, Vector &right, Vector &result, idx_t count) {
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {left.GetType(), right.GetType()});
	input.data[0].Reference(left);
	input.data[1].Reference(right);
	input.SetCardinality(count);
	ScalarFunction::BinaryFunction<T, T, T, OP>(input, *(ExpressionState *)nullptr, result);
}

static void ExecuteBinaryFunction(ScalarFunction function, Vector &left, Vector &right, Vector &result, idx_t count) {
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {left.GetType(), right.GetType()});
	input.data[0].Reference(left);
	input.data[1].Reference(right);
	input.SetCardinality(count);
	typename std::aligned_storage<sizeof(ExpressionState), alignof(ExpressionState)>::type state_storage;
	auto &dummy_state = *reinterpret_cast<ExpressionState *>(&state_storage);
	function.GetFunctionCallback()(input, dummy_state, result);
}

static ScalarFunction GetMultiplyBigintFunction() {
	for (auto &function : OperatorMultiplyFun::GetFunctions().functions) {
		if (function.arguments.size() == 2 && function.arguments[0] == LogicalType::BIGINT &&
		    function.arguments[1] == LogicalType::BIGINT && function.return_type == LogicalType::BIGINT) {
			return function;
		}
	}
	throw InternalException("BIGINT multiply function not found");
}

TEST_CASE("FOR Vector - addition with constant", "[for_vector]") {
	idx_t count = 5;
	auto for_vec = CreateSimpleFOR(1000, count);

	Vector const_vec(Value::BIGINT(42));
	Vector result(LogicalType::BIGINT);
	ExecuteBinaryFunction(AddFunction::GetFunction(LogicalType::BIGINT, LogicalType::BIGINT), for_vec, const_vec,
	                      result, count);

	REQUIRE(result.GetVectorType() == VectorType::FOR_VECTOR);
	result.Flatten(count);
	auto data = FlatVector::GetData<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		REQUIRE(data[i] == static_cast<int64_t>(1042 + i));
	}
}

TEST_CASE("FOR Vector - multiplication with constant", "[for_vector]") {
	idx_t count = 5;
	auto for_vec = CreateSimpleFOR(100, count);

	Vector const_vec(Value::BIGINT(3));
	Vector result(LogicalType::BIGINT);
	ExecuteBinaryFunction(GetMultiplyBigintFunction(), for_vec, const_vec, result, count);

	REQUIRE(result.GetVectorType() == VectorType::FOR_VECTOR);
	result.Flatten(count);
	auto data = FlatVector::GetData<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		REQUIRE(data[i] == static_cast<int64_t>((100 + i) * 3));
	}
}

TEST_CASE("FOR Vector - addition with negative constant", "[for_vector]") {
	idx_t count = 3;
	auto for_vec = CreateSimpleFOR(1000, count);

	Vector const_vec(Value::BIGINT(-500));
	Vector result(LogicalType::BIGINT);
	BinaryExecutor::ExecuteStandard<int64_t, int64_t, int64_t, AddOperatorOverflowCheck>(for_vec, const_vec, result,
	                                                                                     count);

	result.Flatten(count);
	auto data = FlatVector::GetData<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		REQUIRE(data[i] == static_cast<int64_t>(500 + i));
	}
}

TEST_CASE("FOR Vector - compressed materialization decompress with nonzero min", "[for_vector]") {
	idx_t count = 4;
	Vector compressed(LogicalType::UTINYINT);
	auto compressed_data = FlatVector::GetData<uint8_t>(compressed);
	compressed_data[0] = 0;
	compressed_data[1] = 10;
	compressed_data[2] = 20;
	compressed_data[3] = 30;

	Vector min_vec(Value::BIGINT(1000));
	Vector result(LogicalType::BIGINT);
	auto decompress = CMIntegralDecompressFun::GetFunction(LogicalType::UTINYINT, LogicalType::BIGINT);
	ExecuteBinaryFunction(decompress, compressed, min_vec, result, count);

	REQUIRE(result.GetVectorType() == VectorType::FOR_VECTOR);
	REQUIRE(FORVector::GetStoredType(result) == PhysicalType::UINT16);
	REQUIRE(FORVector::GetMax<int64_t>(result) == 1255);
	result.Flatten(count);
	auto data = FlatVector::GetData<int64_t>(result);
	REQUIRE(data[0] == 1000);
	REQUIRE(data[1] == 1010);
	REQUIRE(data[2] == 1020);
	REQUIRE(data[3] == 1030);
}
