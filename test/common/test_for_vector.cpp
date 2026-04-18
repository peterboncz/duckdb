#include "catch.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/bitpacking.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/scalar/compressed_materialization_utils.hpp"
#include "duckdb/function/scalar/operators.hpp"
#include "duckdb/function/scalar/operator_functions.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/expression_executor_state.hpp"

using namespace duckdb;

TEST_CASE("FOR Vector - basic creation and flatten", "[for_vector]") {
	// Create a FOR vector: int64 values [1000, 1001, 1002, 1003, 1004]
	idx_t count = 5;
	Vector vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, 1000, 3);

	auto stored_data = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint16_t>(i);
	}

	REQUIRE(vec.GetVectorType() == VectorType::FOR_VECTOR);
	REQUIRE(FORVector::GetStoredType(vec) == PhysicalType::UINT16);
	REQUIRE(FORVector::GetMin<int64_t>(vec) == 1000);
	REQUIRE(FORVector::GetMax<int64_t>(vec) == 1007);

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
	FORVector::Create<int32_t>(vec, PhysicalType::UINT8, 100, 5);

	auto stored_data = reinterpret_cast<uint8_t *>(FORVector::GetData(vec));
	stored_data[0] = 0;
	stored_data[1] = 10;
	stored_data[2] = 20;
	stored_data[3] = 30;

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
	FORVector::Create<int64_t>(for_vec, PhysicalType::UINT8, 100, 6);
	auto stored_data = reinterpret_cast<uint8_t *>(FORVector::GetData(for_vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint8_t>(i * 10);
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

TEST_CASE("FOR Vector - comparison between two FOR vectors", "[for_vector]") {
	idx_t count = 4;
	// left:  [100, 200, 150, 300] stored as uint16 delta from 0
	// right: [150, 150, 150, 150] stored as uint16 delta from 0
	Vector left(LogicalType::BIGINT);
	FORVector::Create<int64_t>(left, PhysicalType::UINT16, 0, 9);
	auto ldata = reinterpret_cast<uint16_t *>(FORVector::GetData(left));
	ldata[0] = 100; ldata[1] = 200; ldata[2] = 150; ldata[3] = 300;

	Vector right(LogicalType::BIGINT);
	FORVector::Create<int64_t>(right, PhysicalType::UINT16, 0, 8);
	auto rdata = reinterpret_cast<uint16_t *>(FORVector::GetData(right));
	rdata[0] = 150; rdata[1] = 150; rdata[2] = 150; rdata[3] = 150;

	SelectionVector true_sel(count), false_sel(count);
	auto match = VectorOperations::LessThan(left, right, nullptr, count, &true_sel, &false_sel);
	REQUIRE(match == 1);       // only index 0: 100 < 150
	REQUIRE(true_sel.get_index(0) == 0);

	match = VectorOperations::Equals(left, right, nullptr, count, &true_sel, &false_sel);
	REQUIRE(match == 1);       // only index 2: 150 == 150
	REQUIRE(true_sel.get_index(0) == 2);
}

TEST_CASE("FOR Vector - comparison with out-of-range constant", "[for_vector]") {
	idx_t count = 3;

	// Create FOR vector with values [100, 101, 102]
	Vector for_vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(for_vec, PhysicalType::UINT8, 100, 2);
	auto stored_data = reinterpret_cast<uint8_t *>(FORVector::GetData(for_vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint8_t>(i);
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
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, 1010, 5);
	auto stored = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	stored[0] = 0;
	stored[1] = 10;
	stored[2] = 20;
	stored[3] = 30;

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
	FORVector::Create<int64_t>(vec, PhysicalType::UINT8, 0, 2);
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
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, 0, 7);
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
	FORVector::Create<hugeint_t>(vec, PhysicalType::UINT64, hugeint_t(0, 1000), 8);
	auto stored = reinterpret_cast<uint64_t *>(FORVector::GetData(vec));
	stored[0] = 0;
	stored[1] = 100;
	stored[2] = 200;

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
	FORVector::Create<int64_t>(vec, PhysicalType::UINT16, start,
	                           BitpackingPrimitives::MinimumBitWidth<uint64_t, false>(count - 1));
	auto stored_data = reinterpret_cast<uint16_t *>(FORVector::GetData(vec));
	for (idx_t i = 0; i < count; i++) {
		stored_data[i] = NumericCast<uint16_t>(i);
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
	// Zero-initialized fake state — HasContext() reads executor ptr which will be null → returns false
	alignas(ExpressionState) char state_buf[sizeof(ExpressionState)] = {};
	function.GetFunctionCallback()(input, *reinterpret_cast<ExpressionState *>(state_buf), result);
}

static void ExecuteTernaryFunction(ScalarFunction function, Vector &arg0, Vector &arg1, Vector &arg2, Vector &result,
                                   idx_t count) {
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {arg0.GetType(), arg1.GetType(), arg2.GetType()});
	input.data[0].Reference(arg0);
	input.data[1].Reference(arg1);
	input.data[2].Reference(arg2);
	input.SetCardinality(count);
	alignas(ExpressionState) char state_buf[sizeof(ExpressionState)] = {};
	function.GetFunctionCallback()(input, *reinterpret_cast<ExpressionState *>(state_buf), result);
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
	auto compressed_data = FlatVector::GetDataMutable<uint8_t>(compressed);
	compressed_data[0] = 0;
	compressed_data[1] = 10;
	compressed_data[2] = 20;
	compressed_data[3] = 30;

	Vector min_vec(Value::BIGINT(1000));
	Vector max_vec(Value::BIGINT(1255));
	Vector result(LogicalType::BIGINT);
	auto decompress = CMIntegralDecompressFun::GetFunction(LogicalType::UTINYINT, LogicalType::BIGINT);
	ExecuteTernaryFunction(decompress, compressed, min_vec, max_vec, result, count);

	REQUIRE(result.GetVectorType() == VectorType::FOR_VECTOR);
	REQUIRE(FORVector::GetStoredType(result) == PhysicalType::UINT8);
	REQUIRE(FORVector::GetMin<int64_t>(result) == 1000);
	REQUIRE(FORVector::GetMax<int64_t>(result) == 1255);
	result.Flatten(count);
	auto data = FlatVector::GetData<int64_t>(result);
	REQUIRE(data[0] == 1000);
	REQUIRE(data[1] == 1010);
	REQUIRE(data[2] == 1020);
	REQUIRE(data[3] == 1030);
}

TEST_CASE("FOR Vector - arithmetic between two FOR vectors", "[for_vector]") {
	idx_t count = 4;
	auto left = CreateSimpleFOR(100, count);   // [100, 101, 102, 103]
	auto right = CreateSimpleFOR(200, count);  // [200, 201, 202, 203]
	Vector result(LogicalType::BIGINT);
	ExecuteBinaryFunction(AddFunction::GetFunction(LogicalType::BIGINT, LogicalType::BIGINT), left, right, result, count);
	REQUIRE(result.GetVectorType() == VectorType::FOR_VECTOR); // proves FOR path fired
	result.Flatten(count);
	auto data = FlatVector::GetData<int64_t>(result);
	for (idx_t i = 0; i < count; i++) REQUIRE(data[i] == static_cast<int64_t>(300 + 2 * i));
}

TEST_CASE("FOR Vector - DataChunk::Append preserves FOR", "[for_vector]") {
	// Simulates CachingPhysicalOperator: pre-convert target to FOR, then Append copies narrow data
	DataChunk target;
	target.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});

	// First append: manually set target to FOR (as CachingOperator does), then Append
	auto src1 = CreateSimpleFOR(100, 5);
	target.data[0].SetVectorType(VectorType::FOR_VECTOR);
	FORVector::SetMetadata<int64_t>(target.data[0], FORVector::GetStoredType(src1), FORVector::GetMin<int64_t>(src1),
	                                FORVector::GetRangeBits(src1));
	DataChunk chunk1;
	chunk1.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	chunk1.data[0].Reference(src1);
	chunk1.SetCardinality(5);
	target.Append(chunk1);
	REQUIRE(target.data[0].GetVectorType() == VectorType::FOR_VECTOR);
	REQUIRE(target.size() == 5);

	// Second append: incompatible metadata releases the FOR run and falls back to flat copy
	auto src2 = CreateSimpleFOR(200, 3);
	DataChunk chunk2;
	chunk2.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	chunk2.data[0].Reference(src2);
	chunk2.SetCardinality(3);
	target.Append(chunk2);
	REQUIRE(target.data[0].GetVectorType() == VectorType::FLAT_VECTOR);
	REQUIRE(target.size() == 8);

	// Verify values after flatten
	target.data[0].Flatten(8);
	auto data = FlatVector::GetData<int64_t>(target.data[0]);
	REQUIRE(data[0] == 100);
	REQUIRE(data[4] == 104);
	REQUIRE(data[5] == 200);
	REQUIRE(data[7] == 202);
}

TEST_CASE("FOR Vector - TryWidenType preserves FOR across type widening", "[for_vector]") {
	Vector source(LogicalType::BIGINT);
	FORVector::Create<int64_t>(source, PhysicalType::UINT16, 100, 9);
	auto stored = reinterpret_cast<uint16_t *>(FORVector::GetData(source));
	stored[0] = 0; stored[1] = 100; stored[2] = 400;

	Vector result(LogicalType::HUGEINT);
	REQUIRE(FORVector::TryWidenType(source, result));
	REQUIRE(result.GetVectorType() == VectorType::FOR_VECTOR);
	REQUIRE(FORVector::GetStoredType(result) == PhysicalType::UINT16); // same narrow storage
	// Verify the buffer is shared (zero-copy)
	REQUIRE(FORVector::GetData(result) == FORVector::GetData(source));
	// Verify values decompress correctly as hugeint
	result.Flatten(3);
	auto data = FlatVector::GetData<hugeint_t>(result);
	REQUIRE(data[0] == hugeint_t(0, 100));
	REQUIRE(data[1] == hugeint_t(0, 200));
	REQUIRE(data[2] == hugeint_t(0, 500));
}

TEST_CASE("FOR Vector - compress fast path on FOR input", "[for_vector]") {
	// Create FOR vector simulating scan output, then compress (subtract min_val)
	idx_t count = 4;
	Vector for_vec(LogicalType::BIGINT);
	FORVector::Create<int64_t>(for_vec, PhysicalType::UINT16, 1000, 5);
	auto stored = reinterpret_cast<uint16_t *>(FORVector::GetData(for_vec));
	stored[0] = 0; stored[1] = 10; stored[2] = 20; stored[3] = 30;

	// Compress: subtract min_val=1000, result type UTINYINT (fits 0..30)
	Vector min_vec(Value::BIGINT(1000));
	Vector result(LogicalType::UTINYINT);
	auto compress = CMIntegralCompressFun::GetFunction(LogicalType::BIGINT, LogicalType::UTINYINT);
	ExecuteBinaryFunction(compress, for_vec, min_vec, result, count);

	// Result should be FLAT UTINYINT with values [0, 10, 20, 30]
	auto data = FlatVector::GetData<uint8_t>(result);
	REQUIRE(data[0] == 0);
	REQUIRE(data[1] == 10);
	REQUIRE(data[2] == 20);
	REQUIRE(data[3] == 30);
}
