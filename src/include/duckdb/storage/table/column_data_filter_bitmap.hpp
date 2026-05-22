//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/column_data_filter_bitmap.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/table_filter.hpp"

#include <cstring>

namespace duckdb {

static constexpr idx_t COLUMN_FILTER_BITMAP_WORDS = (STANDARD_VECTOR_SIZE + 63) / 64;

static inline ExpressionType ColumnFilterBitmapFlipComparisonType(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_LESSTHAN:
		return ExpressionType::COMPARE_GREATERTHAN;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	case ExpressionType::COMPARE_GREATERTHAN:
		return ExpressionType::COMPARE_LESSTHAN;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ExpressionType::COMPARE_LESSTHANOREQUALTO;
	default:
		return type;
	}
}

static inline bool ColumnFilterBitmapIsSupportedConstantComparison(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOTEQUAL:
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return true;
	default:
		return false;
	}
}

static inline bool ColumnFilterBitmapIsBoundColumnRef(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_REF) {
		return false;
	}
	return expr.Cast<BoundReferenceExpression>().index == 0;
}

static inline bool ColumnFilterBitmapTryGetConstantComparison(const Expression &expr, ExpressionType &comparison_type,
                                                              Value &constant) {
	if (!BoundComparisonExpression::IsComparison(expr.GetExpressionType())) {
		return false;
	}
	auto &comparison = expr.Cast<BoundFunctionExpression>();
	comparison_type = comparison.GetExpressionType();
	auto &left = BoundComparisonExpression::Left(comparison);
	auto &right = BoundComparisonExpression::Right(comparison);

	if (ColumnFilterBitmapIsBoundColumnRef(left) && right.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		constant = right.Cast<BoundConstantExpression>().value;
		return ColumnFilterBitmapIsSupportedConstantComparison(comparison_type) && !constant.IsNull();
	}
	if (ColumnFilterBitmapIsBoundColumnRef(right) && left.GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		comparison_type = ColumnFilterBitmapFlipComparisonType(comparison_type);
		constant = left.Cast<BoundConstantExpression>().value;
		return ColumnFilterBitmapIsSupportedConstantComparison(comparison_type) && !constant.IsNull();
	}
	return false;
}

static inline void ColumnFilterBitmapSetAll(validity_t *bm) {
	memset(bm, 0xFF, COLUMN_FILTER_BITMAP_WORDS * sizeof(validity_t));
}

static inline void ColumnFilterBitmapAnd(validity_t *__restrict dst, const validity_t *__restrict src) {
	for (idx_t i = 0; i < COLUMN_FILTER_BITMAP_WORDS; i++) {
		dst[i] &= src[i];
	}
}

static inline void ColumnFilterBitmapOr(validity_t *__restrict dst, const validity_t *__restrict src) {
	for (idx_t i = 0; i < COLUMN_FILTER_BITMAP_WORDS; i++) {
		dst[i] |= src[i];
	}
}

static inline void ColumnFilterBitmapApplyValidity(Vector &result, idx_t count, validity_t *__restrict bitmap) {
	const auto &validity =
	    result.GetVectorType() == VectorType::FOR_VECTOR ? FORVector::Validity(result) : FlatVector::Validity(result);
	if (validity.CannotHaveNull()) {
		return;
	}
	const auto nwords = (count + 63) / 64;
	for (idx_t i = 0; i < nwords; i++) {
		bitmap[i] &= validity.GetValidityEntry(i);
	}
}

static inline void ColumnFilterBitmapAndSelection(validity_t *__restrict bitmap, idx_t count,
                                                  const SelectionVector &sel, idx_t sel_count) {
	validity_t sel_bitmap[COLUMN_FILTER_BITMAP_WORDS];
	memset(sel_bitmap, 0, sizeof(sel_bitmap));
	for (idx_t i = 0; i < sel_count; i++) {
		auto idx = sel.get_index(i);
		sel_bitmap[idx / 64] |= validity_t(1) << (idx % 64);
	}
	const auto nwords = (count + 63) / 64;
	for (idx_t i = 0; i < nwords; i++) {
		bitmap[i] &= sel_bitmap[i];
	}
}

template <class T, class OP>
static void ColumnFilterBitmapCompareOp(const T *__restrict data, T constant, idx_t count, validity_t *__restrict bm) {
	uint8_t cmp[STANDARD_VECTOR_SIZE];
	for (idx_t i = 0; i < count; i++) {
		cmp[i] = OP::Operation(data[i], constant);
	}
	auto *out = reinterpret_cast<uint8_t *>(bm);
	idx_t full = count / 8;
	for (idx_t b = 0; b < full; b++) {
		auto *c = cmp + b * 8;
		out[b] = c[0] | (c[1] << 1) | (c[2] << 2) | (c[3] << 3) | (c[4] << 4) | (c[5] << 5) | (c[6] << 6) |
		         (c[7] << 7);
	}
	if (full * 8 < count) {
		uint8_t tail = 0;
		for (idx_t i = full * 8; i < count; i++) {
			tail |= cmp[i] << (i - full * 8);
		}
		out[full] = tail;
	}
}

template <class T>
static void ColumnFilterBitmapCompare(const T *data, T constant, idx_t count, ExpressionType cmp, validity_t *bm) {
	switch (cmp) {
	case ExpressionType::COMPARE_EQUAL:
		ColumnFilterBitmapCompareOp<T, Equals>(data, constant, count, bm);
		break;
	case ExpressionType::COMPARE_NOTEQUAL:
		ColumnFilterBitmapCompareOp<T, NotEquals>(data, constant, count, bm);
		break;
	case ExpressionType::COMPARE_LESSTHAN:
		ColumnFilterBitmapCompareOp<T, LessThan>(data, constant, count, bm);
		break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		ColumnFilterBitmapCompareOp<T, LessThanEquals>(data, constant, count, bm);
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
		ColumnFilterBitmapCompareOp<T, GreaterThan>(data, constant, count, bm);
		break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		ColumnFilterBitmapCompareOp<T, GreaterThanEquals>(data, constant, count, bm);
		break;
	default:
		break;
	}
}

template <class T>
static bool ColumnFilterBitmapEvalFilter(const T *data, idx_t count, const TableFilter &filter, validity_t *result) {
	switch (filter.filter_type) {
	case TableFilterType::LEGACY_CONSTANT_COMPARISON: {
		auto &cf = filter.Cast<LegacyConstantFilter>();
		ColumnFilterBitmapCompare<T>(data, cf.constant.GetValueUnsafe<T>(), count, cf.comparison_type, result);
		return true;
	}
	case TableFilterType::LEGACY_CONJUNCTION_AND: {
		auto &conj = filter.Cast<LegacyConjunctionAndFilter>();
		ColumnFilterBitmapSetAll(result);
		validity_t child[COLUMN_FILTER_BITMAP_WORDS];
		for (auto &ch : conj.child_filters) {
			if (!ColumnFilterBitmapEvalFilter<T>(data, count, *ch, child)) {
				return false;
			}
			ColumnFilterBitmapAnd(result, child);
		}
		return true;
	}
	case TableFilterType::LEGACY_CONJUNCTION_OR: {
		auto &conj = filter.Cast<LegacyConjunctionOrFilter>();
		memset(result, 0, sizeof(validity_t) * COLUMN_FILTER_BITMAP_WORDS);
		validity_t child[COLUMN_FILTER_BITMAP_WORDS];
		for (auto &ch : conj.child_filters) {
			if (!ColumnFilterBitmapEvalFilter<T>(data, count, *ch, child)) {
				return false;
			}
			ColumnFilterBitmapOr(result, child);
		}
		return true;
	}
	case TableFilterType::LEGACY_IS_NOT_NULL:
	case TableFilterType::LEGACY_IS_NULL:
		ColumnFilterBitmapSetAll(result);
		return true;
	default:
		return false;
	}
}

template <class LT>
static bool ColumnFilterBitmapConstantResult(ExpressionType comparison_type, bool all_gt, bool all_lt) {
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return false;
	case ExpressionType::COMPARE_NOTEQUAL:
		return true;
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return all_lt;
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return all_gt;
	default:
		throw InternalException("Unsupported comparison type for FOR bitmap filter");
	}
}

template <class LT, class ST>
static bool ColumnFilterBitmapEvalExpressionFOR(const ST *data, idx_t count, const Expression &expr, validity_t *result,
                                                LT max_value);

template <class LT, class ST>
static bool ColumnFilterBitmapEvalConstantExpressionFOR(const ST *data, idx_t count, const Expression &expr,
                                                        validity_t *result, LT max_value) {
	ExpressionType comparison_type;
	Value constant_value;
	if (!ColumnFilterBitmapTryGetConstantComparison(expr, comparison_type, constant_value)) {
		return false;
	}
	auto constant = constant_value.GetValueUnsafe<LT>();
	auto range = FORVector::RangeAnalysis<LT>(constant, max_value);
	if (range.all_gt || range.all_lt) {
		if (ColumnFilterBitmapConstantResult<LT>(comparison_type, range.all_gt, range.all_lt)) {
			ColumnFilterBitmapSetAll(result);
		} else {
			memset(result, 0, sizeof(validity_t) * COLUMN_FILTER_BITMAP_WORDS);
		}
		return true;
	}
	ColumnFilterBitmapCompare<ST>(data, UnsafeNumericCast<ST>(constant), count, comparison_type, result);
	return true;
}

template <class LT, class ST>
static bool ColumnFilterBitmapEvalConjunctionExpressionFOR(const ST *data, idx_t count,
                                                           const BoundConjunctionExpression &conj, validity_t *result,
                                                           LT max_value) {
	validity_t child[COLUMN_FILTER_BITMAP_WORDS];
	switch (conj.GetExpressionType()) {
	case ExpressionType::CONJUNCTION_AND:
		ColumnFilterBitmapSetAll(result);
		for (auto &expr : conj.children) {
			if (!ColumnFilterBitmapEvalExpressionFOR<LT, ST>(data, count, *expr, child, max_value)) {
				return false;
			}
			ColumnFilterBitmapAnd(result, child);
		}
		return true;
	case ExpressionType::CONJUNCTION_OR:
		memset(result, 0, sizeof(validity_t) * COLUMN_FILTER_BITMAP_WORDS);
		for (auto &expr : conj.children) {
			if (!ColumnFilterBitmapEvalExpressionFOR<LT, ST>(data, count, *expr, child, max_value)) {
				return false;
			}
			ColumnFilterBitmapOr(result, child);
		}
		return true;
	default:
		return false;
	}
}

template <class LT, class ST>
static bool ColumnFilterBitmapEvalExpressionFOR(const ST *data, idx_t count, const Expression &expr,
                                                validity_t *result, LT max_value) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION) {
		return ColumnFilterBitmapEvalConjunctionExpressionFOR<LT, ST>(
		    data, count, expr.Cast<BoundConjunctionExpression>(), result, max_value);
	}
	return ColumnFilterBitmapEvalConstantExpressionFOR<LT, ST>(data, count, expr, result, max_value);
}

template <class LT, class ST>
static bool ColumnFilterBitmapEvalFOR(const ST *data, idx_t count, const TableFilter &filter, validity_t *result,
                                      LT max_value) {
	switch (filter.filter_type) {
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expr_filter = filter.Cast<ExpressionFilter>();
		return ColumnFilterBitmapEvalExpressionFOR<LT, ST>(data, count, *expr_filter.expr, result, max_value);
	}
	case TableFilterType::LEGACY_CONSTANT_COMPARISON: {
		auto &cf = filter.Cast<LegacyConstantFilter>();
		auto constant = cf.constant.GetValueUnsafe<LT>();
		auto range = FORVector::RangeAnalysis<LT>(constant, max_value);
		if (range.all_gt || range.all_lt) {
			if (ColumnFilterBitmapConstantResult<LT>(cf.comparison_type, range.all_gt, range.all_lt)) {
				ColumnFilterBitmapSetAll(result);
			} else {
				memset(result, 0, sizeof(validity_t) * COLUMN_FILTER_BITMAP_WORDS);
			}
			return true;
		}
		ColumnFilterBitmapCompare<ST>(data, UnsafeNumericCast<ST>(constant), count, cf.comparison_type, result);
		return true;
	}
	case TableFilterType::LEGACY_CONJUNCTION_AND: {
		auto &conj = filter.Cast<LegacyConjunctionAndFilter>();
		ColumnFilterBitmapSetAll(result);
		validity_t child[COLUMN_FILTER_BITMAP_WORDS];
		for (auto &ch : conj.child_filters) {
			if (!ColumnFilterBitmapEvalFOR<LT, ST>(data, count, *ch, child, max_value)) {
				return false;
			}
			ColumnFilterBitmapAnd(result, child);
		}
		return true;
	}
	case TableFilterType::LEGACY_CONJUNCTION_OR: {
		auto &conj = filter.Cast<LegacyConjunctionOrFilter>();
		memset(result, 0, sizeof(validity_t) * COLUMN_FILTER_BITMAP_WORDS);
		validity_t child[COLUMN_FILTER_BITMAP_WORDS];
		for (auto &ch : conj.child_filters) {
			if (!ColumnFilterBitmapEvalFOR<LT, ST>(data, count, *ch, child, max_value)) {
				return false;
			}
			ColumnFilterBitmapOr(result, child);
		}
		return true;
	}
	default:
		return ColumnFilterBitmapEvalFilter<ST>(data, count, filter, result);
	}
}

static inline bool TryFORBitmapFilter(Vector &result, idx_t count, const TableFilter &filter, validity_t *bitmap) {
	if (result.GetVectorType() != VectorType::FOR_VECTOR) {
		return false;
	}
	auto phys = result.GetType().InternalType();
	if (GetTypeIdSize(phys) <= 1) {
		return false;
	}
	FOR_SWITCH_LOGICAL(phys, LT, {
		auto stored_type = FORVector::GetStoredType(result);
		auto *data = FORVector::GetData(result);
		auto max_value = FORVector::GetMax<LT>(result);
		FOR_SWITCH_STORED(stored_type, ST, {
			return ColumnFilterBitmapEvalFOR<LT, ST>(reinterpret_cast<const ST *>(data), count, filter, bitmap,
			                                         max_value);
		});
	});
}

} // namespace duckdb
