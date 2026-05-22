//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/vector/for_vector_comparison.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"

namespace duckdb {

static inline idx_t FORSelectAll(optional_ptr<const SelectionVector> sel, idx_t count,
                                 optional_ptr<SelectionVector> true_sel,
                                 optional_ptr<SelectionVector> false_sel, bool pass) {
	if (auto &t = pass ? true_sel : false_sel) {
		for (idx_t i = 0; i < count; i++) {
			t->set_index(i, sel ? sel->get_index(i) : i);
		}
	}
	return pass ? count : 0;
}

template <class OP>
static int64_t TryFORSelect(Vector &left, Vector &right, optional_ptr<const SelectionVector> sel, idx_t count,
                            optional_ptr<SelectionVector> true_sel, optional_ptr<SelectionVector> false_sel) {
	int64_t rv = -1;
	if (left.GetVectorType() == VectorType::FOR_VECTOR && right.GetVectorType() == VectorType::FOR_VECTOR &&
	    FORVector::HasSameMetadata(left, right)) {
		auto lv = FORVector::CreateStoredView(left);
		auto rv_vec = FORVector::CreateStoredView(right);
		FOR_SWITCH_STORED(FORVector::GetStoredType(left), S, {
			rv = NumericCast<int64_t>(
			    BinaryExecutor::Select<S, S, OP>(lv, rv_vec, sel.get(), count, true_sel.get(), false_sel.get()));
		});
		return rv;
	}

	Vector *for_vec = nullptr;
	Vector *const_vec = nullptr;
	bool for_is_right = false;
	if (left.GetVectorType() == VectorType::FOR_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		for_vec = &left;
		const_vec = &right;
	} else if (right.GetVectorType() == VectorType::FOR_VECTOR && left.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		for_vec = &right;
		const_vec = &left;
		for_is_right = true;
	} else {
		return -1;
	}
	if (ConstantVector::IsNull(*const_vec)) {
		return NumericCast<int64_t>(FORSelectAll(sel, count, true_sel, false_sel, false));
	}

	FOR_SWITCH_LOGICAL(for_vec->GetType().InternalType(), LT, {
		LT constant = ConstantVector::GetData<LT>(*const_vec)[0];
		auto range = FORVector::RangeAnalysis<LT>(*for_vec, constant);
		bool comparison_result;
		if (FORVector::ShortCircuitComparison<OP, LT>(range, for_is_right, comparison_result)) {
			rv = NumericCast<int64_t>(FORSelectAll(sel, count, true_sel, false_sel, comparison_result));
			return rv;
		}
		auto sv = FORVector::CreateStoredView(*for_vec);
		FOR_SWITCH_STORED(FORVector::GetStoredType(*for_vec), S, {
			Vector cv(Value::CreateValue(UnsafeNumericCast<S>(constant)), count_t(1));
			auto &l = for_is_right ? cv : sv;
			auto &r = for_is_right ? sv : cv;
			rv = NumericCast<int64_t>(
			    BinaryExecutor::Select<S, S, OP>(l, r, sel.get(), count, true_sel.get(), false_sel.get()));
		});
	});
	return rv;
}

} // namespace duckdb
