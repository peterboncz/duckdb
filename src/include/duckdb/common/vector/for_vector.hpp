//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/vector/for_vector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

//! FORVector: compressed vector backed by smaller unsigned integer payload values.
//! Stores non-negative values directly in the payload and keeps the maximum value in metadata.
//!
//! Layout: the narrow stored data lives inside the vector's existing buffer (e.g. a
//! StandardVectorBuffer). The metadata (max value, stored type) lives in auxiliary data.
//! This means FORVector::Create does NOT replace the buffer — it reuses whatever buffer
//! the vector already has. This is critical for compatibility with VectorCache, which
//! restores the original buffer between pipeline iterations.
//! Result of range analysis: whether all FOR values are above or below a constant.
struct FORRangeResult {
	bool all_gt; //! all FOR values > constant
	bool all_lt; //! all FOR values < constant
};

//! Type tag for compile-time type dispatch in DispatchLogicalType.
template <class T>
struct FORTypeTag {
	using type = T;
};

template <class T>
struct FORUnsignedType {
	using type = typename MakeUnsigned<T>::type;
};
template <>
struct FORUnsignedType<hugeint_t> {
	using type = uhugeint_t;
};

struct FORVector {
	template <class T>
	struct ScanData {
		const Vector *for_vec = nullptr;
		const SelectionVector *sel = nullptr;
		PhysicalType stored_type = PhysicalType::INVALID;
		const_data_ptr_t data = nullptr;
		optional_ptr<const ValidityMask> validity;
		T max_value;
	};

	//! Try to get the FOR vector, unwrapping one DICTIONARY layer if needed.
	//! Returns the FOR vector and sets sel to the dictionary's selection (or nullptr if direct FOR).
	static inline const Vector *TryGetFOR(const Vector &vec, const SelectionVector *&sel) {
		if (vec.GetVectorType() == VectorType::FOR_VECTOR) {
			sel = nullptr;
			return &vec;
		}
		if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
			auto &child = DictionaryVector::Child(vec);
			if (child.GetVectorType() == VectorType::FOR_VECTOR) {
				sel = &DictionaryVector::SelVector(vec);
				return &child;
			}
		}
		return nullptr;
	}

	template <class T>
	static inline bool TryGetScanData(const Vector &vec, ScanData<T> &scan_data) {
		const SelectionVector *sel;
		auto *for_vec = TryGetFOR(vec, sel);
		if (!for_vec) {
			return false;
		}
		scan_data.for_vec = for_vec;
		scan_data.sel = sel;
		scan_data.stored_type = GetStoredType(*for_vec);
		scan_data.data = GetData(*for_vec);
		scan_data.validity = Validity(*for_vec);
		scan_data.max_value = GetMax<T>(*for_vec);
		return true;
	}

	template <class T>
	static inline void CopyResultValidity(Vector &result, const ScanData<T> *left_scan, const ScanData<T> *right_scan,
	                                      idx_t count) {
		auto &result_validity = Validity(result);
		result_validity.Reset(count);
		auto apply_validity = [&](const ScanData<T> *scan_data) {
			if (!scan_data || !scan_data->for_vec || !scan_data->validity->CanHaveNull()) {
				return;
			}
			for (idx_t i = 0; i < count; i++) {
				auto src_idx = scan_data->sel ? scan_data->sel->get_index(i) : i;
				if (!scan_data->validity->RowIsValid(src_idx)) {
					result_validity.SetInvalid(i);
				}
			}
		};
		apply_validity(left_scan);
		apply_validity(right_scan);
	}

	template <class STORED_T, class T>
	static inline const STORED_T *CompactData(const ScanData<T> &scan_data, idx_t count,
	                                          unsafe_unique_array<data_t> &compact_buf) {
		auto src = reinterpret_cast<const STORED_T *>(scan_data.data);
		if (!scan_data.sel) {
			return src;
		}
		compact_buf = make_unsafe_uniq_array_uninitialized<data_t>(count * sizeof(STORED_T));
		auto dst = reinterpret_cast<STORED_T *>(compact_buf.get());
		for (idx_t i = 0; i < count; i++) {
			dst[i] = src[scan_data.sel->get_index(i)];
		}
		return dst;
	}

	template <class STORED_T, class T, class FUNC>
	static inline void ForEachValue(const ScanData<T> &scan_data, idx_t count, FUNC &&func) {
		auto data = reinterpret_cast<const STORED_T *>(scan_data.data);
		if (!scan_data.sel) {
			if (!scan_data.validity->CanHaveNull()) {
				for (idx_t i = 0; i < count; i++) {
					func(i, data[i]);
				}
			} else {
				for (idx_t i = 0; i < count; i++) {
					if (scan_data.validity->RowIsValid(i)) {
						func(i, data[i]);
					}
				}
			}
			return;
		}
		for (idx_t i = 0; i < count; i++) {
			auto idx = scan_data.sel->get_index(i);
			if (!scan_data.validity->CanHaveNull() || scan_data.validity->RowIsValid(idx)) {
				func(i, data[idx]);
			}
		}
	}

	static inline data_ptr_t GetData(const Vector &vector) {
		D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
		return vector.buffer->GetData();
	}
	static PhysicalType GetStoredType(const Vector &vector);
	static inline ValidityMask &Validity(const Vector &vector) {
		D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
		return vector.validity;
	}

	//! Get the max value (stored in auxiliary data)
	template <class T>
	static T GetMax(const Vector &vector);

	//! Set FOR metadata (stored_type + max) in auxiliary data
	template <class T>
	static void SetMetadata(Vector &vector, PhysicalType stored_type, T max_value);

	//! Decompress a FOR vector into a flat vector
	static void Decompress(const Vector &source, Vector &target, idx_t count);
	//! Decompress a FOR vector into a flat vector with selection vector
	static void Decompress(const Vector &source, Vector &target, const SelectionVector &sel, idx_t count);

	//! Create a FOR vector. Reuses the existing buffer — does NOT allocate.
	//! Caller must write narrow stored data into GetData(vector) after this call.
	template <class MAX_T>
	static void Create(Vector &vector, PhysicalType stored_type, MAX_T max_value);

	//===--------------------------------------------------------------------===//
	// Shared helpers — used by comparison, filter pushdown, and arithmetic
	//===--------------------------------------------------------------------===//

	//! Map stored PhysicalType to its LogicalType (UINT8→UTINYINT, etc.)
	static LogicalType StoredTypeToLogical(PhysicalType stored_type);

	//! Range analysis for a FOR vector representing values in [0..max_value].
	template <class LOGICAL_T>
	static FORRangeResult RangeAnalysis(LOGICAL_T constant, LOGICAL_T max_value) {
		if (constant < LOGICAL_T(0)) {
			return {true, false};
		}
		if (constant > max_value) {
			return {false, true};
		}
		return {false, false};
	}

	template <class LOGICAL_T>
	static FORRangeResult RangeAnalysis(const Vector &for_vec, LOGICAL_T constant) {
		return RangeAnalysis<LOGICAL_T>(constant, GetMax<LOGICAL_T>(for_vec));
	}

	template <class LOGICAL_T>
	static bool TryGetStoredTypeForMax(LOGICAL_T max_value, PhysicalType &stored_type) {
		if (max_value < LOGICAL_T(0)) {
			return false;
		}
		// Compare in unsigned domain to avoid overflow when LOGICAL_T is signed
		using UNSIGNED_T = typename FORUnsignedType<LOGICAL_T>::type;
		auto umax = static_cast<UNSIGNED_T>(max_value);
		if (umax <= NumericLimits<uint8_t>::Maximum()) {
			stored_type = PhysicalType::UINT8;
			return true;
		}
		if (umax <= NumericLimits<uint16_t>::Maximum()) {
			stored_type = PhysicalType::UINT16;
			return true;
		}
		if (umax <= NumericLimits<uint32_t>::Maximum()) {
			stored_type = PhysicalType::UINT32;
			return true;
		}
		if (umax <= NumericLimits<uint64_t>::Maximum()) {
			stored_type = PhysicalType::UINT64;
			return true;
		}
		return false;
	}

	template <class OP, class LOGICAL_T>
	static bool ShortCircuitComparison(const FORRangeResult &range, bool is_right, bool &res) {
		if (!range.all_gt && !range.all_lt) {
			return false;
		}
		if (range.all_gt) {
			res = is_right ? OP::Operation(LOGICAL_T(0), LOGICAL_T(1)) : OP::Operation(LOGICAL_T(1), LOGICAL_T(0));
		} else {
			res = is_right ? OP::Operation(LOGICAL_T(1), LOGICAL_T(0)) : OP::Operation(LOGICAL_T(0), LOGICAL_T(1));
		}
		return true;
	}

	template <class FUNC>
	static inline bool TryDispatchComparisonConstant(Vector &left, Vector &right, FUNC &&func) {
		Vector *for_vec = nullptr;
		Vector *const_vec = nullptr;
		bool for_is_right = false;

		if (left.GetVectorType() == VectorType::FOR_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			for_vec = &left;
			const_vec = &right;
		} else if (right.GetVectorType() == VectorType::FOR_VECTOR &&
		           left.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			for_vec = &right;
			const_vec = &left;
			for_is_right = true;
		} else {
			return false;
		}

		auto phys_type = for_vec->GetType().InternalType();
		if (GetTypeIdSize(phys_type) <= 1) {
			return false;
		}
		switch (phys_type) {
		case PhysicalType::INT16:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<int16_t> {}, FORTypeTag<uint16_t> {});
		case PhysicalType::UINT16:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<uint16_t> {}, FORTypeTag<uint16_t> {});
		case PhysicalType::INT32:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<int32_t> {}, FORTypeTag<uint32_t> {});
		case PhysicalType::UINT32:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<uint32_t> {}, FORTypeTag<uint32_t> {});
		case PhysicalType::INT64:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<int64_t> {}, FORTypeTag<uint64_t> {});
		case PhysicalType::UINT64:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<uint64_t> {}, FORTypeTag<uint64_t> {});
		case PhysicalType::INT128:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<hugeint_t> {}, FORTypeTag<uhugeint_t> {});
		case PhysicalType::UINT128:
			return func(*for_vec, *const_vec, for_is_right, FORTypeTag<uhugeint_t> {}, FORTypeTag<uhugeint_t> {});
		default:
			throw InternalException("Unsupported logical type for FOR comparison dispatch: %s",
			                        TypeIdToString(phys_type));
		}
	}

	//! Dispatch when both operands are FOR vectors with matching stored type.
	//! Calls func(left_stored_view, right_stored_view, stored_tag).
	template <class FUNC>
	static inline bool TryDispatchComparisonBothFOR(Vector &left, Vector &right, FUNC &&func) {
		if (left.GetVectorType() != VectorType::FOR_VECTOR || right.GetVectorType() != VectorType::FOR_VECTOR)
			return false;
		auto st = GetStoredType(left);
		if (GetTypeIdSize(left.GetType().InternalType()) <= 1 || st != GetStoredType(right))
			return false;
		auto lv = CreateStoredView(left), rv = CreateStoredView(right);
		return DispatchStoredType(st, [&](auto tag) { return func(lv, rv, tag); });
	}

	template <class LOGICAL_T, class FUNC>
	static inline decltype(auto) DispatchConstantInStoredDomain(const Vector &for_vec, LOGICAL_T constant,
	                                                            FUNC &&func) {
		return DispatchStoredType(GetStoredType(for_vec), [&](auto tag) -> decltype(auto) {
			using STORED_T = typename decltype(tag)::type;
			return func(tag, UnsafeNumericCast<STORED_T>(constant));
		});
	}

	template <class OP, class NULL_FUNC, class SHORT_FUNC, class STORED_FUNC>
	static inline bool TryExecuteComparisonConstant(Vector &left, Vector &right, NULL_FUNC &&null_func,
	                                                SHORT_FUNC &&short_func, STORED_FUNC &&stored_func) {
		return TryDispatchComparisonConstant(
		    left, right, [&](Vector &for_vec, Vector &const_vec, bool for_is_right, auto logical_tag, auto) {
			    using LOGICAL_T = typename decltype(logical_tag)::type;
			    if (ConstantVector::IsNull(const_vec)) {
				    null_func(for_vec);
				    return true;
			    }

			    auto constant = ConstantVector::GetData<LOGICAL_T>(const_vec)[0];
			    auto range = RangeAnalysis<LOGICAL_T>(for_vec, constant);
			    bool comparison_result;
			    if (ShortCircuitComparison<OP, LOGICAL_T>(range, for_is_right, comparison_result)) {
				    short_func(for_vec, comparison_result);
				    return true;
			    }

			    return DispatchConstantInStoredDomain(for_vec, constant, [&](auto stored_tag, auto adjusted_const) {
				    stored_func(for_vec, for_is_right, stored_tag, adjusted_const);
				    return true;
			    });
		    });
	}

	//! Try to create a FOR vector from a decoded array of non-negative values.
	//! Returns false if values are negative or can't be narrowed.
	template <class T>
	static inline bool TryCreateFromArray(T *data, idx_t count, Vector &result) {
		if (sizeof(T) <= 1) {
			return false;
		}
		T max_value = 0;
		for (idx_t i = 0; i < count; i++) {
			if (NumericLimits<T>::IsSigned() && data[i] < 0) {
				return false;
			}
			max_value = MaxValue(max_value, data[i]);
		}
		PhysicalType stored_phys;
		if (!TryGetStoredTypeForMax<T>(max_value, stored_phys)) {
			return false;
		}
		Create<T>(result, stored_phys, max_value);
		auto dst = GetData(result);
		DispatchStoredType(stored_phys, [&](auto tag) {
			using STORED_T = typename decltype(tag)::type;
			auto target = reinterpret_cast<STORED_T *>(dst);
			for (idx_t i = 0; i < count; i++) {
				target[i] = UnsafeNumericCast<STORED_T>(data[i]);
			}
		});
		return true;
	}

	//! Transfer FOR metadata to a wider logical type (e.g. int64 FOR → int128 FOR).
	//! Shares the buffer via shared_ptr — zero data copy.
	static bool TryWidenType(Vector &source, Vector &result);

	//! Create a temporary FLAT_VECTOR view over a FOR vector's narrow stored data.
	//! The returned vector references the FOR vector's buffer — caller must not outlive it.
	static Vector CreateStoredView(const Vector &for_vec);

	//! Dispatch a generic lambda over all FOR-supported logical types.
	//! Usage: DispatchLogicalType(phys_type, [&](auto tag) { using T = typename decltype(tag)::type; ... });
	template <class FUNC>
	static inline decltype(auto) DispatchLogicalType(PhysicalType type, FUNC &&func) {
		switch (type) {
		case PhysicalType::INT16:
			return func(FORTypeTag<int16_t> {});
		case PhysicalType::INT32:
			return func(FORTypeTag<int32_t> {});
		case PhysicalType::INT64:
			return func(FORTypeTag<int64_t> {});
		case PhysicalType::INT128:
			return func(FORTypeTag<hugeint_t> {});
		case PhysicalType::UINT16:
			return func(FORTypeTag<uint16_t> {});
		case PhysicalType::UINT32:
			return func(FORTypeTag<uint32_t> {});
		case PhysicalType::UINT64:
			return func(FORTypeTag<uint64_t> {});
		case PhysicalType::UINT128:
			return func(FORTypeTag<uhugeint_t> {});
		default:
			throw InternalException("Unsupported logical type for FOR vector dispatch: %s", TypeIdToString(type));
		}
	}

	//! Dispatch a generic lambda over FOR-supported stored types (UINT8/16/32/64).
	template <class FUNC>
	static inline decltype(auto) DispatchStoredType(PhysicalType type, FUNC &&func) {
		switch (type) {
		case PhysicalType::UINT8:
			return func(FORTypeTag<uint8_t> {});
		case PhysicalType::UINT16:
			return func(FORTypeTag<uint16_t> {});
		case PhysicalType::UINT32:
			return func(FORTypeTag<uint32_t> {});
		case PhysicalType::UINT64:
			return func(FORTypeTag<uint64_t> {});
		default:
			throw InternalException("Unsupported stored type for FOR vector dispatch: %s", TypeIdToString(type));
		}
	}

	//! Widen a stored unsigned value to the logical type.
	//! Uses static_cast: FOR values are guaranteed non-negative and in range by construction.
	template <class LOGICAL_T, class STORED_T>
	static inline LOGICAL_T WidenStored(STORED_T val) {
		return static_cast<LOGICAL_T>(val);
	}
};

//===--------------------------------------------------------------------===//
// FOR arithmetic: operates on narrow stored data, producing narrow FOR result
//===--------------------------------------------------------------------===//
template <class LEFT_T, class RIGHT_T, class RESULT_T, class OP, bool LEFT_CONST, bool RIGHT_CONST>
static inline void FORArithmeticLoop(const LEFT_T *ldata, const RIGHT_T *rdata, RESULT_T *out, idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		auto l = static_cast<RESULT_T>(LEFT_CONST ? ldata[0] : ldata[i]);
		auto r = static_cast<RESULT_T>(RIGHT_CONST ? rdata[0] : rdata[i]);
		out[i] = OP::template Operation<RESULT_T, RESULT_T, RESULT_T>(l, r);
	}
}

//! Try FOR arithmetic on two vectors (each FOR or CONSTANT). BOUNDS::Operation(left_max, right_max, result_max)
//! computes the result bounds. Returns false if inputs aren't suitable.
template <class LOGICAL_T, class OP, class BOUNDS>
static inline bool TryFORArithmetic(Vector &left, Vector &right, Vector &result, idx_t count) {
	struct Operand {
		bool is_for = false;
		LOGICAL_T max_value {};
		FORVector::ScanData<LOGICAL_T> scan_data;
	};
	auto get_operand = [](Vector &vector, Operand &op) {
		op.is_for = FORVector::TryGetScanData(vector, op.scan_data);
		if (op.is_for) {
			op.max_value = op.scan_data.max_value;
			return true;
		}
		if (vector.GetVectorType() != VectorType::CONSTANT_VECTOR || ConstantVector::IsNull(vector)) {
			return false;
		}
		using SIGNED_T = typename MakeSigned<LOGICAL_T>::type;
		auto val = ConstantVector::GetDataUnsafe<SIGNED_T>(vector)[0];
		if (val < 0) {
			return false;
		}
		op.max_value = static_cast<LOGICAL_T>(val);
		return true;
	};

	Operand lop, rop;
	if (!get_operand(left, lop) || !get_operand(right, rop) || (!lop.is_for && !rop.is_for)) {
		return false;
	}
	LOGICAL_T result_max;
	if (!BOUNDS::template Operation<LOGICAL_T>(lop.max_value, rop.max_value, result_max)) {
		return false;
	}
	PhysicalType rst;
	if (!FORVector::TryGetStoredTypeForMax<LOGICAL_T>(result_max, rst)) {
		return false;
	}

	FORVector::Create<LOGICAL_T>(result, rst, result_max);
	FORVector::CopyResultValidity(result, lop.is_for ? &lop.scan_data : nullptr, rop.is_for ? &rop.scan_data : nullptr,
	                              count);
	auto payload = FORVector::GetData(result);

	return FORVector::DispatchStoredType(rst, [&](auto rtag) {
		using RS = typename decltype(rtag)::type;
		auto rdata = reinterpret_cast<RS *>(payload);
		if (!lop.is_for) {
			auto c = UnsafeNumericCast<RS>(lop.max_value);
			return FORVector::DispatchStoredType(rop.scan_data.stored_type, [&](auto tag) {
				using S = typename decltype(tag)::type;
				unsafe_unique_array<data_t> buf;
				FORArithmeticLoop<RS, S, RS, OP, true, false>(&c, FORVector::CompactData<S>(rop.scan_data, count, buf),
				                                              rdata, count);
				return true;
			});
		}
		if (!rop.is_for) {
			auto c = UnsafeNumericCast<RS>(rop.max_value);
			return FORVector::DispatchStoredType(lop.scan_data.stored_type, [&](auto tag) {
				using S = typename decltype(tag)::type;
				unsafe_unique_array<data_t> buf;
				FORArithmeticLoop<S, RS, RS, OP, false, true>(FORVector::CompactData<S>(lop.scan_data, count, buf), &c,
				                                              rdata, count);
				return true;
			});
		}
		return FORVector::DispatchStoredType(lop.scan_data.stored_type, [&](auto ltag) {
			using LS = typename decltype(ltag)::type;
			unsafe_unique_array<data_t> lbuf;
			auto ld = FORVector::CompactData<LS>(lop.scan_data, count, lbuf);
			return FORVector::DispatchStoredType(rop.scan_data.stored_type, [&](auto rtag2) {
				using RRS = typename decltype(rtag2)::type;
				unsafe_unique_array<data_t> rbuf;
				FORArithmeticLoop<LS, RRS, RS, OP, false, false>(
				    ld, FORVector::CompactData<RRS>(rop.scan_data, count, rbuf), rdata, count);
				return true;
			});
		});
	});
}

// Specializations for hugeint_t/uhugeint_t: construct from (0, val)
#define FOR_WIDEN_HUGEINT(LOGICAL, STORED)                                                                             \
	template <>                                                                                                        \
	inline LOGICAL FORVector::WidenStored<LOGICAL, STORED>(STORED val) {                                               \
		return LOGICAL(0, val);                                                                                        \
	}
FOR_WIDEN_HUGEINT(hugeint_t, uint8_t)
FOR_WIDEN_HUGEINT(hugeint_t, uint16_t)
FOR_WIDEN_HUGEINT(hugeint_t, uint32_t)
FOR_WIDEN_HUGEINT(hugeint_t, uint64_t)
FOR_WIDEN_HUGEINT(uhugeint_t, uint8_t)
FOR_WIDEN_HUGEINT(uhugeint_t, uint16_t)
FOR_WIDEN_HUGEINT(uhugeint_t, uint32_t)
FOR_WIDEN_HUGEINT(uhugeint_t, uint64_t)
#undef FOR_WIDEN_HUGEINT

} // namespace duckdb
