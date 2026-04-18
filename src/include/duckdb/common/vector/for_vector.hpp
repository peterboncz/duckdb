//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/vector/for_vector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

//! FORVector: compressed vector backed by smaller unsigned integer delta payload values.
//! Stores logical values as min + delta, where min lives in metadata and delta is stored
//! in the vector payload using UINT8/16/32/64.
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

#define FOR_SWITCH_LOGICAL(TYPE, TYPE_NAME, ...)                                                                      \
	switch (TYPE) {                                                                                                  \
	case PhysicalType::INT16: {                                                                                      \
		typedef int16_t TYPE_NAME;                                                                                   \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::INT32: {                                                                                      \
		typedef int32_t TYPE_NAME;                                                                                   \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::INT64: {                                                                                      \
		typedef int64_t TYPE_NAME;                                                                                   \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::INT128: {                                                                                     \
		typedef hugeint_t TYPE_NAME;                                                                                 \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT16: {                                                                                     \
		typedef uint16_t TYPE_NAME;                                                                                  \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT32: {                                                                                     \
		typedef uint32_t TYPE_NAME;                                                                                  \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT64: {                                                                                     \
		typedef uint64_t TYPE_NAME;                                                                                  \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT128: {                                                                                    \
		typedef uhugeint_t TYPE_NAME;                                                                                \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	default:                                                                                                         \
		throw InternalException("Unsupported logical type for FOR vector dispatch: %s", TypeIdToString(TYPE));       \
	}

#define FOR_SWITCH_STORED(TYPE, TYPE_NAME, ...)                                                                       \
	switch (TYPE) {                                                                                                  \
	case PhysicalType::UINT8: {                                                                                      \
		typedef uint8_t TYPE_NAME;                                                                                   \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT16: {                                                                                     \
		typedef uint16_t TYPE_NAME;                                                                                  \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT32: {                                                                                     \
		typedef uint32_t TYPE_NAME;                                                                                  \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	case PhysicalType::UINT64: {                                                                                     \
		typedef uint64_t TYPE_NAME;                                                                                  \
		__VA_ARGS__;                                                                                                 \
		break;                                                                                                       \
	}                                                                                                                \
	default:                                                                                                         \
		throw InternalException("Unsupported stored type for FOR vector dispatch: %s", TypeIdToString(TYPE));        \
	}

template <class T>
struct FORUnsignedType {
	using type = typename MakeUnsigned<T>::type;
};
template <>
struct FORUnsignedType<hugeint_t> {
	using type = uhugeint_t;
};

template <class T, bool IS_SIGNED = NumericLimits<T>::IsSigned()>
struct FORValueOps;

template <class T>
struct FORValueOps<T, false> {
	static inline uhugeint_t ToUnsignedStorage(T value) {
		return uhugeint_t(UnsafeNumericCast<uint64_t>(value));
	}
	static inline T FromUnsignedStorage(const uhugeint_t &value) {
		return UnsafeNumericCast<T>(value.lower);
	}
	static inline T AddDelta(T min_value, uint64_t delta) {
		return UnsafeNumericCast<T>(uhugeint_t(min_value) + uhugeint_t(delta));
	}
	static inline bool TryGetDelta(T value, T min_value, uint64_t &delta) {
		if (value < min_value) {
			return false;
		}
		delta = UnsafeNumericCast<uint64_t>(value - min_value);
		return true;
	}
};

template <class T>
struct FORValueOps<T, true> {
	static inline uhugeint_t ToUnsignedStorage(T value) {
		return static_cast<uhugeint_t>(Hugeint::Convert(value));
	}
	static inline T FromUnsignedStorage(const uhugeint_t &value) {
		return UnsafeNumericCast<T>(static_cast<hugeint_t>(value));
	}
	static inline T AddDelta(T min_value, uint64_t delta) {
		return UnsafeNumericCast<T>(Hugeint::Convert(min_value) + hugeint_t(0, delta));
	}
	static inline bool TryGetDelta(T value, T min_value, uint64_t &delta) {
		if (value < min_value) {
			return false;
		}
		auto diff = Hugeint::Subtract(Hugeint::Convert(value), Hugeint::Convert(min_value));
		auto udiff = static_cast<uhugeint_t>(diff);
		if (udiff.upper != 0) {
			return false;
		}
		delta = udiff.lower;
		return true;
	}
};

template <>
struct FORValueOps<hugeint_t, true> {
	static inline uhugeint_t ToUnsignedStorage(hugeint_t value) {
		return static_cast<uhugeint_t>(value);
	}
	static inline hugeint_t FromUnsignedStorage(const uhugeint_t &value) {
		return static_cast<hugeint_t>(value);
	}
	static inline hugeint_t AddDelta(hugeint_t min_value, uint64_t delta) {
		return min_value + hugeint_t(0, delta);
	}
	static inline bool TryGetDelta(hugeint_t value, hugeint_t min_value, uint64_t &delta) {
		if (value < min_value) {
			return false;
		}
		auto udiff = static_cast<uhugeint_t>(value - min_value);
		if (udiff.upper != 0) {
			return false;
		}
		delta = udiff.lower;
		return true;
	}
};

template <>
struct FORValueOps<uhugeint_t, false> {
	static inline uhugeint_t ToUnsignedStorage(uhugeint_t value) {
		return value;
	}
	static inline uhugeint_t FromUnsignedStorage(const uhugeint_t &value) {
		return value;
	}
	static inline uhugeint_t AddDelta(uhugeint_t min_value, uint64_t delta) {
		return min_value + uhugeint_t(delta);
	}
	static inline bool TryGetDelta(uhugeint_t value, uhugeint_t min_value, uint64_t &delta) {
		if (value < min_value) {
			return false;
		}
		auto diff = value - min_value;
		if (diff.upper != 0) {
			return false;
		}
		delta = diff.lower;
		return true;
	}
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
	static inline void ForEachValue(const ScanData<T> &scan_data, idx_t lo, idx_t hi, FUNC &&func) {
		auto data = reinterpret_cast<const STORED_T *>(scan_data.data);
		if (!scan_data.sel) {
			if (!scan_data.validity->CanHaveNull()) {
				for (idx_t i = lo; i < hi; i++) func(i, data[i]);
			} else {
				for (idx_t i = lo; i < hi; i++)
					if (scan_data.validity->RowIsValid(i)) func(i, data[i]);
			}
		} else if (!scan_data.validity->CanHaveNull()) {
			for (idx_t i = lo; i < hi; i++) func(i, data[scan_data.sel->get_index(i)]);
		} else {
			for (idx_t i = lo; i < hi; i++) {
				auto idx = scan_data.sel->get_index(i);
				if (scan_data.validity->RowIsValid(idx)) func(i, data[idx]);
			}
		}
	}

	template <class INPUT_TYPE, class FUNC>
	static inline bool DispatchStoredData(const ScanData<INPUT_TYPE> &scan_data, idx_t count, FUNC &func) {
		FOR_SWITCH_STORED(scan_data.stored_type, STORED_T, {
			ForEachValue<STORED_T>(scan_data, 0, count, func);
			return true;
		});
	}

	static inline data_ptr_t GetData(const Vector &vector) {
		D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
		return vector.buffer->GetData();
	}
	static PhysicalType GetStoredType(const Vector &vector);
	static uint8_t GetRangeBits(const Vector &vector);
	static inline ValidityMask &Validity(const Vector &vector) {
		D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
		return vector.buffer->GetValidityMask();
	}

	template <class T>
	static inline T GetMin(const Vector &) {
		return T(0);
	}

	template <class T>
	static inline T AddDelta(T min_value, uint64_t delta) {
		return FORValueOps<T>::AddDelta(min_value, delta);
	}

	template <class T>
	static inline bool TryGetDelta(T value, T min_value, uint64_t &delta) {
		return FORValueOps<T>::TryGetDelta(value, min_value, delta);
	}

	//! Get the max logical value stored in metadata.
	template <class T>
	static T GetMax(const Vector &vector);

	//! Set FOR metadata (stored_type + max) in auxiliary data.
	template <class T>
	static void SetMetadata(Vector &vector, PhysicalType stored_type, T max_value);

	template <class T>
	static inline void SetMetadata(Vector &vector, PhysicalType stored_type, T min_value, uint8_t range_bits) {
		auto max_value = AddDelta(min_value, range_bits >= 64 ? NumericLimits<uint64_t>::Maximum()
		                                                      : (range_bits == 0 ? 0 : ((uint64_t(1) << range_bits) - 1)));
		SetMetadata<T>(vector, stored_type, max_value);
	}

	//! Decompress a FOR vector into a flat vector
	static void Decompress(const Vector &source, Vector &target, idx_t count);
	//! Decompress a FOR vector into a flat vector with selection vector
	static void Decompress(const Vector &source, Vector &target, const SelectionVector &sel, idx_t count);
	//! Copy FOR narrow data directly to a FLAT target, widening per element.
	//! Avoids the allocating Flatten path when use_count > 1.
	static void CopyToFlat(const Vector &source, const SelectionVector &sel, Vector &target,
	                        idx_t source_offset, idx_t target_offset, idx_t copy_count);

	//! Create a FOR vector. Reuses the existing buffer — does NOT allocate.
	//! Caller must write narrow stored data into GetData(vector) after this call.
	template <class T>
	static void Create(Vector &vector, PhysicalType stored_type, T max_value);

	template <class T>
	static inline void Create(Vector &vector, PhysicalType stored_type, T min_value, uint8_t range_bits) {
		vector.vector_type = VectorType::FOR_VECTOR;
		vector.buffer->GetValidityMask().Reset();
		SetMetadata<T>(vector, stored_type, min_value, range_bits);
	}

	//===--------------------------------------------------------------------===//
	// Shared helpers — used by comparison, filter pushdown, and arithmetic
	//===--------------------------------------------------------------------===//

	//! Map stored PhysicalType to its LogicalType (UINT8→UTINYINT, etc.)
	static LogicalType StoredTypeToLogical(PhysicalType stored_type);

	template <class LOGICAL_T>
	static bool TryGetStoredTypeForMax(LOGICAL_T max_value, PhysicalType &stored_type) {
		if (max_value < LOGICAL_T(0)) {
			return false;
		}
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

	static inline bool TryGetStoredTypeForRangeBits(uint8_t range_bits, PhysicalType &stored_type) {
		if (range_bits <= 8) {
			stored_type = PhysicalType::UINT8;
			return true;
		}
		if (range_bits <= 16) {
			stored_type = PhysicalType::UINT16;
			return true;
		}
		if (range_bits <= 32) {
			stored_type = PhysicalType::UINT32;
			return true;
		}
		if (range_bits <= 64) {
			stored_type = PhysicalType::UINT64;
			return true;
		}
		return false;
	}

	static inline bool PackedAccumulationSafe(const Vector &for_vec, uint8_t count_bits, uint8_t shift) {
		auto range_bits = GetRangeBits(for_vec);
		return range_bits + count_bits <= shift;
	}

	static inline bool PlainAccumulationSafe(const Vector &for_vec, uint8_t extra_bits) {
		auto range_bits = GetRangeBits(for_vec);
		return range_bits + extra_bits < 63;
	}

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
	static FORRangeResult RangeAnalysis(LOGICAL_T constant, LOGICAL_T min_value, LOGICAL_T max_value) {
		if (constant < min_value) {
			return {true, false};
		}
		if (constant > max_value) {
			return {false, true};
		}
		return {false, false};
	}

	template <class LOGICAL_T>
	static FORRangeResult RangeAnalysis(const Vector &for_vec, LOGICAL_T constant) {
		return RangeAnalysis<LOGICAL_T>(constant, GetMin<LOGICAL_T>(for_vec), GetMax<LOGICAL_T>(for_vec));
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

	//! Try to create a FOR vector from decoded logical values.
	template <class T>
	static inline bool TryCreateFromArray(T *data, idx_t count, Vector &result) {
		if (sizeof(T) <= 1) {
			return false;
		}
		T max_value = data[0];
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
		FOR_SWITCH_STORED(stored_phys, STORED_T, {
			auto target = reinterpret_cast<STORED_T *>(dst);
			for (idx_t i = 0; i < count; i++) {
				target[i] = UnsafeNumericCast<STORED_T>(data[i]);
			}
		});
		return true;
	}

	static inline bool HasSameMetadata(const Vector &left, const Vector &right) {
		return GetStoredType(left) == GetStoredType(right);
	}

	//! Transfer FOR metadata to a wider logical type (e.g. int64 FOR → int128 FOR).
	//! Shares the buffer via shared_ptr — zero data copy.
	static bool TryWidenType(Vector &source, Vector &result);

	//! Create a temporary FLAT_VECTOR view over a FOR vector's narrow stored data.
	//! The returned vector references the FOR vector's buffer — caller must not outlive it.
	static Vector CreateStoredView(const Vector &for_vec);

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
template <class LOGICAL_T, class OP, class BOUNDS>
static inline bool TryFORArithmetic(Vector &left, Vector &right, Vector &result, idx_t count) {
	struct Operand {
		bool is_for = false;
		LOGICAL_T constant_value {};
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
		op.constant_value = static_cast<LOGICAL_T>(val);
		op.max_value = op.constant_value;
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

	FOR_SWITCH_STORED(rst, RS, {
		auto rdata = reinterpret_cast<RS *>(payload);
		if (!lop.is_for) {
			FOR_SWITCH_STORED(rop.scan_data.stored_type, S, {
				unsafe_unique_array<data_t> buf;
				auto rhs = FORVector::CompactData<S>(rop.scan_data, count, buf);
				for (idx_t i = 0; i < count; i++) {
					rdata[i] = OP::template Operation<RS, S, RS>(UnsafeNumericCast<RS>(lop.constant_value), rhs[i]);
				}
				return true;
			});
		}
		if (!rop.is_for) {
			FOR_SWITCH_STORED(lop.scan_data.stored_type, S, {
				unsafe_unique_array<data_t> buf;
				auto lhs = FORVector::CompactData<S>(lop.scan_data, count, buf);
				for (idx_t i = 0; i < count; i++) {
					rdata[i] = OP::template Operation<S, RS, RS>(lhs[i], UnsafeNumericCast<RS>(rop.constant_value));
				}
				return true;
			});
		}
		FOR_SWITCH_STORED(lop.scan_data.stored_type, LS, {
			unsafe_unique_array<data_t> lbuf;
			auto ld = FORVector::CompactData<LS>(lop.scan_data, count, lbuf);
			FOR_SWITCH_STORED(rop.scan_data.stored_type, RRS, {
				unsafe_unique_array<data_t> rbuf;
				auto rd = FORVector::CompactData<RRS>(rop.scan_data, count, rbuf);
				for (idx_t i = 0; i < count; i++) {
					rdata[i] = OP::template Operation<LS, RRS, RS>(ld[i], rd[i]);
				}
				return true;
			});
		});
	});
	return true;
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
