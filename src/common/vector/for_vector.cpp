#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// FOR metadata — stored inline on VectorBuffer (zero allocation)
//===--------------------------------------------------------------------===//
PhysicalType FORVector::GetStoredType(const Vector &vector) {
	D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
	return vector.buffer->for_stored_type;
}

uint8_t FORVector::GetRangeBits(const Vector &vector) {
	auto max_value = vector.buffer->for_max_value;
	uint8_t result = 0;
	while (max_value != 0) {
		result++;
		max_value >>= 1;
	}
	return result;
}

template <class T>
T FORVector::GetMax(const Vector &vector) {
	D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
	return FORValueOps<T>::FromUnsignedStorage(vector.buffer->for_max_value);
}

template <class T>
void FORVector::SetMetadata(Vector &vector, PhysicalType stored_type, T max_value) {
	D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
	vector.buffer->for_stored_type = stored_type;
	vector.buffer->for_max_value = FORValueOps<T>::ToUnsignedStorage(max_value);
}

// Explicit instantiations
#define FOR_INSTANTIATE(T)                                                                                             \
	template T FORVector::GetMax<T>(const Vector &);                                                                   \
	template void FORVector::SetMetadata<T>(Vector &, PhysicalType, T);                                                \
	template void FORVector::Create<T>(Vector &, PhysicalType, T);
FOR_INSTANTIATE(int16_t)
FOR_INSTANTIATE(int32_t)
FOR_INSTANTIATE(int64_t)
FOR_INSTANTIATE(hugeint_t)
FOR_INSTANTIATE(uint16_t)
FOR_INSTANTIATE(uint32_t)
FOR_INSTANTIATE(uint64_t)
FOR_INSTANTIATE(uhugeint_t)
#undef FOR_INSTANTIATE

//===--------------------------------------------------------------------===//
// Shared helpers
//===--------------------------------------------------------------------===//
LogicalType FORVector::StoredTypeToLogical(PhysicalType stored_type) {
	switch (stored_type) {
	case PhysicalType::UINT8:
		return LogicalType::UTINYINT;
	case PhysicalType::UINT16:
		return LogicalType::USMALLINT;
	case PhysicalType::UINT32:
		return LogicalType::UINTEGER;
	case PhysicalType::UINT64:
		return LogicalType::UBIGINT;
	default:
		throw InternalException("Unsupported stored type for FOR vector");
	}
}

Vector FORVector::CreateStoredView(const Vector &for_vec) {
	D_ASSERT(for_vec.GetVectorType() == VectorType::FOR_VECTOR);
	auto stored_data = const_cast<data_ptr_t>(FORVector::GetData(for_vec));
	Vector stored_vec(StoredTypeToLogical(FORVector::GetStoredType(for_vec)), stored_data);
	stored_vec.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::Validity(stored_vec) = FORVector::Validity(for_vec);
	return stored_vec;
}

//===--------------------------------------------------------------------===//
// Decompress
//===--------------------------------------------------------------------===//
template <class LOGICAL_T>
static void DecompressImpl(const Vector &source, Vector &target, idx_t count, const SelectionVector *sel = nullptr) {
	auto src = FORVector::GetData(source);
	auto dst = FlatVector::GetDataMutable(target);
	FOR_SWITCH_STORED(FORVector::GetStoredType(source), S, {
		auto stored = reinterpret_cast<const S *>(src);
		auto target_data = reinterpret_cast<LOGICAL_T *>(dst);
		for (idx_t i = 0; i < count; i++) {
			target_data[i] = FORVector::WidenStored<LOGICAL_T, S>(stored[sel ? sel->get_index(i) : i]);
		}
	});
}

void FORVector::Decompress(const Vector &source, Vector &target, idx_t count) {
	D_ASSERT(source.GetVectorType() == VectorType::FOR_VECTOR);
	FOR_SWITCH_LOGICAL(source.GetType().InternalType(), T, { DecompressImpl<T>(source, target, count); });
}
void FORVector::Decompress(const Vector &source, Vector &target, const SelectionVector &sel, idx_t count) {
	D_ASSERT(source.GetVectorType() == VectorType::FOR_VECTOR);
	FOR_SWITCH_LOGICAL(source.GetType().InternalType(), T, { DecompressImpl<T>(source, target, count, &sel); });
}

void FORVector::CopyToFlat(const Vector &source, const SelectionVector &sel, Vector &target,
                           idx_t source_offset, idx_t target_offset, idx_t copy_count) {
	D_ASSERT(source.GetVectorType() == VectorType::FOR_VECTOR);
	D_ASSERT(target.GetVectorType() == VectorType::FLAT_VECTOR);
	auto st = GetStoredType(source);
	auto *src = GetData(source);
	FOR_SWITCH_LOGICAL(source.GetType().InternalType(), LT, {
		auto *dst = FlatVector::GetDataMutable<LT>(target);
		FOR_SWITCH_STORED(st, ST, {
			auto *s = reinterpret_cast<const ST *>(src);
			for (idx_t i = 0; i < copy_count; i++) {
				dst[target_offset + i] = WidenStored<LT, ST>(s[sel.get_index(source_offset + i)]);
			}
		});
	});
}

//===--------------------------------------------------------------------===//
// Create
//===--------------------------------------------------------------------===//
template <class T>
void FORVector::Create(Vector &vector, PhysicalType stored_type, T max_value) {
	D_ASSERT(vector.buffer);
	D_ASSERT(vector.buffer->GetData());
	vector.vector_type = VectorType::FOR_VECTOR;
	vector.buffer->GetValidityMask().Reset();
	FORVector::SetMetadata<T>(vector, stored_type, max_value);
}

bool FORVector::TryWidenType(Vector &source, Vector &result) {
	if (source.GetVectorType() != VectorType::FOR_VECTOR) {
		return false;
	}
	if (GetTypeIdSize(result.GetType().InternalType()) < GetTypeIdSize(source.GetType().InternalType())) {
		return false;
	}
	auto st = GetStoredType(source);
	result.CopyBuffer(source);
	result.SetVectorType(VectorType::FOR_VECTOR);
	Validity(result) = Validity(source);
	uint64_t max_raw = UnsafeNumericCast<uint64_t>(source.buffer->for_max_value);
	FOR_SWITCH_LOGICAL(result.GetType().InternalType(), T, { SetMetadata<T>(result, st, WidenStored<T, uint64_t>(max_raw)); });
	return true;
}

} // namespace duckdb
