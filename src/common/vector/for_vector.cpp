#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// FOR metadata storage via AuxiliaryData
//===--------------------------------------------------------------------===//
// Base class with stored_type (accessible without knowing T)
struct FORMetadataBase : public AuxiliaryDataHolder {
	PhysicalType stored_type;
	explicit FORMetadataBase(PhysicalType stored_type_p) : stored_type(stored_type_p) {
	}
};

// Typed subclass with the max value
template <class T>
struct FORMetadata : public FORMetadataBase {
	T max_value;
	FORMetadata(PhysicalType stored_type_p, T max_value_p) : FORMetadataBase(stored_type_p), max_value(max_value_p) {
	}
};

PhysicalType FORVector::GetStoredType(const Vector &vector) {
	D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
	auto &aux = vector.buffer->GetAuxiliaryData();
	D_ASSERT(aux);
	D_ASSERT(!aux->data.empty());
	return static_cast<FORMetadataBase &>(*aux->data[0]).stored_type;
}

template <class T>
T FORVector::GetMax(const Vector &vector) {
	D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
	auto &aux = vector.buffer->GetAuxiliaryData();
	D_ASSERT(aux);
	D_ASSERT(!aux->data.empty());
	using META_T = typename FORUnsignedType<T>::type;
	return UnsafeNumericCast<T>(static_cast<FORMetadata<META_T> &>(*aux->data[0]).max_value);
}

template <class T>
void FORVector::SetMetadata(Vector &vector, PhysicalType stored_type, T max_value) {
	D_ASSERT(vector.GetVectorType() == VectorType::FOR_VECTOR);
	using META_T = typename FORUnsignedType<T>::type;
	vector.buffer->ClearAuxiliaryData();
	vector.buffer->AddAuxiliaryData(make_uniq<FORMetadata<META_T>>(stored_type, UnsafeNumericCast<META_T>(max_value)));
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
	auto dst = FlatVector::GetData(target);
	FORVector::DispatchStoredType(FORVector::GetStoredType(source), [&](auto tag) {
		using S = typename decltype(tag)::type;
		auto stored = reinterpret_cast<const S *>(src);
		auto target_data = reinterpret_cast<LOGICAL_T *>(dst);
		for (idx_t i = 0; i < count; i++) {
			target_data[i] = FORVector::WidenStored<LOGICAL_T, S>(stored[sel ? sel->get_index(i) : i]);
		}
	});
}

void FORVector::Decompress(const Vector &source, Vector &target, idx_t count) {
	D_ASSERT(source.GetVectorType() == VectorType::FOR_VECTOR);
	DispatchLogicalType(source.GetType().InternalType(),
	                    [&](auto tag) { DecompressImpl<typename decltype(tag)::type>(source, target, count); });
}
void FORVector::Decompress(const Vector &source, Vector &target, const SelectionVector &sel, idx_t count) {
	D_ASSERT(source.GetVectorType() == VectorType::FOR_VECTOR);
	DispatchLogicalType(source.GetType().InternalType(),
	                    [&](auto tag) { DecompressImpl<typename decltype(tag)::type>(source, target, count, &sel); });
}

//===--------------------------------------------------------------------===//
// Create
//===--------------------------------------------------------------------===//
template <class MAX_T>
void FORVector::Create(Vector &vector, PhysicalType stored_type, MAX_T max_value) {
	D_ASSERT(vector.buffer);
	D_ASSERT(vector.buffer->GetData());
	vector.vector_type = VectorType::FOR_VECTOR;
	vector.validity.Reset();
	FORVector::SetMetadata<MAX_T>(vector, stored_type, max_value);
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
	uint64_t max_raw = DispatchLogicalType(source.GetType().InternalType(), [&](auto tag) {
		return UnsafeNumericCast<uint64_t>(GetMax<typename decltype(tag)::type>(source));
	});
	DispatchLogicalType(result.GetType().InternalType(), [&](auto tag) {
		using T = typename decltype(tag)::type;
		SetMetadata<T>(result, st, WidenStored<T, uint64_t>(max_raw));
	});
	return true;
}

} // namespace duckdb
