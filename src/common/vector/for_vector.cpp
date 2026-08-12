#include "duckdb/common/vector/for_vector.hpp"

#include "duckdb/common/autovec.hpp"

namespace duckdb {

void ForVector::Create(Vector &vector, PhysicalType stored_type, uint64_t max_stored, idx_t count) {
	auto &buffer = vector.BufferMutable();
	D_ASSERT(buffer.GetData());
	D_ASSERT(buffer.cache_owned); // in-place widening needs the full-stride allocation
	buffer.for_count = count;
	buffer.SetVectorTypeOnly(VectorType::FOR_VECTOR);
	buffer.for_stored_type = stored_type;
	buffer.for_max = max_stored;
	buffer.for_active = false; // spent on producing; an exploit site refills it
}

bool ForVector::TryStoredConstant(const Vector &vector, const Value &constant, uint64_t &result) {
	int64_t value;
	switch (vector.GetType().InternalType()) {
	case PhysicalType::INT16:
		value = constant.GetValueUnsafe<int16_t>();
		break;
	case PhysicalType::INT32:
		value = constant.GetValueUnsafe<int32_t>();
		break;
	case PhysicalType::INT64:
		value = constant.GetValueUnsafe<int64_t>();
		break;
	case PhysicalType::UINT16:
		value = constant.GetValueUnsafe<uint16_t>();
		break;
	case PhysicalType::UINT32:
		value = constant.GetValueUnsafe<uint32_t>();
		break;
	case PhysicalType::UINT64: {
		const auto unsigned_value = constant.GetValueUnsafe<uint64_t>();
		if (unsigned_value > MaxStored(vector)) {
			return false;
		}
		result = unsigned_value;
		return true;
	}
	default:
		return false;
	}
	// the payload holds absolute values in [0, for_max], so anything outside that compares uniformly
	if (value < 0 || static_cast<uint64_t>(value) > MaxStored(vector)) {
		return false;
	}
	result = static_cast<uint64_t>(value);
	return true;
}

//===--------------------------------------------------------------------===//
// Widen - the only execution loop FOR owns
//===--------------------------------------------------------------------===//
//! Back-to-front so a wider destination can overwrite its own narrower source
template <class SRC, class DST>
DUCKDB_AUTOVEC_TARGET static void WidenInPlaceLoop(data_ptr_t data, idx_t count) {
	auto src = reinterpret_cast<const SRC *>(data);
	auto dst = reinterpret_cast<DST *>(data);
	for (idx_t i = count; i-- > 0;) {
		dst[i] = static_cast<DST>(src[i]);
	}
}

template <class DST>
static void WidenInPlaceTo(data_ptr_t data, PhysicalType stored, idx_t count) {
	switch (GetTypeIdSize(stored)) {
	case 1:
		return WidenInPlaceLoop<uint8_t, DST>(data, count);
	case 2:
		return WidenInPlaceLoop<uint16_t, DST>(data, count);
	default:
		return WidenInPlaceLoop<uint32_t, DST>(data, count);
	}
}

void ForVector::WidenInPlace(data_ptr_t data, PhysicalType stored, PhysicalType target_type, idx_t count) {
	switch (GetTypeIdSize(target_type)) {
	case 2:
		return WidenInPlaceTo<uint16_t>(data, stored, count);
	case 4:
		return WidenInPlaceTo<uint32_t>(data, stored, count);
	default:
		return WidenInPlaceTo<uint64_t>(data, stored, count);
	}
}

void ForVector::WidenInPlace(const LogicalType &type, VectorBuffer &buffer) {
	WidenInPlace(buffer.GetData(), buffer.for_stored_type, type.InternalType(), buffer.for_count);
	buffer.SetVectorTypeOnly(VectorType::FLAT_VECTOR);
	buffer.for_stored_type = PhysicalType::INVALID;
}

} // namespace duckdb
