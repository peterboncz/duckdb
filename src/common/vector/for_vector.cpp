#include "duckdb/common/vector/for_vector.hpp"

#include "duckdb/common/autovec.hpp"

namespace duckdb {

void ForVector::Create(Vector &vector, PhysicalType stored_type, uint64_t max_stored, idx_t count) {
	auto &buffer = vector.BufferMutable();
	D_ASSERT(buffer.GetData());
	D_ASSERT(buffer.cache_owned); // in-place widening needs the full-stride allocation
	// the size has to be right before the flag goes on: a widen reads it to know how much payload there is
	buffer.SetVectorSize(count);
	buffer.SetVectorTypeOnly(VectorType::FOR_VECTOR);
	buffer.for_stored_type = stored_type;
	buffer.for_max = max_stored;
	buffer.for_active = false; // spent on producing; an exploit site refills it
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

void ForVector::WidenInPlace(const LogicalType &type, VectorBuffer &buffer) {
	const auto count = buffer.Size();
	const auto stored = buffer.for_stored_type;
	auto data = buffer.GetData();
	switch (GetTypeIdSize(type.InternalType())) {
	case 2:
		WidenInPlaceTo<uint16_t>(data, stored, count);
		break;
	case 4:
		WidenInPlaceTo<uint32_t>(data, stored, count);
		break;
	default:
		WidenInPlaceTo<uint64_t>(data, stored, count);
		break;
	}
	buffer.SetVectorTypeOnly(VectorType::FLAT_VECTOR);
	buffer.for_stored_type = PhysicalType::INVALID;
}

template <class SRC, class DST>
DUCKDB_AUTOVEC_TARGET static void WidenLoop(const SRC *DUCKDB_BITPACKING_RESTRICT src,
                                            DST *DUCKDB_BITPACKING_RESTRICT target, idx_t count) {
	DUCKDB_UNROLL_LOOP
	for (idx_t i = 0; i < count; i++) {
		target[i] = static_cast<DST>(src[i]);
	}
}

template <class DST>
static void WidenToTarget(const_data_ptr_t src, PhysicalType stored, data_ptr_t target, idx_t count) {
	auto dst = reinterpret_cast<DST *>(target);
	switch (GetTypeIdSize(stored)) {
	case 1:
		return WidenLoop<uint8_t, DST>(reinterpret_cast<const uint8_t *>(src), dst, count);
	case 2:
		return WidenLoop<uint16_t, DST>(reinterpret_cast<const uint16_t *>(src), dst, count);
	default:
		return WidenLoop<uint32_t, DST>(reinterpret_cast<const uint32_t *>(src), dst, count);
	}
}

void ForVector::WidenPayload(const_data_ptr_t src, PhysicalType stored, data_ptr_t target, PhysicalType target_type,
                             idx_t count) {
	switch (GetTypeIdSize(target_type)) {
	case 2:
		return WidenToTarget<uint16_t>(src, stored, target, count);
	case 4:
		return WidenToTarget<uint32_t>(src, stored, target, count);
	default:
		return WidenToTarget<uint64_t>(src, stored, target, count);
	}
}

} // namespace duckdb
