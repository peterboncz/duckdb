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
}

bool ForVector::TryRetype(Vector &source, Vector &result, idx_t count) {
	if (source.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		// a sliced input retypes its child and keeps the same selection
		auto &child = DictionaryVector::Child(source);
		Vector retyped(result.GetType(), buffer_ptr<VectorBuffer>());
		if (!TryRetype(child, retyped, child.size())) {
			return false;
		}
		result.Dictionary(make_buffer<DictionaryEntry>(std::move(retyped)), DictionaryVector::SelVector(source), count);
		return true;
	}
	// only a plain integer narrowing reinterprets the payload as-is; decimal rescaling, dates and enums do not
	if (!IsFor(source) || !source.GetType().IsIntegral() || !result.GetType().IsIntegral()) {
		return false;
	}
	const auto target_size = GetTypeIdSize(result.GetType().InternalType());
	if (GetTypeIdSize(StoredType(source)) != target_size || target_size > 4) {
		return false; // only an exact-width handoff avoids the copy
	}
	// every value is in [0, for_max], so the cast cannot fail once the target holds for_max
	const uint64_t target_max = result.GetType().IsSigned() ? (uint64_t(1) << (target_size * 8 - 1)) - 1
	                                                        : (uint64_t(1) << (target_size * 8)) - 1;
	if (MaxStored(source) > target_max) {
		return false;
	}
	auto buffer = make_buffer<StandardVectorBuffer>(source.BufferMutable().GetData(), count_t(count), target_size);
	buffer->GetValidityMask().Initialize(source.Buffer().GetValidityMask());
	buffer->AddAuxiliaryData(make_uniq<VectorBufferHolder>(source.GetBufferRef()));
	// the view aliases the payload, so the source must lose its in-place widen permission: a later flatten has to
	// copy out rather than rewrite the bytes this view is showing
	source.BufferMutable().cache_owned = false;
	MarkExploited(source); // the retype used the narrow payload: keep the scan producing FOR for this column
	result.SetBuffer(std::move(buffer));
	return true;
}

//! A constant above the payload's ceiling makes every comparison uniform, and replacing it with for_max + 1 gives
//! that same answer through the ordinary kernel: no value can equal or exceed it, and every value is below it.
//! So the comparison stays a narrow one instead of widening the payload to reach a foregone conclusion.
static bool ClampStoredConstant(const Vector &vector, uint64_t &value) {
	const auto max_stored = ForVector::MaxStored(vector);
	if (value <= max_stored) {
		return true;
	}
	const auto stored_size = GetTypeIdSize(ForVector::StoredType(vector));
	const uint64_t stored_max = stored_size == 8 ? NumericLimits<uint64_t>::Maximum()
	                                            : ((uint64_t(1) << (stored_size * 8)) - 1);
	if (max_stored >= stored_max) {
		return false; // no headroom for the sentinel
	}
	value = max_stored + 1;
	return true;
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
	case PhysicalType::UINT64:
		result = constant.GetValueUnsafe<uint64_t>();
		return ClampStoredConstant(vector, result);
	default:
		return false;
	}
	if (value < 0) {
		return false; // no unsigned stored constant expresses "below every value"
	}
	result = static_cast<uint64_t>(value);
	return ClampStoredConstant(vector, result);
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

template <class SRC, class DST>
DUCKDB_AUTOVEC_TARGET static void WidenGatherLoop(const_data_ptr_t src_p, data_ptr_t target, const SelectionVector &sel,
                                                  idx_t count) {
	auto src = reinterpret_cast<const SRC *>(src_p);
	auto dst = reinterpret_cast<DST *>(target);
	for (idx_t i = 0; i < count; i++) {
		dst[i] = static_cast<DST>(src[sel.get_index(i)]);
	}
}

template <class DST>
static void WidenGatherTo(const_data_ptr_t src, PhysicalType stored, data_ptr_t target, const SelectionVector &sel,
                          idx_t count) {
	switch (GetTypeIdSize(stored)) {
	case 1:
		return WidenGatherLoop<uint8_t, DST>(src, target, sel, count);
	case 2:
		return WidenGatherLoop<uint16_t, DST>(src, target, sel, count);
	default:
		return WidenGatherLoop<uint32_t, DST>(src, target, sel, count);
	}
}

void ForVector::WidenGather(const_data_ptr_t src, PhysicalType stored, data_ptr_t target, PhysicalType target_type,
                            const SelectionVector &sel, idx_t count) {
	switch (GetTypeIdSize(target_type)) {
	case 2:
		return WidenGatherTo<uint16_t>(src, stored, target, sel, count);
	case 4:
		return WidenGatherTo<uint32_t>(src, stored, target, sel, count);
	default:
		return WidenGatherTo<uint64_t>(src, stored, target, sel, count);
	}
}

void ForVector::WidenInPlace(const LogicalType &type, VectorBuffer &buffer) {
	WidenInPlace(buffer.GetData(), buffer.for_stored_type, type.InternalType(), buffer.for_count);
	// nothing used the narrow payload: sit out a cooldown before producing it again
	buffer.for_cooldown = COOLDOWN;
	buffer.SetVectorTypeOnly(VectorType::FLAT_VECTOR);
	buffer.for_stored_type = PhysicalType::INVALID;
}

} // namespace duckdb
