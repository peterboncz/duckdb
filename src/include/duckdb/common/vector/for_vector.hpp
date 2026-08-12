//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/vector/for_vector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/vector.hpp"

namespace duckdb {

//! A FOR vector holds absolute values in a narrower physical type than its logical type: the frame of reference is
//! folded into the payload by the producer, so widening is a pure cast and no consumer reasons about a frame.
//! The state lives on the vector's own cache-owned buffer, so the chunk reset un-FORs it and Flatten widens in place.
struct ForVector {
	static bool IsFor(const Vector &vector) {
		return vector.GetVectorType() == VectorType::FOR_VECTOR;
	}
	//! Mark a flat vector as FOR: its buffer already holds the narrow payload for the first count rows
	static void Create(Vector &vector, PhysicalType stored_type, uint64_t max_stored, idx_t count);
	//! A producer about to overwrite the whole payload can drop the flag instead of widening it. Only safe when
	//! the writer covers every published row: rows past the new size keep narrow bytes, which is fine because
	//! nothing reads past the vector size, but a partial overwrite would leave a narrow hole inside it.
	static void Discard(Vector &vector, idx_t rows_to_write) {
		if (!IsFor(vector)) {
			return;
		}
		auto &buffer = vector.BufferMutable();
		if (rows_to_write < buffer.for_count) {
			Widen(vector);
			return;
		}
		buffer.SetVectorTypeOnly(VectorType::FLAT_VECTOR);
		buffer.for_stored_type = PhysicalType::INVALID;
	}
	//! Safety net for FOR-unaware code: widen back to the logical width. Free - the payload is widened in place.
	static void Widen(const Vector &vector) {
		if (IsFor(vector)) {
			vector.Flatten();
		}
	}
	static PhysicalType StoredType(const Vector &vector) {
		return vector.GetBufferRef()->for_stored_type;
	}
	//! Largest value the payload holds, so a consumer can prove a computation cannot overflow
	static uint64_t MaxStored(const Vector &vector) {
		return vector.GetBufferRef()->for_max;
	}
	//! Keepalive: producers only emit FOR while the token is set, and spend it on producing.
	//! A consumer that exploited the narrow payload refills it.
	static bool TokenSet(const Vector &vector) {
		return vector.GetBufferRef()->for_active;
	}
	static void MarkExploited(const Vector &vector) {
		vector.GetBufferRef()->for_active = true;
	}
	//! Hand a narrow payload straight to an integer downcast of the same width, with no copy. False when the cast
	//! is not a pure reinterpretation of the payload, so the caller must run the real cast.
	static bool TryRetype(Vector &source, Vector &result, idx_t count);
	//! Rewrite a comparison constant into the stored space. False when it falls outside the payload's range:
	//! the comparison is then uniform, which the normal path handles just as well.
	static bool TryStoredConstant(const Vector &vector, const Value &constant, uint64_t &result);
	//! Widen the payload within its own allocation and turn the vector back into a flat one
	static void WidenInPlace(const LogicalType &type, VectorBuffer &buffer);
	//! Gather-widen only the selected values into a target of the logical width. For a selective filter this
	//! touches the survivors instead of the whole vector, which is the whole point of keeping the payload narrow.
	static void WidenGather(const_data_ptr_t src, PhysicalType stored, data_ptr_t target, PhysicalType target_type,
	                        const SelectionVector &sel, idx_t count);
	//! Widen a narrow payload where it lies. Always in place: a producer abandoning FOR part-way through a vector
	//! has its source and target in the same bytes, so this must never be a forward copy.
	static void WidenInPlace(data_ptr_t data, PhysicalType stored, PhysicalType target_type, idx_t count);
};

} // namespace duckdb
