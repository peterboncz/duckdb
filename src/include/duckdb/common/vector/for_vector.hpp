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
	//! Drop the FOR flag without widening, for a producer that is about to overwrite the whole payload.
	//! Does not refill the token: only a consumer that exploited the narrow payload earns another FOR vector.
	static void Reset(Vector &vector) {
		if (IsFor(vector)) {
			auto &buffer = vector.BufferMutable();
			buffer.SetVectorTypeOnly(VectorType::FLAT_VECTOR);
			buffer.for_stored_type = PhysicalType::INVALID;
		}
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
	//! Widen the payload within its own allocation and turn the vector back into a flat one
	static void WidenInPlace(const LogicalType &type, VectorBuffer &buffer);
	//! Widen a narrow payload into a separate target, for producers that abandon FOR part-way through a vector
	static void WidenPayload(const_data_ptr_t src, PhysicalType stored, data_ptr_t target, PhysicalType target_type,
	                         idx_t count);
};

} // namespace duckdb
