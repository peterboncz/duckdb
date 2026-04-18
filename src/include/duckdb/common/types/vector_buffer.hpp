//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/vector_buffer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"

namespace duckdb {

class BufferHandle;
class VectorBuffer;
class Vector;
struct ValidityMask;

enum class VectorBufferType : uint8_t {
	STANDARD_BUFFER,   // VectorType::FLAT/CONSTANT - Fixed-Size Type - Holds a single array of data
	STRING_BUFFER,     // VectorType::FLAT/CONSTANT - String          - Holds string_t array and StringHeap
	STRUCT_BUFFER,     // VectorType::FLAT/CONSTANT - Struct          - Holds struct child vectors
	LIST_BUFFER,       // VectorType::FLAT/CONSTANT - List            - Holds list_entry_t array and list child vector
	ARRAY_BUFFER,      // VectorType::FLAT/CONSTANT - Array           - Holds array child vector
	DICTIONARY_BUFFER, // VectorType::DICTIONARY    - Any             - Holds SelectionVector and dict child vector
	FSST_BUFFER,       // VectorType::FSST          - String          - Holds string_t array, StringHeap and FSST table
	SHREDDED_BUFFER,   // VectorType::SHREDDED      - Variant         - Holds shredded variant
	SEQUENCE_BUFFER    // VectorType::SEQUENCE      - Any             - Holds linear numeric sequence (start, increment)
};

struct AuxiliaryDataHolder {
	virtual ~AuxiliaryDataHolder() = default;
};

struct AuxiliaryDataSet {
	vector<unique_ptr<AuxiliaryDataHolder>> data;
};

class PinnedBufferHolder : public AuxiliaryDataHolder {
public:
	explicit PinnedBufferHolder(BufferHandle handle);
	~PinnedBufferHolder() override;

private:
	BufferHandle handle;
};

class AuxiliaryDataSetHolder : public AuxiliaryDataHolder {
public:
	explicit AuxiliaryDataSetHolder(buffer_ptr<AuxiliaryDataSet> buffer) : auxiliary_data(std::move(buffer)) {
	}

private:
	buffer_ptr<AuxiliaryDataSet> auxiliary_data;
};

//! The VectorBuffer is a class used by the vector to hold its data
class VectorBuffer {
public:
	explicit VectorBuffer(VectorBufferType type) : buffer_type(type) {
	}
	virtual ~VectorBuffer() {
	}

public:
	virtual data_ptr_t GetData() {
		return nullptr;
	}
	virtual ValidityMask &GetValidityMask() {
		throw InternalException("VectorBuffer does not have a ValidityMask");
	}

	void AddAuxiliaryData(unique_ptr<AuxiliaryDataHolder> aux_data_p) {
		if (!auxiliary_data) {
			auxiliary_data = make_buffer<AuxiliaryDataSet>();
		}
		auxiliary_data->data.push_back(std::move(aux_data_p));
	}
	buffer_ptr<AuxiliaryDataSet> &GetAuxiliaryData() {
		return auxiliary_data;
	}
	virtual void ClearAuxiliaryData() {
		auxiliary_data.reset();
	}

	virtual optional_ptr<Allocator> GetAllocator() const {
		return nullptr;
	}

	//! Inline FOR vector metadata — avoids per-chunk heap allocation.
	//! Only meaningful when the vector has VectorType::FOR_VECTOR.
	//! Logical value is stored directly in the payload, and max tracks the largest payload value.
	PhysicalType for_stored_type = PhysicalType::INVALID;
	uhugeint_t for_max_value = 0;
	//! True when this buffer is managed by a VectorCache. Allows Flatten
	//! to widen FOR data in-place at use_count==2 (one ref from cache,
	//! one from vector) without allocating a new buffer.
	bool cache_owned = false;
	//! Cached flatten buffer — reused across iterations to avoid per-chunk
	//! malloc/free in the Flatten-before-Slice path for FOR vectors.
	buffer_ptr<VectorBuffer> flatten_cache;

	static buffer_ptr<VectorBuffer> CreateStandardVector(PhysicalType type, idx_t capacity = STANDARD_VECTOR_SIZE);
	static buffer_ptr<VectorBuffer> CreateConstantVector(PhysicalType type);
	static buffer_ptr<VectorBuffer> CreateConstantVector(const LogicalType &logical_type);
	static buffer_ptr<VectorBuffer> CreateStandardVector(const LogicalType &logical_type,
	                                                     idx_t capacity = STANDARD_VECTOR_SIZE);

	inline VectorBufferType GetBufferType() const {
		return buffer_type;
	}

protected:
	VectorBufferType buffer_type;
	buffer_ptr<AuxiliaryDataSet> auxiliary_data;

public:
	template <class TARGET>
	TARGET &Cast() {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<TARGET &>(*this);
	}
	template <class TARGET>
	const TARGET &Cast() const {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<const TARGET &>(*this);
	}
};

} // namespace duckdb
