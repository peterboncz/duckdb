//===----------------------------------------------------------------------===//
//                         DuckDB
//
// json_common.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/operator/decimal_cast_operators.hpp"
#include "duckdb/common/operator/string_cast.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "yyjson.hpp"

using namespace duckdb_yyjson; // NOLINT

namespace duckdb {

class JSONAllocator;

class JSONStringVectorBuffer : public VectorBuffer {
public:
	explicit JSONStringVectorBuffer(shared_ptr<JSONAllocator> allocator_p)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), allocator(std::move(allocator_p)) {
	}

private:
	shared_ptr<JSONAllocator> allocator;
};

//! JSON allocator is a custom allocator for yyjson that prevents many tiny allocations
class JSONAllocator : public enable_shared_from_this<JSONAllocator> {
public:
	explicit JSONAllocator(Allocator &allocator)
	    : arena_allocator(allocator), yyjson_allocator({Allocate, Reallocate, Free, this}) {
	}

	inline yyjson_alc *GetYYAlc() {
		return &yyjson_allocator;
	}

	void Reset() {
		arena_allocator.Reset();
	}

	void AddBuffer(Vector &vector) {
		if (vector.GetType().InternalType() == PhysicalType::VARCHAR) {
			StringVector::AddBuffer(vector, make_buffer<JSONStringVectorBuffer>(shared_from_this()));
		}
	}

	static void AddBuffer(Vector &vector, yyjson_alc *alc) {
		auto alloc = (JSONAllocator *)alc->ctx; // NOLINT
		alloc->AddBuffer(vector);
	}

private:
	static inline void *Allocate(void *ctx, size_t size) {
		auto alloc = (JSONAllocator *)ctx; // NOLINT
		return alloc->arena_allocator.AllocateAligned(size);
	}

	static inline void *Reallocate(void *ctx, void *ptr, size_t old_size, size_t size) {
		auto alloc = (JSONAllocator *)ctx; // NOLINT
		return alloc->arena_allocator.ReallocateAligned(data_ptr_cast(ptr), old_size, size);
	}

	static inline void Free(void *ctx, void *ptr) {
		// NOP because ArenaAllocator can't free
	}

private:
	ArenaAllocator arena_allocator;
	yyjson_alc yyjson_allocator;
};

//! JSONKey / json_key_map_t speeds up mapping from JSON key to column ID
struct JSONKey {
	const char *ptr;
	size_t len;
};

struct JSONKeyHash {
	inline std::size_t operator()(const JSONKey &k) const {
		size_t result;
		if (k.len >= sizeof(size_t)) {
			memcpy(&result, k.ptr + k.len - sizeof(size_t), sizeof(size_t));
		} else {
			result = 0;
			FastMemcpy(&result, k.ptr, k.len);
		}
		return result;
	}
};

struct JSONKeyEquality {
	inline bool operator()(const JSONKey &a, const JSONKey &b) const {
		if (a.len != b.len) {
			return false;
		}
		return FastMemcmp(a.ptr, b.ptr, a.len) == 0;
	}
};

template <typename T>
using json_key_map_t = unordered_map<JSONKey, T, JSONKeyHash, JSONKeyEquality>;
using json_key_set_t = unordered_set<JSONKey, JSONKeyHash, JSONKeyEquality>;

//! Common JSON functionality for most JSON functions
struct JSONCommon {
public:
	//! Read/Write flags
	static constexpr auto READ_FLAG =
	    YYJSON_READ_ALLOW_INF_AND_NAN | YYJSON_READ_ALLOW_TRAILING_COMMAS | YYJSON_READ_BIGNUM_AS_RAW;
	static constexpr auto READ_STOP_FLAG = READ_FLAG | YYJSON_READ_STOP_WHEN_DONE;
	static constexpr auto READ_INSITU_FLAG = READ_STOP_FLAG | YYJSON_READ_INSITU;
	static constexpr auto WRITE_FLAG = YYJSON_WRITE_ALLOW_INF_AND_NAN;
	static constexpr auto WRITE_PRETTY_FLAG = YYJSON_WRITE_ALLOW_INF_AND_NAN | YYJSON_WRITE_PRETTY;

public:
	//! Constant JSON type strings
	static constexpr char const *TYPE_STRING_NULL = "NULL";
	static constexpr char const *TYPE_STRING_BOOLEAN = "BOOLEAN";
	static constexpr char const *TYPE_STRING_BIGINT = "BIGINT";
	static constexpr char const *TYPE_STRING_UBIGINT = "UBIGINT";
	static constexpr char const *TYPE_STRING_DOUBLE = "DOUBLE";
	static constexpr char const *TYPE_STRING_HUGEINT = "HUGEINT";
	static constexpr char const *TYPE_STRING_VARCHAR = "VARCHAR";
	static constexpr char const *TYPE_STRING_ARRAY = "ARRAY";
	static constexpr char const *TYPE_STRING_OBJECT = "OBJECT";

	static inline const char *ValTypeToString(yyjson_val *val) {
		switch (yyjson_get_tag(val)) {
		case YYJSON_TYPE_NULL | YYJSON_SUBTYPE_NONE:
			return TYPE_STRING_NULL;
		case YYJSON_TYPE_STR | YYJSON_SUBTYPE_NOESC:
		case YYJSON_TYPE_STR | YYJSON_SUBTYPE_NONE:
			return TYPE_STRING_VARCHAR;
		case YYJSON_TYPE_ARR | YYJSON_SUBTYPE_NONE:
			return TYPE_STRING_ARRAY;
		case YYJSON_TYPE_OBJ | YYJSON_SUBTYPE_NONE:
			return TYPE_STRING_OBJECT;
		case YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_TRUE:
		case YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_FALSE:
			return TYPE_STRING_BOOLEAN;
		case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_UINT:
			return TYPE_STRING_UBIGINT;
		case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_SINT:
			return TYPE_STRING_BIGINT;
		case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL:
		case YYJSON_TYPE_RAW | YYJSON_SUBTYPE_NONE:
			return TYPE_STRING_DOUBLE;
		default:
			throw InternalException("Unexpected yyjson tag in ValTypeToString");
		}
	}

	static inline string_t ValTypeToStringT(yyjson_val *val) {
		return string_t(ValTypeToString(val));
	}

	static inline LogicalTypeId ValTypeToLogicalTypeId(yyjson_val *val) {
		switch (yyjson_get_tag(val)) {
		case YYJSON_TYPE_NULL | YYJSON_SUBTYPE_NONE:
			return LogicalTypeId::SQLNULL;
		case YYJSON_TYPE_STR | YYJSON_SUBTYPE_NOESC:
		case YYJSON_TYPE_STR | YYJSON_SUBTYPE_NONE:
			return LogicalTypeId::VARCHAR;
		case YYJSON_TYPE_ARR | YYJSON_SUBTYPE_NONE:
			return LogicalTypeId::LIST;
		case YYJSON_TYPE_OBJ | YYJSON_SUBTYPE_NONE:
			return LogicalTypeId::STRUCT;
		case YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_TRUE:
		case YYJSON_TYPE_BOOL | YYJSON_SUBTYPE_FALSE:
			return LogicalTypeId::BOOLEAN;
		case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_UINT:
			return LogicalTypeId::UBIGINT;
		case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_SINT:
			return LogicalTypeId::BIGINT;
		case YYJSON_TYPE_NUM | YYJSON_SUBTYPE_REAL:
		case YYJSON_TYPE_RAW | YYJSON_SUBTYPE_NONE:
			return LogicalTypeId::DOUBLE;
		default:
			throw InternalException("Unexpected yyjson tag in ValTypeToLogicalTypeId");
		}
	}

public:
	//===--------------------------------------------------------------------===//
	// Document creation / reading / writing
	//===--------------------------------------------------------------------===//
	template <class T>
	static T *AllocateArray(yyjson_alc *alc, idx_t count) {
		return reinterpret_cast<T *>(alc->malloc(alc->ctx, sizeof(T) * count));
	}

	template <class T>
	static T *AllocateArray(yyjson_mut_doc *doc, idx_t count) {
		return AllocateArray<T>(&doc->alc, count);
	}

	static inline yyjson_mut_doc *CreateDocument(yyjson_alc *alc) {
		D_ASSERT(alc);
		return yyjson_mut_doc_new(alc);
	}
	static inline yyjson_doc *ReadDocumentUnsafe(char *data, idx_t size, const yyjson_read_flag flg, yyjson_alc *alc,
	                                             yyjson_read_err *err = nullptr) {
		D_ASSERT(alc);
		return yyjson_read_opts(data, size, flg, alc, err);
	}
	static inline yyjson_doc *ReadDocumentUnsafe(const string_t &input, const yyjson_read_flag flg, yyjson_alc *alc,
	                                             yyjson_read_err *err = nullptr) {
		return ReadDocumentUnsafe(input.GetDataWriteable(), input.GetSize(), flg, alc, err);
	}
	static inline yyjson_doc *ReadDocument(char *data, idx_t size, const yyjson_read_flag flg, yyjson_alc *alc) {
		yyjson_read_err error;
		auto result = ReadDocumentUnsafe(data, size, flg, alc, &error);
		if (error.code != YYJSON_READ_SUCCESS) {
			ThrowParseError(data, size, error);
		}
		return result;
	}
	static inline yyjson_doc *ReadDocument(const string_t &input, const yyjson_read_flag flg, yyjson_alc *alc) {
		return ReadDocument(input.GetDataWriteable(), input.GetSize(), flg, alc);
	}

	static string FormatParseError(const char *data, idx_t length, yyjson_read_err &error, const string &extra = "") {
		D_ASSERT(error.code != YYJSON_READ_SUCCESS);
		// Go to blob so we can have a better error message for weird strings
		auto blob = Value::BLOB(string(data, length));
		// Truncate, so we don't print megabytes worth of JSON
		string input = blob.ToString();
		input = input.length() > 50 ? string(input.c_str(), 47) + "..." : input;
		// Have to replace \r, otherwise output is unreadable
		input = StringUtil::Replace(input, "\r", "\\r");
		return StringUtil::Format("Malformed JSON at byte %lld of input: %s. %s Input: \"%s\"", error.pos, error.msg,
		                          extra, input);
	}
	static void ThrowParseError(const char *data, idx_t length, yyjson_read_err &error, const string &extra = "") {
		throw InvalidInputException(FormatParseError(data, length, error, extra));
	}

	template <class YYJSON_VAL_T>
	static inline char *WriteVal(YYJSON_VAL_T *val, yyjson_alc *alc, idx_t &len) {
		throw InternalException("Unknown yyjson val type");
	}
	template <class YYJSON_VAL_T>
	static inline string_t WriteVal(YYJSON_VAL_T *val, yyjson_alc *alc) {
		D_ASSERT(alc);
		idx_t len;
		auto data = WriteVal<YYJSON_VAL_T>(val, alc, len);
		return string_t(data, len);
	}

	//! Slow and easy ToString for errors
	static string ValToString(yyjson_val *val, idx_t max_len = DConstants::INVALID_INDEX);
	//! Throw an error with the printed yyjson_val
	static void ThrowValFormatError(string error_string, yyjson_val *val);

public:
	//===--------------------------------------------------------------------===//
	// JSON pointer / path
	//===--------------------------------------------------------------------===//
	enum class JSONPathType : uint8_t {
		//! Extract a single value
		REGULAR = 0,
		//! Extract multiple values (when we have a '*' wildcard in the JSON Path)
		WILDCARD = 1,
	};

	//! Get JSON value using JSON path query (safe, checks the path query)
	static inline yyjson_val *Get(yyjson_val *val, const string_t &path_str, bool integral_argument) {
		auto ptr = path_str.GetData();
		auto len = path_str.GetSize();
		if (len == 0) {
			return GetUnsafe(val, ptr, len);
		}
		if (integral_argument) {
			auto str = "$[" + path_str.GetString() + "]";
			return GetUnsafe(val, str.c_str(), str.length());
		}
		switch (*ptr) {
		case '/': {
			// '/' notation must be '\0'-terminated
			auto str = string(ptr, len);
			return GetUnsafe(val, str.c_str(), len);
		}
		case '$': {
			if (ValidatePath(ptr, len, false) == JSONPathType::WILDCARD) {
				throw InvalidInputException(
				    "JSON path cannot contain wildcards if the path is not a constant parameter");
			}
			return GetUnsafe(val, ptr, len);
		}
		default: {
			string path;
			if (memchr(ptr, '"', len)) {
				path = "/" + string(ptr, len);
			} else {
				path = "$.\"" + path_str.GetString() + "\"";
			}
			return GetUnsafe(val, path.c_str(), path.length());
		}
		}
	}

	//! Get JSON value using JSON path query (unsafe)
	static inline yyjson_val *GetUnsafe(yyjson_val *val, const char *ptr, const idx_t &len) {
		if (len == 0) {
			return val;
		}
		switch (*ptr) {
		case '/':
			return GetPointer(val, ptr, len);
		case '$':
			return GetPath(val, ptr, len);
		default:
			throw InternalException("JSON pointer/path does not start with '/' or '$'");
		}
	}

	//! Get JSON value using JSON path query (unsafe)
	static void GetWildcardPath(yyjson_val *val, const char *ptr, const idx_t &len, vector<yyjson_val *> &vals);

	//! Validate JSON Path ($.field[index]... syntax), returns true if there are wildcards in the path
	static JSONPathType ValidatePath(const char *ptr, const idx_t &len, const bool binder);

public:
	//! Same as BigQuery json_value
	static inline string_t JSONValue(yyjson_val *val, yyjson_alc *alc, Vector &, ValidityMask &mask, idx_t idx) {
		switch (yyjson_get_tag(val)) {
		case YYJSON_TYPE_NULL | YYJSON_SUBTYPE_NONE:
		case YYJSON_TYPE_ARR | YYJSON_SUBTYPE_NONE:
		case YYJSON_TYPE_OBJ | YYJSON_SUBTYPE_NONE:
			mask.SetInvalid(idx);
			return string_t {};
		default:
			return JSONCommon::WriteVal<yyjson_val>(val, alc);
		}
	}

private:
	//! Get JSON pointer (/field/index/... syntax)
	static inline yyjson_val *GetPointer(yyjson_val *val, const char *ptr, const idx_t &len) {
		yyjson_ptr_err err;
		return unsafe_yyjson_ptr_getx(val, ptr, len, &err);
	}
	//! Get JSON path ($.field[index]... syntax)
	static yyjson_val *GetPath(yyjson_val *val, const char *ptr, const idx_t &len);
};

template <>
inline char *JSONCommon::WriteVal(yyjson_val *val, yyjson_alc *alc, idx_t &len) {
	return yyjson_val_write_opts(val, JSONCommon::WRITE_FLAG, alc, reinterpret_cast<size_t *>(&len), nullptr);
}
template <>
inline char *JSONCommon::WriteVal(yyjson_mut_val *val, yyjson_alc *alc, idx_t &len) {
	return yyjson_mut_val_write_opts(val, JSONCommon::WRITE_FLAG, alc, reinterpret_cast<size_t *>(&len), nullptr);
}

struct yyjson_doc_deleter {
	void operator()(yyjson_doc *doc) {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
};

struct yyjson_mut_doc_deleter {
	void operator()(yyjson_mut_doc *doc) {
		if (doc) {
			yyjson_mut_doc_free(doc);
		}
	}
};

using yyjson_doc_ptr = unique_ptr<yyjson_doc, yyjson_doc_deleter>;
using yyjson_mut_doc_ptr = unique_ptr<yyjson_mut_doc, yyjson_mut_doc_deleter>;

} // namespace duckdb
