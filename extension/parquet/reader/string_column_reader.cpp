#include "reader/string_column_reader.hpp"
#include "utf8proc_wrapper.hpp"
#include "parquet_reader.hpp"
#include "duckdb/common/types/blob.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// String Column Reader
//===--------------------------------------------------------------------===//
StringColumnReader::StringColumnReader(ParquetReader &reader, LogicalType type_p, const SchemaElement &schema_p,
                                       idx_t schema_idx_p, idx_t max_define_p, idx_t max_repeat_p)
    : ColumnReader(reader, std::move(type_p), schema_p, schema_idx_p, max_define_p, max_repeat_p) {
	fixed_width_string_length = 0;
	if (schema_p.type == Type::FIXED_LEN_BYTE_ARRAY) {
		D_ASSERT(schema_p.__isset.type_length);
		fixed_width_string_length = schema_p.type_length;
	}
}

void StringColumnReader::VerifyString(const char *str_data, uint32_t str_len, const bool is_varchar) {
	if (!is_varchar) {
		return;
	}
	// verify if a string is actually UTF8, and if there are no null bytes in the middle of the string
	// technically Parquet should guarantee this, but reality is often disappointing
	UnicodeInvalidReason reason;
	size_t pos;
	auto utf_type = Utf8Proc::Analyze(str_data, str_len, &reason, &pos);
	if (utf_type == UnicodeType::INVALID) {
		throw InvalidInputException("Invalid string encoding found in Parquet file: value \"" +
		                            Blob::ToString(string_t(str_data, str_len)) + "\" is not valid UTF8!");
	}
}

void StringColumnReader::VerifyString(const char *str_data, uint32_t str_len) {
	VerifyString(str_data, str_len, Type().id() == LogicalTypeId::VARCHAR);
}

class ParquetStringVectorBuffer : public VectorBuffer {
public:
	explicit ParquetStringVectorBuffer(shared_ptr<ResizeableBuffer> buffer_p)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), buffer(std::move(buffer_p)) {
	}

private:
	shared_ptr<ResizeableBuffer> buffer;
};

void StringColumnReader::Plain(shared_ptr<ResizeableBuffer> &plain_data, uint8_t *defines, idx_t num_values,
                               idx_t result_offset, Vector &result) {
	StringVector::AddBuffer(result, make_buffer<ParquetStringVectorBuffer>(plain_data));
	PlainTemplated<string_t, StringParquetValueConversion>(*plain_data, defines, num_values, result_offset, result);
}

void StringColumnReader::PlainSkip(ByteBuffer &plain_data, uint8_t *defines, idx_t num_values) {
	PlainSkipTemplated<StringParquetValueConversion>(plain_data, defines, num_values);
}

} // namespace duckdb
