//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/scan_filter_bitmap.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/validity_mask.hpp"

#include <cstring>

namespace duckdb {

static constexpr idx_t TABLE_SCAN_FILTER_BITMAP_WORDS = (STANDARD_VECTOR_SIZE + 63) / 64;

static inline void ScanFilterBitmapSetAll(validity_t *__restrict bitmap, idx_t count) {
	memset(bitmap, 0, TABLE_SCAN_FILTER_BITMAP_WORDS * sizeof(validity_t));
	const auto word_count = (count + 63) / 64;
	for (idx_t i = 0; i < word_count; i++) {
		bitmap[i] = ~validity_t(0);
	}
	if (count & 63) {
		bitmap[word_count - 1] &= (validity_t(1) << (count & 63)) - 1;
	}
}

static inline void ScanFilterBitmapFromSelection(validity_t *__restrict bitmap, const SelectionVector &sel,
                                                 idx_t count) {
	memset(bitmap, 0, TABLE_SCAN_FILTER_BITMAP_WORDS * sizeof(validity_t));
	for (idx_t i = 0; i < count; i++) {
		const auto idx = sel.get_index(i);
		bitmap[idx / 64] |= validity_t(1) << (idx % 64);
	}
}

static inline void ScanFilterBitmapAnd(validity_t *__restrict dst, const validity_t *__restrict src, idx_t count) {
	const auto word_count = (count + 63) / 64;
	for (idx_t i = 0; i < word_count; i++) {
		dst[i] &= src[i];
	}
	if (count & 63) {
		dst[word_count - 1] &= (validity_t(1) << (count & 63)) - 1;
	}
}

static inline idx_t ScanFilterBitmapCount(const validity_t *__restrict bitmap, idx_t count) {
	const auto word_count = (count + 63) / 64;
	idx_t result = 0;
	for (idx_t i = 0; i < word_count; i++) {
		auto word = bitmap[i];
		if (i + 1 == word_count && (count & 63)) {
			word &= (validity_t(1) << (count & 63)) - 1;
		}
		result += UnsafeNumericCast<idx_t>(__builtin_popcountll(word));
	}
	return result;
}

} // namespace duckdb
