//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/core_functions/aggregate/sum_helpers.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/vector/for_vector.hpp"
#include "duckdb/function/aggregate_state.hpp"
#include "duckdb/common/operator/cast_operators.hpp"

namespace duckdb {

static inline void KahanAddInternal(double input, double &summed, double &err) {
	double diff = input - err;
	double newval = summed + diff;
	err = (newval - summed) - diff;
	summed = newval;
}

template <class T>
struct SumState {
	bool isset;
	T value;

	void Initialize() {
		this->isset = false;
		this->value = 0;
	}

	void Combine(const SumState<T> &other) {
		this->isset = other.isset || this->isset;
		this->value += other.value;
	}
};

struct KahanSumState {
	bool isset;
	double value;
	double err;

	void Initialize() {
		this->isset = false;
		this->err = 0.0;
	}

	void Combine(const KahanSumState &other) {
		this->isset = other.isset || this->isset;
		KahanAddInternal(other.value, this->value, this->err);
		KahanAddInternal(other.err, this->value, this->err);
	}
};

struct RegularAdd {
	template <class STATE, class T>
	static void AddNumber(STATE &state, T input) {
		state.value += input;
	}

	template <class STATE, class T>
	static void AddConstant(STATE &state, T input, idx_t count) {
		state.value += input * int64_t(count);
	}
};

struct HugeintAdd {
	template <class STATE, class T>
	static void AddNumber(STATE &state, T input) {
		state.value = Hugeint::Add(state.value, input);
	}

	template <class STATE, class T>
	static void AddConstant(STATE &state, T input, idx_t count) {
		AddNumber(state, Hugeint::Multiply(input, UnsafeNumericCast<int64_t>(count)));
	}
};

struct IntervalAdd {
	template <class STATE, class T>
	static void AddNumber(STATE &state, T input) {
		state.value = AddOperator::Operation<interval_t, interval_t, interval_t>(state.value, input);
	}

	template <class STATE, class T>
	static void AddConstant(STATE &state, T input, idx_t count) {
		const auto count64 = Cast::Operation<idx_t, int64_t>(count);
		input = MultiplyOperator::Operation<interval_t, int64_t, interval_t>(input, count64);
		state.value = AddOperator::Operation<interval_t, interval_t, interval_t>(state.value, input);
	}
};

struct KahanAdd {
	template <class STATE, class T>
	static void AddNumber(STATE &state, T input) {
		KahanAddInternal(input, state.value, state.err);
	}

	template <class STATE, class T>
	static void AddConstant(STATE &state, T input, idx_t count) {
		KahanAddInternal(input * count, state.value, state.err);
	}
};

struct AddToHugeint {
	static void AddValue(hugeint_t &result, uint64_t value, int positive) {
		// integer summation taken from Tim Gubner et al. - Efficient Query Processing
		// with Optimistically Compressed Hash Tables & Strings in the USSR

		// add the value to the lower part of the hugeint
		result.lower += value;
		// now handle overflows
		int overflow = result.lower < value;
		// we consider two situations:
		// (1) input[idx] is positive, and current value is lower than value: overflow
		// (2) input[idx] is negative, and current value is higher than value: underflow
		if (!(overflow ^ positive)) {
			// in the case of an overflow or underflow we either increment or decrement the upper base
			// positive: +1, negative: -1
			result.upper += -1 + 2 * positive;
		}
	}

	template <class STATE, class T>
	static void AddNumber(STATE &state, T input) {
		AddValue(state.value, uint64_t(input), input >= 0);
	}

	template <class STATE, class T>
	static void AddConstant(STATE &state, T input, idx_t count) {
		// add a constant X number of times
		// fast path: check if value * count fits into a uint64_t
		// note that we check if value * VECTOR_SIZE fits in a uint64_t to avoid having to actually do a division
		// this is still a pretty high number (18014398509481984) so most positive numbers will fit
		if (input >= 0 && uint64_t(input) < (NumericLimits<uint64_t>::Maximum() / STANDARD_VECTOR_SIZE)) {
			// if it does just multiply it and add the value
			uint64_t value = uint64_t(input) * count;
			AddValue(state.value, value, 1);
		} else {
			// if it doesn't fit we have two choices
			// either we loop over count and add the values individually
			// or we convert to a hugeint and multiply the hugeint
			// the problem is that hugeint multiplication is expensive
			// hence we switch here: with a low count we do the loop
			// with a high count we do the hugeint multiplication
			if (count < 8) {
				for (idx_t i = 0; i < count; i++) {
					AddValue(state.value, uint64_t(input), input >= 0);
				}
			} else {
				hugeint_t addition = hugeint_t(input) * Hugeint::Convert(count);
				state.value += addition;
			}
		}
	}
};

template <class STATEOP, class ADDOP>
struct BaseSumOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.value = 0;
		STATEOP::template Initialize<STATE>(state);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
		STATEOP::template Combine<STATE>(source, target, aggr_input_data);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		STATEOP::template AddValues<STATE>(state, 1);
		ADDOP::template AddNumber<STATE, INPUT_TYPE>(state, input);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &, idx_t count) {
		STATEOP::template AddValues<STATE>(state, count);
		ADDOP::template AddConstant<STATE, INPUT_TYPE>(state, input, count);
	}
	static bool IgnoreNull() {
		return true;
	}
};

template <class STATE, class ADDOP>
static inline void AddFORPartial(STATE &state, int64_t sum) {
	if (sum == 0)
		return;
	state.isset = true;
	ADDOP::template AddNumber<STATE, int64_t>(state, sum);
}
template <class INPUT_TYPE>
static bool TryGetFORSumScanData(Vector &input, idx_t count,
                                 FORVector::ScanData<typename FORUnsignedType<INPUT_TYPE>::type> &sd) {
	return FORVector::TryGetScanData(input, sd) && count != 0 &&
	       NumericCast<int64_t>(sd.max_value) <= NumericLimits<int64_t>::Maximum() / NumericCast<int64_t>(count);
}
template <class INPUT_TYPE, class STATE, class ADDOP>
static bool TryFORSimpleSumUpdate(Vector &input, STATE &state, idx_t count) {
	using U = typename FORUnsignedType<INPUT_TYPE>::type;
	FORVector::ScanData<U> sd;
	if (!TryGetFORSumScanData<INPUT_TYPE>(input, count, sd))
		return false;
	return FORVector::DispatchStoredType(sd.stored_type, [&](auto tag) {
		using S = typename decltype(tag)::type;
		int64_t sum = 0;
		FORVector::ForEachValue<S>(sd, count, [&](idx_t, S v) { sum += UnsafeNumericCast<int64_t>(v); });
		AddFORPartial<STATE, ADDOP>(state, sum);
		return true;
	});
}
template <class INPUT_TYPE, class STORED_T, class STATE, class ADDOP>
static void FORGroupedSum(const FORVector::ScanData<typename FORUnsignedType<INPUT_TYPE>::type> &sd, idx_t count,
                          STATE *const *states, const SelectionVector &sel, idx_t max_gid) {
	if (max_gid >= 256) { // T2: run-length locality
		idx_t prev = DConstants::INVALID_INDEX;
		int64_t sum = 0;
		FORVector::ForEachValue<STORED_T>(sd, count, [&](idx_t i, STORED_T v) {
			auto gid = sel.get_index(i);
			if (prev != DConstants::INVALID_INDEX && gid != prev) {
				AddFORPartial<STATE, ADDOP>(*states[prev], sum);
				sum = 0;
			}
			sum += UnsafeNumericCast<int64_t>(v);
			prev = gid;
		});
		if (prev != DConstants::INVALID_INDEX)
			AddFORPartial<STATE, ADDOP>(*states[prev], sum);
		return;
	}
	// T3: bitmap-tracked ≤256 groups
	uint64_t seen_bm[4] = {};
	uint8_t seen[256];
	int64_t sums[256] = {};
	idx_t n_seen = 0;
	FORVector::ForEachValue<STORED_T>(sd, count, [&](idx_t i, STORED_T v) {
		auto gid = sel.get_index(i);
		auto bit = uint64_t(1) << (gid & 63);
		if (!(seen_bm[gid >> 6] & bit)) {
			seen_bm[gid >> 6] |= bit;
			seen[n_seen++] = UnsafeNumericCast<uint8_t>(gid);
		}
		sums[gid] += UnsafeNumericCast<int64_t>(v);
	});
	for (idx_t i = 0; i < n_seen; i++)
		AddFORPartial<STATE, ADDOP>(*states[seen[i]], sums[seen[i]]);
}
template <class INPUT_TYPE, class STATE, class ADDOP>
static bool TryFORGroupedSumUpdate(Vector &input, Vector &states, idx_t count) {
	using U = typename FORUnsignedType<INPUT_TYPE>::type;
	FORVector::ScanData<U> sd;
	if (!TryGetFORSumScanData<INPUT_TYPE>(input, count, sd))
		return false;
	UnifiedVectorFormat sdata;
	states.ToUnifiedFormat(count, sdata);
	auto ptrs = UnifiedVectorFormat::GetData<STATE *>(sdata);
	auto &sel = *sdata.sel;
	idx_t max_gid = 0;
	for (idx_t i = 0; i < count; i++)
		max_gid = MaxValue(max_gid, sel.get_index(i));
	return FORVector::DispatchStoredType(sd.stored_type, [&](auto tag) {
		using S = typename decltype(tag)::type;
		FORGroupedSum<INPUT_TYPE, S, STATE, ADDOP>(sd, count, ptrs, sel, max_gid);
		return true;
	});
}

} // namespace duckdb
