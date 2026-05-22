#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/uuid.hpp"

namespace duckdb {

ExecuteFunctionState::ExecuteFunctionState(const Expression &expr, ExpressionExecutorState &root)
    : ExpressionState(expr, root) {
	// Check if the expression is eligible for dictionary optimization
	if (!expr.IsConsistent() || expr.IsVolatile() || expr.CanThrow()) {
		return; // Needs to be consistent, non-volatile, and non-throwing
	}

	if (expr.GetReturnType().InternalType() == PhysicalType::STRUCT) {
		return; // FIXME: get this working for STRUCT
	}
	dictionary_eligible = true;

	// Set input_col_idx accordingly, marking the expression as eligible for dictionary optimization
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_FUNCTION: {
		auto &bound_function = expr.Cast<BoundFunctionExpression>();
		auto &children = bound_function.children;
		for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
			auto &child = *children[child_idx];
			if (child.IsFoldable()) {
				continue; // Constant
			}
			if (input_col_idx.IsValid()) {
				input_col_idx.SetInvalid(); // Found more than 1 non-constant
				break;
			}
			if (child.GetReturnType().InternalType() == PhysicalType::STRUCT) {
				break; // FIXME
			}
			input_col_idx = child_idx;
		}
		break;
	}
	default:
		break;
	}
}

ExecuteFunctionState::~ExecuteFunctionState() {
}

bool ExecuteFunctionState::TryExecuteSlicedDictionaryExpression(const BoundFunctionExpression &expr, DataChunk &args,
                                                               ExpressionState &state, Vector &result) {
	static constexpr idx_t MIN_SLICED_DICTIONARY_COUNT = 256;

	if (!dictionary_eligible || args.size() <= MIN_SLICED_DICTIONARY_COUNT) {
		return false;
	}

	const SelectionVector *dict_sel = nullptr;
	optional_idx dict_size;
	bool has_dictionary = false;
	bool has_for_child = false;
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		auto &arg = args.data[col_idx];
		if (arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			continue;
		}
		if (arg.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
			return false;
		}
		if (!DictionaryVector::DictionaryId(arg).empty()) {
			return false;
		}
		auto &child = DictionaryVector::Child(arg);
		if (child.GetVectorType() != VectorType::FLAT_VECTOR && child.GetVectorType() != VectorType::FOR_VECTOR) {
			return false;
		}
		has_for_child = has_for_child || child.GetVectorType() == VectorType::FOR_VECTOR;
		auto child_size = DictionaryVector::DictionarySize(arg);
		if (!child_size.IsValid() || child_size.GetIndex() > STANDARD_VECTOR_SIZE) {
			return false;
		}
		auto &sel = DictionaryVector::SelVector(arg);
		if (!dict_sel) {
			dict_sel = &sel;
			dict_size = child_size;
		} else if (dict_sel->data() != sel.data() || dict_size.GetIndex() != child_size.GetIndex()) {
			return false;
		}
		has_dictionary = true;
	}
	if (!has_dictionary || !dict_sel || !dict_size.IsValid()) {
		return false;
	}
	if (!has_for_child) {
		return false;
	}

	if (sliced_dictionary_input.ColumnCount() != args.ColumnCount()) {
		sliced_dictionary_input.Destroy();
		sliced_dictionary_input.InitializeEmpty(args.GetTypes());
	}
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		auto &arg = args.data[col_idx];
		if (arg.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
			sliced_dictionary_input.data[col_idx].Reference(DictionaryVector::Child(arg));
		} else {
			sliced_dictionary_input.data[col_idx].Reference(arg);
		}
	}

	const auto child_count = dict_size.GetIndex();
	sliced_dictionary_input.SetCardinality(child_count);
	expr.function.GetFunctionCallback()(sliced_dictionary_input, state, result);
	auto child_result_type = result.GetVectorType();
	if (child_result_type == VectorType::FLAT_VECTOR || child_result_type == VectorType::CONSTANT_VECTOR ||
	    child_result_type == VectorType::FOR_VECTOR) {
		FlatVector::SetSize(result, child_count);
	}

	if (!sliced_dictionary_output) {
		sliced_dictionary_output = make_buffer<DictionaryEntry>(Vector(result.GetType(), nullptr));
	}
	sliced_dictionary_output->data.Reference(result);
	sliced_dictionary_output->cached_hashes.reset();
	if (!sliced_dictionary_buffer) {
		sliced_dictionary_buffer = make_buffer<DictionaryBuffer>(*dict_sel, args.size(), sliced_dictionary_output);
	} else {
		sliced_dictionary_buffer->SetEntry(sliced_dictionary_output);
		sliced_dictionary_buffer->SetSelVector(*dict_sel);
		sliced_dictionary_buffer->SetVectorSizeOnly(args.size());
	}
	result.SetBuffer(sliced_dictionary_buffer);
	return true;
}

bool ExecuteFunctionState::TryExecuteDictionaryExpression(const BoundFunctionExpression &expr, DataChunk &args,
                                                          ExpressionState &state, Vector &result) {
	static constexpr idx_t MAX_DICTIONARY_SIZE_THRESHOLD = 20000;
	static constexpr double CHUNK_FILL_RATIO_THRESHOLD = 0.5;

	if (TryExecuteSlicedDictionaryExpression(expr, args, state, result)) {
		return true;
	}

	if (!input_col_idx.IsValid()) {
		return false; // This expression is not eligible for dictionary optimization
	}

	// Figure out if we can do the optimization
	const auto &unary_input = args.data[input_col_idx.GetIndex()];
	if (unary_input.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
		return false; // Not a dictionary
	}
	// Skip dictionary optimization for DICTIONARY(FOR) — let the function handle FOR directly
	if (DictionaryVector::Child(unary_input).GetVectorType() == VectorType::FOR_VECTOR) {
		return false;
	}

	const auto input_dictionary_size_opt = DictionaryVector::DictionarySize(unary_input);
	const auto &input_dictionary_id = DictionaryVector::DictionaryId(unary_input);
	if (!input_dictionary_size_opt.IsValid() || input_dictionary_id.empty()) {
		return false; // Not a dictionary that comes from storage
	}

	const auto input_dictionary_size = input_dictionary_size_opt.GetIndex();
	if (input_dictionary_size >= MAX_DICTIONARY_SIZE_THRESHOLD) {
		return false; // Dictionary is too large, bail
	}

	if (!output_dictionary || current_input_dictionary_id != input_dictionary_id) {
		// We haven't seen this dictionary before
		const auto chunk_fill_ratio = static_cast<double>(args.size()) / STANDARD_VECTOR_SIZE;
		if (input_dictionary_size > STANDARD_VECTOR_SIZE && chunk_fill_ratio <= CHUNK_FILL_RATIO_THRESHOLD) {
			// If the dictionary size is <= STANDARD_VECTOR_SIZE, we always do the optimization
			// If it's greater, we only do the optimization if the chunk is more than 50% full
			// This protects the optimization against selective filters
			return false;
		}

		// We can do dictionary optimization! Re-initialize
		output_dictionary = DictionaryVector::CreateReusableDictionary(result.GetType(), input_dictionary_size);
		current_input_dictionary_id = input_dictionary_id;

		// Set up the input chunk
		DataChunk input_chunk;
		input_chunk.InitializeEmpty(args.GetTypes());
		for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
			if (col_idx != input_col_idx.GetIndex()) {
				input_chunk.data[col_idx].Reference(args.data[col_idx]);
			}
		}

		// Loop over the dictionary, executing at most STANDARD_VECTOR_SIZE at a time
		for (idx_t offset = 0; offset < input_dictionary_size; offset += STANDARD_VECTOR_SIZE) {
			const auto count = MinValue<idx_t>(input_dictionary_size - offset, STANDARD_VECTOR_SIZE);

			// Offset the input dictionary
			Vector offset_input(DictionaryVector::Child(unary_input), offset, offset + count);
			input_chunk.data[input_col_idx.GetIndex()].Reference(offset_input);
			input_chunk.SetCardinality(count);

			// Execute, storing the result in an intermediate vector, and copying it to the output dictionary
			Vector output_intermediate(result.GetType());
			expr.function.GetFunctionCallback()(input_chunk, state, output_intermediate);
			VectorOperations::Copy(output_intermediate, output_dictionary->data, count, 0, offset);
		}
	}

	// Result references the dictionary. Reuse the wrapper; the dictionary entry itself is
	// cached above and rebuilt only when the storage dictionary id changes.
	auto &sel = DictionaryVector::SelVector(unary_input);
	if (!output_dictionary_buffer) {
		output_dictionary_buffer = make_buffer<DictionaryBuffer>(sel, args.size(), output_dictionary);
	} else {
		output_dictionary_buffer->SetEntry(output_dictionary);
		output_dictionary_buffer->SetSelVector(sel);
		output_dictionary_buffer->SetVectorSizeOnly(args.size());
	}
	result.SetBuffer(output_dictionary_buffer);

	return true;
}

void ExecuteFunctionState::ResetDictionaryStates() {
	// Clear the cached dictionary information
	current_input_dictionary_id.clear();
	output_dictionary.reset();
	output_dictionary_buffer.reset();

	for (const auto &child_state : child_states) {
		child_state->ResetDictionaryStates();
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundFunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExecuteFunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(*child);
	}

	result->Finalize();
	if (expr.function.HasInitStateCallback()) {
		result->local_state = expr.function.GetInitStateCallback()(*result, expr, expr.bind_info.get());
	}
	return std::move(result);
}

static void VerifyNullHandling(const BoundFunctionExpression &expr, DataChunk &args, Vector &result) {
#ifdef DEBUG
	if (args.data.empty() || expr.function.GetNullHandling() != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return;
	}

	// Combine all the argument validity masks into a flat validity mask
	idx_t count = args.size();
	ValidityMask combined_mask(count);
	for (auto &arg : args.data) {
		auto entries = arg.Validity();
		if (!entries.CanHaveNull()) {
			continue;
		}
		for (idx_t i = 0; i < count; i++) {
			if (!entries.IsValid(i)) {
				combined_mask.SetInvalid(i);
			}
		}
	}

	// Default is that if any of the arguments are NULL, the result is also NULL
	auto result_validity = result.Validity();
	for (idx_t i = 0; i < count; i++) {
		if (!combined_mask.RowIsValid(i)) {
			D_ASSERT(!result_validity.IsValid(i));
		}
	}
#endif
}

static void ExecuteSelectFunction(const BoundFunctionExpression &expr, DataChunk &args, ExpressionState &state,
                                  Vector &result) {
	if (expr.GetReturnType() != LogicalType::BOOLEAN) {
		throw InvalidInputException("Function %s only has a select callback but returns %s", expr.function.GetName(),
		                            expr.GetReturnType().ToString());
	}
	if (expr.function.GetNullHandling() == FunctionNullHandling::SPECIAL_HANDLING) {
		throw InvalidInputException("Function %s only has a select callback with SPECIAL_HANDLING but projected "
		                            "execution requires a scalar callback to produce NULL results",
		                            expr.function.GetName());
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto count = args.size();
	auto result_data = FlatVector::GetDataMutable<bool>(result);
	for (idx_t i = 0; i < count; i++) {
		result_data[i] = false;
	}

	auto &result_validity = FlatVector::ValidityMutable(result);
	result_validity.SetAllValid(count);
	D_ASSERT(expr.function.GetNullHandling() == FunctionNullHandling::DEFAULT_NULL_HANDLING);
	for (auto &arg : args.data) {
		auto entries = arg.Validity();
		if (!entries.CanHaveNull()) {
			continue;
		}
		for (idx_t i = 0; i < count; i++) {
			if (!entries.IsValid(i)) {
				result_validity.SetInvalid(i);
			}
		}
	}

	SelectionVector true_sel(count);
	auto true_count =
	    expr.function.GetSelectCallback()(args, state, FlatVector::IncrementalSelectionVector(), &true_sel, nullptr);
	for (idx_t i = 0; i < true_count; i++) {
		result_data[true_sel.get_index(i)] = true;
	}
}

void ExpressionExecutor::Execute(const BoundFunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	state->intermediate_chunk.Reset();
	auto &arguments = state->intermediate_chunk;
	// if the input is constant and there function is non-volatile we only need to run it on one value
	bool all_constant = true;
	if (expr.function.GetStability() == FunctionStability::VOLATILE) {
		// we cannot optimize away constant vectors for volatile functions
		all_constant = false;
	}
	auto default_null_handling = expr.function.GetNullHandling() == FunctionNullHandling::DEFAULT_NULL_HANDLING;
	if (!state->types.empty()) {
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(state->types[i] == expr.children[i]->GetReturnType());
			Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
			if (arguments.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
				all_constant = false;
			} else if (default_null_handling && ConstantVector::IsNull(arguments.data[i])) {
				// constant NULL input: result is NULL
				ConstantVector::SetNull(result, count_t(count));
				return;
			}
		}
	}
	if (all_constant) {
		// if all arguments are constant temporarily set the child cardinality to 1
		arguments.SetChildCardinality(1ULL);
	} else {
		arguments.SetCardinality(count);
	}
	arguments.Verify(context);

	auto &execute_function_state = state->Cast<ExecuteFunctionState>();
	auto dictionary_executed = expr.function.HasFunctionCallback() && !all_constant &&
	                           execute_function_state.TryExecuteDictionaryExpression(expr, arguments, *state, result);
	if (!dictionary_executed) {
		if (expr.function.HasFunctionCallback()) {
			expr.function.GetFunctionCallback()(arguments, *state, result);
		} else if (expr.function.HasSelectCallback()) {
			ExecuteSelectFunction(expr, arguments, *state, result);
		} else {
			throw InternalException("Scalar function %s has neither an execution nor a select callback",
			                        expr.function.GetName());
		}
	}
	if (all_constant) {
		// restore the input cardinality
		for (auto &arg : arguments.data) {
			arg.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		arguments.SetChildCardinality(count);
		// ensure the result type is constant
		if (result.GetVectorType() != VectorType::FLAT_VECTOR &&
		    result.GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.Flatten();
		}
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	FlatVector::SetSize(result, count_t(count));

	VerifyNullHandling(expr, arguments, result);
	D_ASSERT(result.GetType() == expr.GetReturnType());
}

idx_t ExpressionExecutor::Select(const BoundFunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                 SelectionVector *false_sel) {
	if (!expr.function.HasSelectCallback()) {
		return DefaultSelect(expr, state, sel, count, true_sel, false_sel);
	}
	// FIXME: push constant handling in here similar to Execute
	state->intermediate_chunk.Reset();
	auto &arguments = state->intermediate_chunk;
	for (idx_t i = 0; i < expr.children.size(); i++) {
		D_ASSERT(state->types[i] == expr.children[i]->GetReturnType());
		Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
	}
	arguments.SetCardinality(count);
	arguments.Verify(context);
	return expr.function.GetSelectCallback()(arguments, *state, sel, true_sel, false_sel);
}

} // namespace duckdb
