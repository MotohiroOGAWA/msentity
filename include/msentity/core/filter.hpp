#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "msentity/value.hpp"

namespace msentity {

class MSDataset;

enum class FilterLogic { And, Or };

enum class FilterOperator {
  TextEquals,
  TextNotEquals,
  Contains,
  NumericEquals,
  Greater,
  GreaterEqual,
  Less,
  LessEqual,
  IsMissing,
  IsNotMissing,
  SpectrumPeak,
  NeutralLoss,
  Mask,
  RowIds,
};

struct FilterCondition {
  FilterOperator op = FilterOperator::TextEquals;
  std::string column;
  Value value{};
  bool case_sensitive = false;
  double mz = 0.0;
  double tolerance = 0.0;
  bool tolerance_ppm = false;
  double minimum_relative_intensity = 0.0;
  std::string precursor_column = "PrecursorMZ";
  std::vector<bool> mask;
  std::vector<int64_t> row_ids;
};

struct FilterGroup;
using FilterNode = std::variant<FilterCondition, FilterGroup>;

struct FilterGroup {
  FilterLogic logic = FilterLogic::And;
  std::vector<FilterNode> children;
};

class FilterEvaluator {
 public:
  // Every leaf is evaluated against the same input view. Groups only combine
  // masks; they never repeatedly materialize filtered datasets.
  static std::vector<bool> evaluate(const MSDataset& dataset, const FilterGroup& group);
  static MSDataset filter(const MSDataset& dataset, const FilterGroup& group);
};

}  // namespace msentity
