#include "msentity/core/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>
#include <unordered_set>

#include "msentity/core/ms_dataset.hpp"

namespace msentity {
namespace {

bool missing(const Value& value) {
  if (is_null(value)) return true;
  const auto* number = std::get_if<double>(&value);
  return number && std::isnan(*number);
}

bool number(const Value& value, double& out) {
  if (const auto* v = std::get_if<bool>(&value)) out = *v ? 1.0 : 0.0;
  else if (const auto* v = std::get_if<int64_t>(&value)) out = static_cast<double>(*v);
  else if (const auto* v = std::get_if<double>(&value)) out = *v;
  else if (const auto* v = std::get_if<std::string>(&value)) {
    std::size_t used = 0;
    try { out = std::stod(*v, &used); } catch (const std::exception&) { return false; }
    if (used != v->size()) return false;
  }
  else return false;
  return std::isfinite(out);
}

std::string text(const Value& value) {
  if (const auto* v = std::get_if<std::string>(&value)) return *v;
  if (const auto* v = std::get_if<bool>(&value)) return *v ? "true" : "false";
  if (const auto* v = std::get_if<int64_t>(&value)) return std::to_string(*v);
  if (const auto* v = std::get_if<double>(&value)) return std::to_string(*v);
  return "";
}

std::string folded(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::vector<bool> condition_mask(const MSDataset& dataset, const FilterCondition& condition) {
  const auto n = static_cast<std::size_t>(dataset.size());
  if (condition.op == FilterOperator::Mask) {
    if (condition.mask.size() != n) throw std::invalid_argument("filter mask length does not match dataset length");
    return condition.mask;
  }
  if (condition.op == FilterOperator::RowIds) {
    std::unordered_set<int64_t> wanted(condition.row_ids.begin(), condition.row_ids.end());
    std::vector<bool> result(n);
    const auto& ids = dataset.peaks().index();
    for (std::size_t i = 0; i < n; ++i) result[i] = wanted.contains(ids[i]);
    return result;
  }
  if (condition.op == FilterOperator::SpectrumPeak || condition.op == FilterOperator::NeutralLoss) {
    if (!std::isfinite(condition.mz) || !std::isfinite(condition.tolerance) || condition.tolerance < 0 ||
        !std::isfinite(condition.minimum_relative_intensity) || condition.minimum_relative_intensity < 0 ||
        condition.minimum_relative_intensity > 1) {
      throw std::invalid_argument("invalid peak filter parameters");
    }
    std::vector<double> precursor(n);
    if (condition.op == FilterOperator::NeutralLoss) {
      if (!dataset.has_column(condition.precursor_column)) throw KeyError("column not available in current view: " + condition.precursor_column);
      auto values = dataset.column_values(condition.precursor_column);
      for (std::size_t i = 0; i < n; ++i) {
        if (!number(values[i], precursor[i])) throw std::invalid_argument("neutral loss precursor column must be finite numeric values");
      }
    }
    const double delta = condition.tolerance_ppm ? std::abs(condition.mz) * condition.tolerance / 1'000'000.0 : condition.tolerance;
    std::vector<bool> result(n, false);
    for (std::size_t i = 0; i < n; ++i) {
      auto spectrum = dataset.peaks().spectrum(static_cast<int64_t>(i));
      double base = 0;
      for (int64_t p = 0; p < spectrum.size(); ++p) base = std::max(base, spectrum.intensity(p));
      for (int64_t p = 0; p < spectrum.size(); ++p) {
        double x = condition.op == FilterOperator::NeutralLoss ? precursor[i] - spectrum.mz(p) : spectrum.mz(p);
        if (std::abs(x - condition.mz) <= delta &&
            (condition.minimum_relative_intensity == 0 || (base > 0 && spectrum.intensity(p) / base >= condition.minimum_relative_intensity))) {
          result[i] = true;
          break;
        }
      }
    }
    return result;
  }
  if (condition.column.empty() || !dataset.has_column(condition.column)) throw KeyError("column not available in current view: " + condition.column);
  auto values = dataset.column_values(condition.column);
  std::vector<bool> result(n, false);
  for (std::size_t i = 0; i < n; ++i) {
    const Value& left = values[i];
    if (condition.op == FilterOperator::IsMissing || condition.op == FilterOperator::IsNotMissing) {
      result[i] = condition.op == FilterOperator::IsMissing ? missing(left) : !missing(left);
      continue;
    }
    if (condition.op >= FilterOperator::NumericEquals && condition.op <= FilterOperator::LessEqual) {
      double a = 0, b = 0;
      if (!number(left, a) || !number(condition.value, b)) continue;
      switch (condition.op) {
        case FilterOperator::NumericEquals: result[i] = a == b; break;
        case FilterOperator::Greater: result[i] = a > b; break;
        case FilterOperator::GreaterEqual: result[i] = a >= b; break;
        case FilterOperator::Less: result[i] = a < b; break;
        case FilterOperator::LessEqual: result[i] = a <= b; break;
        default: break;
      }
      continue;
    }
    std::string a = text(left), b = text(condition.value);
    if (!condition.case_sensitive) { a = folded(std::move(a)); b = folded(std::move(b)); }
    if (condition.op == FilterOperator::TextEquals) result[i] = a == b;
    else if (condition.op == FilterOperator::TextNotEquals) result[i] = a != b;
    else if (condition.op == FilterOperator::Contains) result[i] = a.find(b) != std::string::npos;
    else throw std::invalid_argument("invalid filter operator");
  }
  return result;
}

std::vector<bool> evaluate_group(const MSDataset& dataset, const FilterGroup& group) {
  if (group.logic != FilterLogic::And && group.logic != FilterLogic::Or) throw std::invalid_argument("invalid filter group logic");
  std::vector<bool> result(static_cast<std::size_t>(dataset.size()), group.logic == FilterLogic::And);
  for (const auto& child : group.children) {
    auto mask = std::visit([&](const auto& item) -> std::vector<bool> {
      using T = std::decay_t<decltype(item)>;
      if constexpr (std::is_same_v<T, FilterCondition>) return condition_mask(dataset, item);
      else return evaluate_group(dataset, item);
    }, child);
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = group.logic == FilterLogic::And ? result[i] && mask[i] : result[i] || mask[i];
    }
  }
  return result;
}
}  // namespace

std::vector<bool> FilterEvaluator::evaluate(const MSDataset& dataset, const FilterGroup& group) { return evaluate_group(dataset, group); }
MSDataset FilterEvaluator::filter(const MSDataset& dataset, const FilterGroup& group) { return dataset.filter(evaluate(dataset, group)); }

}  // namespace msentity
