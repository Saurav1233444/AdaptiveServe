#include "rule_router.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
namespace adaptiveserve {
namespace {
bool feasible(const ModelInfo &profile, const RouteContext &context) {
  return context.cpu_available > 0.0 && profile.latency_ms &&
         profile.memory_mb &&
         *profile.latency_ms <= context.latency_budget_ms &&
         *profile.memory_mb <= context.memory_budget_mb;
}
} // namespace
RuleRouter::RuleRouter(std::vector<ModelInfo> profiles)
    : profiles_(std::move(profiles)) {
  if (profiles_.empty())
    throw std::invalid_argument("rule router requires model profiles");
  std::unordered_set<std::string> names;
  for (const auto &profile : profiles_) {
    if (profile.name.empty() || !names.insert(profile.name).second)
      throw std::invalid_argument("profile names must be non-empty and unique");
    if (!profile.accuracy || !profile.latency_ms || !profile.memory_mb)
      throw std::invalid_argument("rule router requires calibrated accuracy, "
                                  "latency, and memory profiles");
    if (!std::isfinite(*profile.accuracy) ||
        !std::isfinite(*profile.latency_ms) ||
        !std::isfinite(*profile.memory_mb) || *profile.accuracy < 0.0 ||
        *profile.accuracy > 1.0 || *profile.latency_ms < 0.0 ||
        *profile.memory_mb < 0.0)
      throw std::invalid_argument("model profile measurements are invalid");
  }
}
RouteDecision RuleRouter::select(const RouteContext &context) const {
  if (!std::isfinite(context.complexity) ||
      !std::isfinite(context.latency_budget_ms) ||
      !std::isfinite(context.cpu_available) ||
      !std::isfinite(context.memory_budget_mb))
    throw std::invalid_argument("routing context values must be finite");
  if (context.complexity < 0.0 || context.complexity > 1.0)
    throw std::invalid_argument("complexity must be in [0,1]");
  if (context.cpu_available < 0.0 || context.cpu_available > 1.0)
    throw std::invalid_argument("cpu_available must be in [0,1]");
  if (context.latency_budget_ms < 0.0 || context.memory_budget_mb < 0.0)
    throw std::invalid_argument("resource budgets must be non-negative");
  const std::string preferred = context.complexity < 0.4 ? "mobilenet_v3_small"
                                : context.complexity < 0.7 ? "resnet50"
                                                           : "efficientnet_b0";
  auto preferred_it =
      std::find_if(profiles_.begin(), profiles_.end(),
                   [&](const ModelInfo &p) { return p.name == preferred; });
  if (preferred_it != profiles_.end() && feasible(*preferred_it, context))
    return {preferred_it->name,
            "Complexity threshold selected the preferred feasible model.",
            true};

  const ModelInfo *best = nullptr;
  for (const auto &profile : profiles_) {
    if (!feasible(profile, context))
      continue;
    if (!best ||
        profile.accuracy.value_or(-1.0) > best->accuracy.value_or(-1.0))
      best = &profile;
  }
  if (best)
    return {best->name,
            "Preferred model was infeasible; selected highest-accuracy "
            "feasible model.",
            true};

  best = &*std::min_element(
      profiles_.begin(), profiles_.end(),
      [](const ModelInfo &left, const ModelInfo &right) {
        return std::pair(*left.latency_ms, *left.memory_mb) <
               std::pair(*right.latency_ms, *right.memory_mb);
      });
  return {best->name,
          "No calibrated profile satisfies all resource constraints; fastest "
          "fallback violates constraints.",
          false};
}
} // namespace adaptiveserve
