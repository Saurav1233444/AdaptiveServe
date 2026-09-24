#pragma once
#include "core/registry/model_info.hpp"
#include <string>
#include <vector>
namespace adaptiveserve {
struct RouteContext {
  double complexity;
  double latency_budget_ms;
  double cpu_available;
  double memory_budget_mb;
};
struct RouteDecision {
  std::string model;
  std::string reason;
  bool constraint_satisfied;
};
class RuleRouter {
public:
  explicit RuleRouter(std::vector<ModelInfo> profiles);
  [[nodiscard]] RouteDecision select(const RouteContext &context) const;

private:
  std::vector<ModelInfo> profiles_;
};
} // namespace adaptiveserve
