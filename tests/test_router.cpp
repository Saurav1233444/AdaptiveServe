#include "core/router/rule_router.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
adaptiveserve::ModelInfo model(std::string name, double accuracy,
                               double latency, double memory) {
  adaptiveserve::ModelInfo result;
  result.name = std::move(name);
  result.accuracy = accuracy;
  result.latency_ms = latency;
  result.memory_mb = memory;
  return result;
}
} // namespace

int main() {
  adaptiveserve::RuleRouter router({
      model("mobilenet_v3_small", 0.67, 5.0, 80.0),
      model("resnet50", 0.76, 12.0, 220.0),
      model("efficientnet_b0", 0.77, 9.0, 150.0),
  });
  auto decision = router.select({.complexity = 0.2,
                                 .latency_budget_ms = 20.0,
                                 .cpu_available = 1.0,
                                 .memory_budget_mb = 500.0});
  require(decision.model == "mobilenet_v3_small", "easy inputs use MobileNet");
  require(decision.constraint_satisfied,
          "feasible preference satisfies constraints");
  decision = router.select({.complexity = 0.55,
                            .latency_budget_ms = 10.0,
                            .cpu_available = 1.0,
                            .memory_budget_mb = 160.0});
  require(decision.model == "efficientnet_b0",
          "router must mask infeasible preferred model");
  require(decision.constraint_satisfied,
          "alternate feasible model satisfies constraints");
  decision = router.select({.complexity = 0.9,
                            .latency_budget_ms = 1.0,
                            .cpu_available = 1.0,
                            .memory_budget_mb = 10.0});
  require(decision.model == "mobilenet_v3_small",
          "fallback minimizes normalized violation");
  require(!decision.constraint_satisfied, "fallback violation must be visible");

  decision = router.select({.complexity = 0.2,
                            .latency_budget_ms = 1000.0,
                            .cpu_available = 0.0,
                            .memory_budget_mb = 1000.0});
  require(!decision.constraint_satisfied,
          "zero available CPU must make every profile infeasible");

  adaptiveserve::RuleRouter fallback_router(
      {model("fast", 0.4, 2.0, 1000.0), model("slow", 0.9, 9.0, 20.0)});
  decision = fallback_router.select({.complexity = 0.5,
                                     .latency_budget_ms = 0.0,
                                     .cpu_available = 1.0,
                                     .memory_budget_mb = 0.0});
  require(decision.model == "fast", "fallback must use fastest measured model");
  require(!decision.constraint_satisfied,
          "fastest fallback must report its violation");
  bool rejected = false;
  try {
    static_cast<void>(router.select({.complexity = 1.2,
                                     .latency_budget_ms = 10.0,
                                     .cpu_available = 1.0,
                                     .memory_budget_mb = 100.0}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "complexity outside [0,1] must be rejected");
  rejected = false;
  try {
    static_cast<void>(
        router.select({.complexity = std::numeric_limits<double>::quiet_NaN(),
                       .latency_budget_ms = 10.0,
                       .cpu_available = 1.0,
                       .memory_budget_mb = 100.0}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "non-finite context values must be rejected");
  std::cout << "router tests passed\n";
}
