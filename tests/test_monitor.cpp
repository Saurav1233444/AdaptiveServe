#include "core/monitor/resource_monitor.hpp"

#include <iostream>
#include <stdexcept>

int main() {
  const auto before = adaptiveserve::ResourceMonitor::snapshot();
  volatile unsigned long value = 0;
  for (unsigned long index = 0; index < 2'000'000; ++index)
    value += index;
  const auto after = adaptiveserve::ResourceMonitor::snapshot();
  const auto cpu = adaptiveserve::ResourceMonitor::cpu_percent(before, after);
  if (after.rss_mb <= 0.0)
    throw std::runtime_error("process RSS must be positive");
  if (cpu < 0.0)
    throw std::runtime_error("process CPU percentage cannot be negative");
  std::cout << "monitor tests passed\n";
}
