#pragma once
#include <chrono>
namespace adaptiveserve {
struct ResourceSnapshot {
  std::chrono::steady_clock::time_point wall;
  double process_cpu_seconds;
  double rss_mb;
};
class ResourceMonitor {
public:
  [[nodiscard]] static ResourceSnapshot snapshot();
  [[nodiscard]] static double
  cpu_percent(const ResourceSnapshot &before,
              const ResourceSnapshot &after) noexcept;
};
} // namespace adaptiveserve
