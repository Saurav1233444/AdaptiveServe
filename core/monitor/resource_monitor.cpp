#include "resource_monitor.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
namespace adaptiveserve {
ResourceSnapshot ResourceMonitor::snapshot() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  std::ifstream statm("/proc/self/statm");
  long total_pages = 0;
  long resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages))
    throw std::runtime_error("failed to read /proc/self/statm");
  const auto seconds = [](const timeval &value) {
    return static_cast<double>(value.tv_sec) + value.tv_usec / 1'000'000.0;
  };
  return {std::chrono::steady_clock::now(),
          seconds(usage.ru_utime) + seconds(usage.ru_stime),
          resident_pages * static_cast<double>(sysconf(_SC_PAGESIZE)) /
              (1024.0 * 1024.0)};
}
double ResourceMonitor::cpu_percent(const ResourceSnapshot &before,
                                    const ResourceSnapshot &after) noexcept {
  const auto wall =
      std::chrono::duration<double>(after.wall - before.wall).count();
  if (wall <= 0.0)
    return 0.0;
  return std::max(0.0,
                  (after.process_cpu_seconds - before.process_cpu_seconds) /
                      wall * 100.0);
}
} // namespace adaptiveserve
