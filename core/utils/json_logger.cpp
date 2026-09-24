#include "json_logger.hpp"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
namespace adaptiveserve {
void JsonLogger::write(std::string_view event, nlohmann::json fields) {
  static std::mutex mutex;
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream timestamp;
  timestamp << std::put_time(&utc, "%FT%TZ");
  fields["timestamp"] = timestamp.str();
  fields["event"] = event;
  std::lock_guard lock(mutex);
  std::cerr << fields.dump() << '\n';
}
} // namespace adaptiveserve
