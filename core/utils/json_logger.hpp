#pragma once
#include <nlohmann/json_fwd.hpp>
#include <string_view>
namespace adaptiveserve {
class JsonLogger {
public:
  static void write(std::string_view event, nlohmann::json fields);
};
} // namespace adaptiveserve
