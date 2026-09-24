#pragma once
#include <filesystem>
#include <optional>
#include <string>
namespace adaptiveserve {
struct ModelInfo {
  std::string name;
  std::string display_name;
  std::filesystem::path onnx_path;
  std::size_t input_size = 0;
  std::size_t resize_size = 0;
  std::string interpolation;
  std::optional<double> accuracy;
  std::optional<double> latency_ms;
  std::optional<double> memory_mb;
  std::string input_type;
  std::string endpoint;
  std::string weights;
};
} // namespace adaptiveserve
