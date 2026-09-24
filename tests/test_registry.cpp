#include "core/registry/model_registry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <typename Fn>
void require_throws(Fn &&fn, const std::string &fragment) {
  try {
    fn();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(fragment) != std::string::npos,
            "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected exception containing: " + fragment);
}
std::filesystem::path write_registry(const std::string &body) {
  const auto directory =
      std::filesystem::temp_directory_path() / "adaptiveserve-registry-test";
  std::filesystem::create_directories(directory / "configs");
  std::ofstream file(directory / "configs" / "models.json");
  file << body;
  return directory / "configs" / "models.json";
}
} // namespace

int main() {
  const auto path = write_registry(R"JSON({
      "models": [
        {"name":"mobilenet_v3_small","display_name":"MobileNetV3-Small",
         "onnx_path":"models/mobilenet.onnx","input_size":224,"resize_size":256,
         "interpolation":"bilinear","accuracy":null,"latency_ms":null,
         "memory_mb":null,"input_type":"float32_nchw","endpoint":"local",
         "weights":"IMAGENET1K_V1"},
        {"name":"resnet50","display_name":"ResNet-50",
         "onnx_path":"models/resnet.onnx","input_size":224,"resize_size":256,
         "interpolation":"bilinear","accuracy":0.76,"latency_ms":12.5,
         "memory_mb":310.0,"input_type":"float32_nchw","endpoint":"local",
         "weights":"IMAGENET1K_V2"}
      ]
    })JSON");
  adaptiveserve::ModelRegistry registry(path);
  require(registry.all().size() == 2, "registry must preserve all models");
  const auto &mobile = registry.at("mobilenet_v3_small");
  require(!mobile.accuracy.has_value(),
          "uncalibrated accuracy must remain null");
  require(mobile.onnx_path ==
              path.parent_path().parent_path() / "models/mobilenet.onnx",
          "ONNX path must resolve relative to repository root");
  require(registry.at("resnet50").latency_ms == 12.5,
          "measured latency must be retained");
  require_throws([&] { static_cast<void>(registry.at("missing")); },
                 "unknown model");
  const auto invalid = write_registry(R"JSON({"models":[{"name":"bad"}]})JSON");
  require_throws([&] { adaptiveserve::ModelRegistry ignored(invalid); },
                 "display_name");
  const auto invalid_accuracy = write_registry(R"JSON({"models":[
      {"name":"bad","display_name":"Bad","onnx_path":"bad.onnx","input_size":224,
       "resize_size":256,"interpolation":"bilinear","accuracy":1.2,"latency_ms":null,
       "memory_mb":null,"input_type":"float32_nchw","endpoint":"local","weights":"test"}
    ]})JSON");
  require_throws(
      [&] { adaptiveserve::ModelRegistry ignored(invalid_accuracy); },
      "accuracy");
  std::cout << "registry tests passed\n";
}
