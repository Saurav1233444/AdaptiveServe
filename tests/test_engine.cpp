#include "core/inference/engine.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
  const auto root =
      std::filesystem::temp_directory_path() / "adaptiveserve-engine-test";
  std::filesystem::create_directories(root / "configs");
  std::ofstream registry(root / "configs/models.json");
  registry << R"JSON({"models":[{"name":"missing","display_name":"Missing",
      "onnx_path":"models/missing.onnx","input_size":224,"resize_size":256,
      "interpolation":"bilinear","accuracy":null,"latency_ms":null,"memory_mb":null,
      "input_type":"float32_nchw","endpoint":"local","weights":"test"}]})JSON";
  registry.close();

  adaptiveserve::InferenceEngine engine(root / "configs/models.json", 1);
  if (!engine.loaded_models().empty())
    throw std::runtime_error("sessions must load lazily");
  std::vector<float> input(1 * 3 * 224 * 224);
  try {
    static_cast<void>(engine.predict("missing", input));
  } catch (const std::exception &error) {
    const std::string message(error.what());
    if (message.find("ONNX model artifact is missing") == std::string::npos ||
        message.find("missing.onnx") == std::string::npos) {
      throw std::runtime_error("missing model error must be actionable: " +
                               message);
    }
    std::cout << "engine tests passed\n";
    return 0;
  }
  throw std::runtime_error("predict must reject a missing model artifact");
}
