#include "core/inference/engine.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
int main(int argc, char **argv) {
  if (argc != 2)
    throw std::runtime_error(
        "expected path to tiny float32 [1,3,2,2] ONNX model");
  const auto root =
      std::filesystem::temp_directory_path() / "adaptiveserve-onnx-integration";
  std::filesystem::create_directories(root / "configs");
  std::filesystem::copy_file(argv[1], root / "tiny.onnx",
                             std::filesystem::copy_options::overwrite_existing);
  std::ofstream registry(root / "configs/models.json");
  registry
      << R"JSON({"models":[{"name":"tiny","display_name":"Tiny","onnx_path":"tiny.onnx",
      "input_size":2,"resize_size":2,"interpolation":"bilinear","accuracy":null,
      "latency_ms":null,"memory_mb":null,"input_type":"float32_nchw","endpoint":"local","weights":"test"}]})JSON";
  registry.close();
  adaptiveserve::InferenceEngine engine(root / "configs/models.json", 1);
  const auto cold = engine.predict("tiny", std::vector<float>(12, 1.0F));
  if (cold.logits.empty() || cold.load_ms <= 0.0)
    throw std::runtime_error("cold inference did not load and run model");
  const auto warm = engine.predict("tiny", std::vector<float>(12, 1.0F));
  if (warm.load_ms != 0.0)
    throw std::runtime_error("warm inference unexpectedly reloaded model");
  engine.unload("tiny");
  if (!engine.loaded_models().empty())
    throw std::runtime_error("unload did not release cached session");
}
