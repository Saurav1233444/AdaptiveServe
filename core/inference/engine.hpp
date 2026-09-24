#pragma once
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace adaptiveserve {
struct PredictionResult {
  std::vector<float> logits;
  double inference_ms = 0.0;
  double load_ms = 0.0;
  double memory_mb = 0.0;
  double cpu_percent = 0.0;
};
class InferenceEngine {
public:
  explicit InferenceEngine(const std::filesystem::path &registry_path,
                           std::size_t threads = 1);
  ~InferenceEngine();
  InferenceEngine(InferenceEngine &&) noexcept;
  InferenceEngine &operator=(InferenceEngine &&) noexcept;
  InferenceEngine(const InferenceEngine &) = delete;
  InferenceEngine &operator=(const InferenceEngine &) = delete;
  [[nodiscard]] PredictionResult predict(std::string_view model_name,
                                         std::span<const float> tensor);
  void unload(std::string_view model_name);
  [[nodiscard]] std::vector<std::string> loaded_models() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace adaptiveserve
