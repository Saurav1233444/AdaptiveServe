#pragma once
#include "model_info.hpp"
#include <filesystem>
#include <string_view>
#include <vector>
namespace adaptiveserve {
class ModelRegistry {
public:
  explicit ModelRegistry(const std::filesystem::path &path);
  [[nodiscard]] const std::vector<ModelInfo> &all() const noexcept;
  [[nodiscard]] const ModelInfo &at(std::string_view name) const;

private:
  std::vector<ModelInfo> models_;
};
} // namespace adaptiveserve
