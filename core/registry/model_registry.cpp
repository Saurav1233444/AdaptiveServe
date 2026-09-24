#include "model_registry.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
namespace adaptiveserve {
namespace {
using Json = nlohmann::json;
template <typename T>
T required(const Json &item, const char *field, std::size_t index) {
  try {
    return item.at(field).get<T>();
  } catch (const std::exception &error) {
    throw std::runtime_error("models[" + std::to_string(index) + "]." + field +
                             " is invalid or missing: " + error.what());
  }
}
std::optional<double> optional_measurement(const Json &item, const char *field,
                                           std::size_t index) {
  if (!item.contains(field) || item.at(field).is_null())
    return std::nullopt;
  const auto value = required<double>(item, field, index);
  if (value < 0.0)
    throw std::runtime_error("models[" + std::to_string(index) + "]." + field +
                             " must be non-negative or null");
  return value;
}
} // namespace
ModelRegistry::ModelRegistry(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("failed to open model registry: " + path.string());
  Json document;
  try {
    stream >> document;
  } catch (const std::exception &error) {
    throw std::runtime_error("invalid model registry JSON: " +
                             std::string(error.what()));
  }
  if (!document.contains("models") || !document.at("models").is_array() ||
      document.at("models").empty())
    throw std::runtime_error(
        "model registry must contain a non-empty models array");
  const auto repository_root =
      std::filesystem::absolute(path).parent_path().parent_path();
  std::unordered_set<std::string> names;
  for (std::size_t index = 0; index < document.at("models").size(); ++index) {
    const auto &item = document.at("models").at(index);
    if (!item.is_object())
      throw std::runtime_error("models[" + std::to_string(index) +
                               "] must be an object");
    ModelInfo model;
    model.name = required<std::string>(item, "name", index);
    model.display_name = required<std::string>(item, "display_name", index);
    const auto configured_path =
        required<std::filesystem::path>(item, "onnx_path", index);
    model.onnx_path =
        (configured_path.is_absolute() ? configured_path
                                       : repository_root / configured_path)
            .lexically_normal();
    model.input_size = required<std::size_t>(item, "input_size", index);
    model.resize_size = required<std::size_t>(item, "resize_size", index);
    model.interpolation = required<std::string>(item, "interpolation", index);
    model.accuracy = optional_measurement(item, "accuracy", index);
    model.latency_ms = optional_measurement(item, "latency_ms", index);
    model.memory_mb = optional_measurement(item, "memory_mb", index);
    model.input_type = required<std::string>(item, "input_type", index);
    model.endpoint = required<std::string>(item, "endpoint", index);
    model.weights = required<std::string>(item, "weights", index);
    if (model.accuracy && *model.accuracy > 1.0)
      throw std::runtime_error("models[" + std::to_string(index) +
                               "].accuracy must be in [0,1] or null");
    if (model.name.empty() || !names.insert(model.name).second)
      throw std::runtime_error("model names must be non-empty and unique: " +
                               model.name);
    if (model.input_size == 0 || model.resize_size < model.input_size)
      throw std::runtime_error("invalid input/resize size for model: " +
                               model.name);
    if (model.interpolation != "bilinear" && model.interpolation != "bicubic")
      throw std::runtime_error("unsupported interpolation for model: " +
                               model.name);
    if (model.input_type.empty())
      throw std::runtime_error("input_type must be non-empty for model: " +
                               model.name);
    models_.push_back(std::move(model));
  }
}
const std::vector<ModelInfo> &ModelRegistry::all() const noexcept {
  return models_;
}
const ModelInfo &ModelRegistry::at(std::string_view name) const {
  for (const auto &model : models_)
    if (model.name == name)
      return model;
  throw std::out_of_range("unknown model: " + std::string(name));
}
} // namespace adaptiveserve
