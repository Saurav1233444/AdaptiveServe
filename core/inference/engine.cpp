#include "engine.hpp"
#include "core/monitor/resource_monitor.hpp"
#include "core/registry/model_registry.hpp"
#include "core/scheduler/bounded_scheduler.hpp"
#include "core/utils/json_logger.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <unordered_map>
namespace adaptiveserve {
namespace {
using Clock = std::chrono::steady_clock;
struct SessionState {
  Ort::Session session;
  std::string input_name;
  std::string output_name;
  explicit SessionState(Ort::Session &&value) : session(std::move(value)) {}
};
double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

class InferenceEngine::Impl {
public:
  Impl(const std::filesystem::path &registry_path, std::size_t threads)
      : registry(registry_path),
        env(ORT_LOGGING_LEVEL_WARNING, "AdaptiveServe"), scheduler(threads),
        threads_(threads) {}

  std::pair<std::shared_ptr<SessionState>, double>
  session_for(const ModelInfo &model) {
    std::lock_guard lock(sessions_mutex);
    if (const auto found = sessions.find(model.name); found != sessions.end())
      return {found->second, 0.0};
    if (!std::filesystem::is_regular_file(model.onnx_path))
      throw std::runtime_error("ONNX model artifact is missing for '" +
                               model.name + "': " + model.onnx_path.string() +
                               ". Run the model export command first.");

    const auto started = Clock::now();
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(static_cast<int>(threads_));
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto state = std::make_shared<SessionState>(
        Ort::Session(env, model.onnx_path.c_str(), options));
    if (state->session.GetInputCount() != 1 ||
        state->session.GetOutputCount() < 1)
      throw std::runtime_error("model '" + model.name +
                               "' must have one input and at least one output");
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = state->session.GetInputNameAllocated(0, allocator);
    const auto output_name =
        state->session.GetOutputNameAllocated(0, allocator);
    state->input_name = input_name.get();
    state->output_name = output_name.get();
    const auto input_type = state->session.GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error(
          "model '" + model.name + "' input must be float32 (reported type " +
          std::to_string(static_cast<int>(input_info.GetElementType())) + ")");
    const auto shape = input_info.GetShape();
    if (shape.size() != 4 || (shape[0] > 0 && shape[0] != 1) ||
        (shape[1] > 0 && shape[1] != 3) ||
        (shape[2] > 0 && shape[2] != static_cast<int64_t>(model.input_size)) ||
        (shape[3] > 0 && shape[3] != static_cast<int64_t>(model.input_size)))
      throw std::runtime_error("model '" + model.name +
                               "' input shape does not match registry [1,3," +
                               std::to_string(model.input_size) + "," +
                               std::to_string(model.input_size) + "]");
    const double load_ms = elapsed_ms(started, Clock::now());
    sessions.emplace(model.name, state);
    JsonLogger::write("model_loaded", {{"model", model.name},
                                       {"load_ms", load_ms},
                                       {"path", model.onnx_path.string()}});
    return {std::move(state), load_ms};
  }

  ModelRegistry registry;
  Ort::Env env;
  BoundedScheduler scheduler;
  std::size_t threads_;
  mutable std::mutex sessions_mutex;
  std::unordered_map<std::string, std::shared_ptr<SessionState>> sessions;
};

InferenceEngine::InferenceEngine(const std::filesystem::path &registry_path,
                                 std::size_t threads)
    : impl_(std::make_unique<Impl>(registry_path, threads)) {}
InferenceEngine::~InferenceEngine() = default;
InferenceEngine::InferenceEngine(InferenceEngine &&) noexcept = default;
InferenceEngine &
InferenceEngine::operator=(InferenceEngine &&) noexcept = default;

PredictionResult InferenceEngine::predict(std::string_view model_name,
                                          std::span<const float> tensor) {
  auto lease = impl_->scheduler.try_acquire();
  if (!lease) {
    JsonLogger::write("request_rejected",
                      {{"model", model_name}, {"reason", "overloaded"}});
    throw std::runtime_error(
        "inference engine overloaded: all bounded execution slots are busy");
  }
  const auto &model = impl_->registry.at(model_name);
  const std::size_t expected = 3 * model.input_size * model.input_size;
  if (tensor.size() != expected)
    throw std::invalid_argument("tensor for '" + model.name +
                                "' must contain " + std::to_string(expected) +
                                " float32 values");
  auto [session, load_ms] = impl_->session_for(model);
  const std::array<int64_t, 4> shape{1, 3,
                                     static_cast<int64_t>(model.input_size),
                                     static_cast<int64_t>(model.input_size)};
  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(
      memory, const_cast<float *>(tensor.data()), tensor.size(), shape.data(),
      shape.size());
  const char *input_names[] = {session->input_name.c_str()};
  const char *output_names[] = {session->output_name.c_str()};
  const auto resources_before = ResourceMonitor::snapshot();
  const auto started = Clock::now();
  auto outputs = session->session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input, 1, output_names, 1);
  const auto finished = Clock::now();
  const auto resources_after = ResourceMonitor::snapshot();
  if (outputs.empty() || !outputs.front().IsTensor())
    throw std::runtime_error("model '" + model.name +
                             "' returned no tensor output");
  const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
  if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("model '" + model.name +
                             "' output must be float32 logits");
  const auto count = output_info.GetElementCount();
  const float *values = outputs.front().GetTensorData<float>();
  PredictionResult result{
      {values, values + count},
      elapsed_ms(started, finished),
      load_ms,
      resources_after.rss_mb,
      ResourceMonitor::cpu_percent(resources_before, resources_after)};
  JsonLogger::write("prediction", {{"model", model.name},
                                   {"inference_ms", result.inference_ms},
                                   {"load_ms", result.load_ms},
                                   {"memory_mb", result.memory_mb},
                                   {"cpu_percent", result.cpu_percent}});
  return result;
}

void InferenceEngine::unload(std::string_view model_name) {
  static_cast<void>(impl_->registry.at(model_name));
  std::lock_guard lock(impl_->sessions_mutex);
  const bool removed = impl_->sessions.erase(std::string(model_name)) > 0;
  JsonLogger::write("model_unloaded",
                    {{"model", model_name}, {"was_loaded", removed}});
}

std::vector<std::string> InferenceEngine::loaded_models() const {
  std::lock_guard lock(impl_->sessions_mutex);
  std::vector<std::string> names;
  names.reserve(impl_->sessions.size());
  for (const auto &[name, ignored] : impl_->sessions)
    names.push_back(name);
  std::sort(names.begin(), names.end());
  return names;
}
} // namespace adaptiveserve
