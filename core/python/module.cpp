#include "core/inference/engine.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
PYBIND11_MODULE(adaptive_core, module) {
  module.doc() = "AdaptiveServe ONNX Runtime C++ inference engine";
  py::class_<adaptiveserve::InferenceEngine>(module, "Engine")
      .def(py::init([](const std::string &registry_path, std::size_t threads) {
             return adaptiveserve::InferenceEngine(registry_path, threads);
           }),
           py::arg("registry_path"), py::arg("threads") = 1)
      .def(
          "predict",
          [](adaptiveserve::InferenceEngine &engine, const std::string &model,
             py::array_t<float, py::array::c_style> tensor) {
            if (tensor.ndim() != 4 || tensor.shape(0) != 1 ||
                tensor.shape(1) != 3)
              throw py::value_error("tensor must have shape [1,3,H,W]");
            const auto size = static_cast<std::size_t>(tensor.size());
            const auto *data = tensor.data();
            adaptiveserve::PredictionResult result;
            {
              py::gil_scoped_release release;
              result =
                  engine.predict(model, std::span<const float>(data, size));
            }
            py::dict output;
            output["logits"] = std::move(result.logits);
            output["inference_ms"] = result.inference_ms;
            output["load_ms"] = result.load_ms;
            output["memory_mb"] = result.memory_mb;
            output["cpu_percent"] = result.cpu_percent;
            return output;
          },
          py::arg("model_name"), py::arg("tensor"))
      .def(
          "unload",
          [](adaptiveserve::InferenceEngine &engine, const std::string &model) {
            py::gil_scoped_release release;
            engine.unload(model);
          })
      .def("loaded_models", &adaptiveserve::InferenceEngine::loaded_models);
}
