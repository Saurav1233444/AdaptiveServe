#include "core/inference/engine.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
struct Arguments {
  std::string registry;
  std::string model;
  std::size_t threads = 1;
};
Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if ((option == "--registry" || option == "--model" ||
         option == "--threads") &&
        index + 1 >= argc)
      throw std::invalid_argument("missing value for " + option);
    if (option == "--registry")
      result.registry = argv[++index];
    else if (option == "--model")
      result.model = argv[++index];
    else if (option == "--threads")
      result.threads = std::stoul(argv[++index]);
    else
      throw std::invalid_argument("unknown argument: " + option);
  }
  if (result.registry.empty() || result.model.empty())
    throw std::invalid_argument(
        "usage: adaptive_worker --registry PATH --model NAME [--threads N]");
  return result;
}
} // namespace
int main(int argc, char **argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    adaptiveserve::InferenceEngine engine(args.registry, args.threads);
    std::string line;
    while (std::getline(std::cin, line)) {
      nlohmann::json response;
      try {
        const auto request = nlohmann::json::parse(line);
        const auto tensor = request.at("tensor").get<std::vector<float>>();
        const auto result = engine.predict(args.model, tensor);
        response = {{"logits", result.logits},
                    {"inference_ms", result.inference_ms},
                    {"load_ms", result.load_ms},
                    {"memory_mb", result.memory_mb},
                    {"cpu_percent", result.cpu_percent}};
      } catch (const std::exception &error) {
        response = {{"error", error.what()}};
      }
      std::cout << response.dump() << '\n' << std::flush;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
