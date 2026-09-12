#pragma once
#include <vector>
#include <string>
#include "model_info.hpp"

class ModelRegistry{
private:
    std::vector<ModelInfo> models;

public:
    void load(const std::string& path);

    std::vector<ModelInfo> getAll() const;

    std::vector<ModelInfo> getByType(
        const std::string& type
    ) const;

    ModelInfo getModel(
        const std::string& name
    ) const;

    bool exists(
        const std::string& name
    ) const;
};
