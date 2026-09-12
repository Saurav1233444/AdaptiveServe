#include "model_registry.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json=nlohmann::json;

void ModelRegistry::load(const std::string& path){
    std::ifstream file(path);

    if(!file.is_open())
        throw std::runtime_error(
            "Failed to open model registry: "+path
        );

    json data;
    file>>data;

    models.clear();

    for(const auto& item:data.at("models")){
        ModelInfo model;

        model.name=item.at("name");
        model.type=item.at("type");
        model.host=item.value(
            "host",
            "127.0.0.1"
        );
        model.port=item.at("port");
        model.accuracy=item.value(
            "accuracy",
            0.0
        );
        model.latency=item.value(
            "latency",
            0.0
        );
        model.memory=item.value(
            "memory",
            0.0
        );

        models.push_back(model);
    }

    std::cout
    <<models.size()
    <<" models loaded\n";
}

std::vector<ModelInfo> ModelRegistry::getAll() const{
    return models;
}

std::vector<ModelInfo> ModelRegistry::getByType(
    const std::string& type
) const{
    std::vector<ModelInfo> result;

    for(const auto& model:models)
        if(model.type==type)
            result.push_back(model);

    return result;
}

ModelInfo ModelRegistry::getModel(
    const std::string& name
) const{
    for(const auto& model:models)
        if(model.name==name)
            return model;

    return ModelInfo();
}

bool ModelRegistry::exists(
    const std::string& name
) const{
    for(const auto& model:models)
        if(model.name==name)
            return true;

    return false;
}
