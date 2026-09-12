#pragma once

#include <string>

#include "../registry/model_registry.hpp"

#include "scoring.hpp"

#include "policy.hpp"



class AdaptiveRouter
{

private:

    ModelRegistry* registry;

    ScoringEngine scorer;

    RouterPolicy policy;



public:


    AdaptiveRouter(
        ModelRegistry* registry
    );


    ModelInfo select(
        const std::string& type
    );

};
