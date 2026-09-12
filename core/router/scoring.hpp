#pragma once

#include "../registry/model_info.hpp"
#include "policy.hpp"


class ScoringEngine
{

public:

    double calculate(
        const ModelInfo& model,
        const RouterPolicy& policy
    );


private:

    double latencyScore(
        double latency
    );


    double memoryScore(
        double memory
    );

};
