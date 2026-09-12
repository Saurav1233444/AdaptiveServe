#include "scoring.hpp"


double ScoringEngine::latencyScore(
    double latency
)
{
    return 1.0/
    (1.0+latency/100.0);
}



double ScoringEngine::memoryScore(
    double memory
)
{
    return 1.0/
    (1.0+memory/2000.0);
}




double ScoringEngine::calculate(
    const ModelInfo& model,
    const RouterPolicy& policy
)
{

    double score=0;


    score +=
    policy.accuracy_weight
    *
    model.accuracy;


    score +=
    policy.latency_weight
    *
    latencyScore(
        model.latency
    );


    score +=
    policy.memory_weight
    *
    memoryScore(
        model.memory
    );


    return score;
}
