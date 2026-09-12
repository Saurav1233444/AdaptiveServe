#pragma once
#include <string>

struct ModelInfo{
    std::string name;
    std::string type;
    std::string host;
    int port;
    double accuracy;
    double latency;
    double memory;

    ModelInfo();

    ModelInfo(
        std::string name,
        std::string type,
        std::string host,
        int port,
        double accuracy,
        double latency,
        double memory
    );
};
