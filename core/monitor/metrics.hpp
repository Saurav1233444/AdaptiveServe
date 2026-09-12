#pragma once

#include <string>


struct Metrics
{

    std::string request_id;


    std::string model;


    std::string type;


    double latency;


    double routing_score;


    double cpu_usage;


    double memory_usage;


};
