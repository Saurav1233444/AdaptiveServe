#include "model_info.hpp"

ModelInfo::ModelInfo(){
    name="";
    type="";
    host="127.0.0.1";
    port=0;
    accuracy=0;
    latency=0;
    memory=0;
}

ModelInfo::ModelInfo(
    std::string name,
    std::string type,
    std::string host,
    int port,
    double accuracy,
    double latency,
    double memory
){
    this->name=name;
    this->type=type;
    this->host=host;
    this->port=port;
    this->accuracy=accuracy;
    this->latency=latency;
    this->memory=memory;
}
