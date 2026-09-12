#include <iostream>
#include "../core/registry/model_registry.hpp"

int main(){
    ModelRegistry registry;
    registry.load("configs/models.json");

    auto models=registry.getByType("vision");

    for(const auto& m:models){
        std::cout
        <<m.name<<" "
        <<m.port<<" "
        <<m.accuracy<<" "
        <<m.latency<<" "
        <<m.memory<<"\n";
    }

    return 0;
}
