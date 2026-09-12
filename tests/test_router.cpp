#include <iostream>

#include "../core/registry/model_registry.hpp"

#include "../core/router/router.hpp"



int main()
{

    ModelRegistry registry;


    registry.load(
        "configs/models.json"
    );



    AdaptiveRouter router(
        &registry
    );



    auto model=
    router.select(
        "vision"
    );



    std::cout
    <<"Selected : "
    <<model.name
    <<std::endl;


    return 0;
}
