#include "router.hpp"

#include <iostream>


AdaptiveRouter::AdaptiveRouter(
    ModelRegistry* registry
)
{
    this->registry=registry;
}




ModelInfo AdaptiveRouter::select(
    const std::string& type
)
{

    auto models=
    registry->getByType(type);



    double bestScore=-1;


    ModelInfo best;



    for(const auto& model:models)
    {

        double score=
        scorer.calculate(
            model,
            policy
        );


        std::cout
        <<model.name
        <<" score = "
        <<score
        <<std::endl;



        if(score>bestScore)
        {
            bestScore=score;

            best=model;
        }

    }


    return best;
}
