#include "executor.hpp"


#include <nlohmann/json.hpp>


using json=nlohmann::json;



TaskExecutor::TaskExecutor(
    ModelRegistry* registry
)
{
    this->registry=registry;
}





std::string TaskExecutor::execute(
    const Task& task
)
{

    ModelInfo model =
    registry->getModel(
        task.model
    );



    json request;


    request["input"]
    =
    task.payload;



    return client.send(

        model.host,

        model.port,

        request.dump()

    );

}
