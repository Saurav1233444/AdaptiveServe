#pragma once

#include "queue.hpp"

#include "../../workers/common/worker_client.hpp"

#include "../registry/model_registry.hpp"


class TaskExecutor
{

private:

    WorkerClient client;


    ModelRegistry* registry;



public:


    TaskExecutor(
        ModelRegistry* registry
    );


    std::string execute(
        const Task& task
    );

};
