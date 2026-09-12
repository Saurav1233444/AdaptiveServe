#pragma once


#include <httplib.h>


#include "../registry/model_registry.hpp"

#include "../router/router.hpp"

#include "../scheduler/scheduler.hpp"



class GatewayServer
{

private:

    ModelRegistry registry;

    AdaptiveRouter router;

    Scheduler scheduler;



public:

    GatewayServer();


    void start(
        int port
    );

};
