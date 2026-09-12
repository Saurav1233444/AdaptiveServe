#pragma once

#include <string>


class WorkerClient
{

public:


    std::string send(
        std::string host,
        int port,
        std::string data
    );

};
