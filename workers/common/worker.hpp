#pragma once

#include <string>


class Worker
{

protected:

    std::string model_name;


public:

    Worker(
        const std::string& name
    );


    virtual std::string predict(
        const std::string& input
    )=0;


    virtual ~Worker(){}

};
