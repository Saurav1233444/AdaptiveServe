#pragma once


#include "queue.hpp"



class Scheduler
{

private:

    RequestQueue queue;



public:


    void submit(
        const Task& task
    );


    Task next();


    bool hasTask() const;


    int pending() const;


};
