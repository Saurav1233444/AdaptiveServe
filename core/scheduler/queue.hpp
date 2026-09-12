#pragma once

#include <queue>
#include <string>


struct Task
{

    std::string id;

    std::string model;

    std::string payload;


};



class RequestQueue
{

private:

    std::queue<Task> tasks;


public:


    void push(
        const Task& task
    );


    Task pop();


    bool empty() const;


    int size() const;


};
