#include <iostream>

#include "../core/scheduler/scheduler.hpp"


int main()
{

    Scheduler scheduler;


    Task t;


    t.id="1";

    t.model="mobilenet";

    t.payload="image1";



    scheduler.submit(t);



    while(scheduler.hasTask())
    {

        auto task =
        scheduler.next();


        std::cout
        <<task.model
        <<" "
        <<task.payload
        <<std::endl;

    }


    return 0;
}
