#include "latency.hpp"



void Timer::start()
{

    start_time =
    std::chrono::high_resolution_clock::now();

}





double Timer::stop()
{

    auto end_time =
    std::chrono::high_resolution_clock::now();



    std::chrono::duration<double,std::milli> duration =
    end_time-start_time;



    return duration.count();

}
