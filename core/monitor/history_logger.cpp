#include "history_logger.hpp"


#include <fstream>



void HistoryLogger::save(
    const Metrics& metrics
)
{

    std::ofstream file(
        "history/data.jsonl",
        std::ios::app
    );



    file

    <<"{"

    <<"\"request_id\":\""
    <<metrics.request_id
    <<"\","


    <<"\"model\":\""
    <<metrics.model
    <<"\","


    <<"\"type\":\""
    <<metrics.type
    <<"\","


    <<"\"latency\":"
    <<metrics.latency
    <<","


    <<"\"score\":"
    <<metrics.routing_score
    <<","


    <<"\"cpu\":"
    <<metrics.cpu_usage
    <<","


    <<"\"memory\":"
    <<metrics.memory_usage


    <<"}\n";

}
