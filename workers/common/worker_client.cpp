#include "worker_client.hpp"

#include <httplib.h>


std::string WorkerClient::send(
    std::string host,
    int port,
    std::string data
)
{

    httplib::Client client(
        host,
        port
    );


    auto response =
    client.Post(
        "/predict",
        data,
        "application/json"
    );


    if(!response)
        return "{\"error\":\"worker unavailable\"}";


    return response->body;
}
