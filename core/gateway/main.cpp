#include "server.hpp"


int main()
{

    GatewayServer server;


    server.start(
        8000
    );


    return 0;
}
