#include "server.hpp"


#include <iostream>

#include <nlohmann/json.hpp>


#include "../monitor/latency.hpp"

#include "../monitor/resource_monitor.hpp"

#include "../monitor/history_logger.hpp"



using json=nlohmann::json;



GatewayServer::GatewayServer()

:
router(&registry),
executor(&registry)
{

    registry.load(
        "configs/models.json"
    );

}





void GatewayServer::start(
    int port
)
{

    httplib::Server server;



    server.Post(
        "/predict",

        [&](const httplib::Request& req,
            httplib::Response& res)

        {


            Timer timer;


            timer.start();



            auto body =
            json::parse(
                req.body
            );



            std::string type =
            body.value(
                "type",
                "vision"
            );



            ModelInfo model =
            router.select(
                type
            );



            Task task;


            task.id="1";

            task.model=model.name;

            task.payload="input";



            scheduler.submit(task);
            std::string result;
            if(scheduler.hasTask()){
                Task current = scheduler.next();
                result = executor.execute(current);

            }



            ResourceMonitor monitor;



            double latency =
            timer.stop();



            Metrics metrics;


            metrics.request_id="1";

            metrics.model=model.name;

            metrics.type=type;

            metrics.latency=latency;

            metrics.routing_score=0;

            metrics.cpu_usage=
            monitor.cpu();


            metrics.memory_usage=
            monitor.memory();



            HistoryLogger logger;


            logger.save(
                metrics
            );



            json response;


            response["selected_model"]
            =
            model.name;



            response["host"]
            =
            model.host;



            response["port"]
            =
            model.port;



            response["latency"]
            =
            latency;

            response["worker_response"] = json::parse(result);

            res.set_content(
                response.dump(4),
                "application/json"
            );

        }

    );



    std::cout
    <<"Gateway running on port "
    <<port
    <<std::endl;



    server.listen(
        "127.0.0.1",
        port
    );

}
