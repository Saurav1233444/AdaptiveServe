#include <iostream>

#include <httplib.h>

#include <nlohmann/json.hpp>


using json=nlohmann::json;



int main()
{

    httplib::Server server;



    server.Post(
        "/predict",

        [](const httplib::Request& req,
           httplib::Response& res)
        {


            json result;


            result["model"]
                ="MobileNetV3";


            result["prediction"]
                ="cat";


            result["confidence"]
                =0.91;



            res.set_content(
                result.dump(4),
                "application/json"
            );

        }

    );



    std::cout
    <<"MobileNet worker running :9001\n";



    server.listen(
        "127.0.0.1",
        9001
    );


    return 0;
}
