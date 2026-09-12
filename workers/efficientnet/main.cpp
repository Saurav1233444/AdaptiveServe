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
                ="EfficientNet";


            result["prediction"]
                ="complex_object";


            result["confidence"]
                =0.97;



            res.set_content(
                result.dump(4),
                "application/json"
            );

        }

    );



    std::cout
    <<"EfficientNet worker running :9002\n";



    server.listen(
        "127.0.0.1",
        9002
    );


    return 0;
}
