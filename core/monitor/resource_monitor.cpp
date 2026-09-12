#include "resource_monitor.hpp"


#include <fstream>

#include <string>

#include <sstream>



double ResourceMonitor::memory()
{

    std::ifstream file(
        "/proc/meminfo"
    );


    std::string line;



    while(std::getline(file,line))
    {

        if(line.find("MemAvailable")
            !=std::string::npos)
        {

            std::stringstream ss(line);


            std::string key;


            double value;



            ss>>key>>value;



            return value/1024.0;

        }

    }


    return 0;

}





double ResourceMonitor::cpu()
{

    std::ifstream file(
        "/proc/loadavg"
    );


    double load=0;


    file>>load;


    return load;

}
