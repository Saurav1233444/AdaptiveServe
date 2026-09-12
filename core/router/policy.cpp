#include "policy.hpp"


RouterPolicy::RouterPolicy()
{
    accuracy_weight=0.7;

    latency_weight=0.2;

    memory_weight=0.1;
}
