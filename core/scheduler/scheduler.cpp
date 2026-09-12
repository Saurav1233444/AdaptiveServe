#include "scheduler.hpp"



void Scheduler::submit(
    const Task& task
)
{

    queue.push(task);

}





Task Scheduler::next()
{
    return queue.pop();
}




bool Scheduler::hasTask() const
{
    return !queue.empty();
}




int Scheduler::pending() const
{
    return queue.size();
}
