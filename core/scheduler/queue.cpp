#include "queue.hpp"



void RequestQueue::push(
    const Task& task
)
{
    tasks.push(task);
}




Task RequestQueue::pop()
{

    Task task =
    tasks.front();


    tasks.pop();


    return task;

}





bool RequestQueue::empty() const
{
    return tasks.empty();
}




int RequestQueue::size() const
{
    return tasks.size();
}
