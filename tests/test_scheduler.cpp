#include "core/scheduler/bounded_scheduler.hpp"

#include <iostream>
#include <stdexcept>

int main() {
  adaptiveserve::BoundedScheduler scheduler(1);
  auto first = scheduler.try_acquire();
  if (!first || scheduler.in_flight() != 1)
    throw std::runtime_error("first request must be admitted");
  auto overloaded = scheduler.try_acquire();
  if (overloaded)
    throw std::runtime_error("request above capacity must be rejected");
  first.reset();
  auto next = scheduler.try_acquire();
  if (!next || scheduler.in_flight() != 1)
    throw std::runtime_error("released capacity must be reusable");
  bool rejected = false;
  try {
    adaptiveserve::BoundedScheduler invalid(0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("zero capacity must be rejected");
  std::cout << "scheduler tests passed\n";
}
