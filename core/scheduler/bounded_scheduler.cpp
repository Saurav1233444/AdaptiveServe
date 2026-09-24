#include "bounded_scheduler.hpp"
#include <stdexcept>
namespace adaptiveserve {
BoundedScheduler::BoundedScheduler(std::size_t capacity) : capacity_(capacity) {
  if (capacity == 0)
    throw std::invalid_argument("scheduler capacity must be positive");
}
std::unique_ptr<BoundedScheduler::Lease> BoundedScheduler::try_acquire() {
  auto current = in_flight_.load(std::memory_order_relaxed);
  while (current < capacity_) {
    if (in_flight_.compare_exchange_weak(current, current + 1,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed))
      return std::make_unique<Lease>(*this);
  }
  return nullptr;
}
std::size_t BoundedScheduler::capacity() const noexcept { return capacity_; }
std::size_t BoundedScheduler::in_flight() const noexcept {
  return in_flight_.load(std::memory_order_relaxed);
}
void BoundedScheduler::release() noexcept {
  in_flight_.fetch_sub(1, std::memory_order_release);
}
} // namespace adaptiveserve
