#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
namespace adaptiveserve {
class BoundedScheduler {
public:
  class Lease;
  explicit BoundedScheduler(std::size_t capacity);
  [[nodiscard]] std::unique_ptr<Lease> try_acquire();
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t in_flight() const noexcept;

private:
  friend class Lease;
  void release() noexcept;
  const std::size_t capacity_;
  std::atomic<std::size_t> in_flight_{0};
};
class BoundedScheduler::Lease {
public:
  explicit Lease(BoundedScheduler &owner) noexcept : owner_(&owner) {}
  ~Lease() {
    if (owner_)
      owner_->release();
  }
  Lease(const Lease &) = delete;
  Lease &operator=(const Lease &) = delete;

private:
  BoundedScheduler *owner_;
};
} // namespace adaptiveserve
