#pragma once

#include <memory>

namespace zedinfer {
namespace core {

class Storage;
using storage_t = std::shared_ptr<Storage>;

namespace memory {
class MemoryAllocator;
}
using allocator_t = std::unique_ptr<memory::MemoryAllocator>;

class Runtime;
using runtime_t = std::unique_ptr<Runtime>;

class Context;

// Returns the thread-local Context instance.
Context& context();

} // namespace core
} // namespace zedinfer