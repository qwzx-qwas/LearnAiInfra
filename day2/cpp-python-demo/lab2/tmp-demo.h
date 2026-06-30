#include <cstddef>
#include <memory>

template <typename T>
class Buffer {
public:
    explicit Buffer(std::size_t size):
        size_(size),
        data_(std::make_unique<T[]>(size)) {}
    
    T* data() {
            return data_.get();
        }

    std::size_t size() const {
            return size_;
        }

private:
    std::unique<T[]>(size) 
};
