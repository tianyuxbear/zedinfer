#pragma once

#include "neollm.h"
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace neollm::loader {

struct TensorInfo {
    std::string name;
    NeollmDataType_t dtype;
    std::vector<size_t> shape;
    size_t data_offset; // Offset of tensor data within the file
    size_t num_bytes;   // Size of tensor data in bytes

    size_t numel() const {
        return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>{});
    }
};

// Abstract interface for a model file
class IModelFile {
public:
    virtual ~IModelFile() = default;

    // Returns a pointer to the tensor's raw data (e.g., via mmap)
    virtual const void *get_tensor_data(const std::string &name) const = 0;

    // Returns metadata for the given tensor
    virtual const TensorInfo *get_tensor_info(const std::string &name) const = 0;

    // Returns names of all tensors in the file
    virtual std::vector<std::string> get_tensor_names() const = 0;

    // Checks if the file contains a tensor with the given name
    virtual bool has_tensor(const std::string &name) const = 0;
};

// Abstract factory for model loaders
class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    // Loads the model from the specified path
    virtual void load(const std::string &model_path) = 0;

    // Retrieves raw data for the specified tensor
    virtual const void *get_tensor_data(const std::string &name) const = 0;

    // Retrieves metadata for the specified tensor
    virtual const TensorInfo *get_tensor_info(const std::string &name) const = 0;

    // Returns names of all tensors in the loaded model
    virtual std::vector<std::string> get_all_tensor_names() const = 0;

    // Returns the number of files loaded
    virtual size_t get_num_files() const = 0;
};

} // namespace neollm::loader