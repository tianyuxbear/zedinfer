#pragma once

#include "interface.hpp"
#include "neollm.h"
#include <memory>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#endif

namespace neollm::loader {

// Represents a single .safetensors file using memory mapping
class SafeTensorFile : public IModelFile {
private:
    std::string filepath;
    void *mmap_data;
    size_t file_size;
    size_t data_offset;
    std::unordered_map<std::string, TensorInfo> tensors;

#ifdef _WIN32
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int fd;
#endif

    neollmDataType_t parse_dtype(const std::string &dtype_str);
    void load_metadata(); // Parses header and populates `tensors`
    void open_mmap();     // Maps file into memory
    void close_mmap();    // Unmaps file and releases resources

public:
    explicit SafeTensorFile(const std::string &path);
    ~SafeTensorFile() override;

    // Non-copyable
    SafeTensorFile(const SafeTensorFile &) = delete;
    SafeTensorFile &operator=(const SafeTensorFile &) = delete;

    // Movable
    SafeTensorFile(SafeTensorFile &&other) noexcept;
    SafeTensorFile &operator=(SafeTensorFile &&other) noexcept;

    // IModelFile interface
    const void *get_tensor_data(const std::string &name) const override;
    const TensorInfo *get_tensor_info(const std::string &name) const override;
    std::vector<std::string> get_tensor_names() const override;
    bool has_tensor(const std::string &name) const override;
};

// Loader for models split across one or more .safetensors files
class SafeTensorsLoader : public IModelLoader {
private:
    std::vector<std::unique_ptr<IModelFile>> files;
    std::unordered_map<std::string, size_t> tensor_to_file; // Maps tensor name → file index

public:
    // Factory method to create a loader instance
    static std::unique_ptr<IModelLoader> create(const std::string &model_path);

    explicit SafeTensorsLoader(const std::string &model_path);

    // IModelLoader interface
    void load(const std::string &model_path) override;
    const void *get_tensor_data(const std::string &name) const override;
    const TensorInfo *get_tensor_info(const std::string &name) const override;
    std::vector<std::string> get_all_tensor_names() const override;
    size_t get_num_files() const override;
};

} // namespace neollm::loader