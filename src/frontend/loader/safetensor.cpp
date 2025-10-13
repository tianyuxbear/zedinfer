#include "frontend/loader/safetensors.hpp"
#include "neollm.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace neollm::loader {

// ============================================
// SafeTensorFile Implementation
// ============================================

std::unique_ptr<IModelLoader> SafeTensorsLoader::create(const std::string &model_path) {
    return std::make_unique<SafeTensorsLoader>(model_path);
}

SafeTensorFile::SafeTensorFile(const std::string &path)
    : filepath(path), mmap_data(nullptr), file_size(0), data_offset(0) {
#ifdef _WIN32
    file_handle = INVALID_HANDLE_VALUE;
    mapping_handle = NULL;
#else
    fd = -1;
#endif
    load_metadata();
    open_mmap();
}

SafeTensorFile::~SafeTensorFile() {
    close_mmap();
}

SafeTensorFile::SafeTensorFile(SafeTensorFile &&other) noexcept
    : filepath(std::move(other.filepath)),
      mmap_data(other.mmap_data),
      file_size(other.file_size),
      data_offset(other.data_offset),
      tensors(std::move(other.tensors)) {
#ifdef _WIN32
    file_handle = other.file_handle;
    mapping_handle = other.mapping_handle;
    other.file_handle = INVALID_HANDLE_VALUE;
    other.mapping_handle = NULL;
#else
    fd = other.fd;
    other.fd = -1;
#endif
    other.mmap_data = nullptr;
}

SafeTensorFile &SafeTensorFile::operator=(SafeTensorFile &&other) noexcept {
    if (this != &other) {
        close_mmap();

        filepath = std::move(other.filepath);
        mmap_data = other.mmap_data;
        file_size = other.file_size;
        data_offset = other.data_offset;
        tensors = std::move(other.tensors);

#ifdef _WIN32
        file_handle = other.file_handle;
        mapping_handle = other.mapping_handle;
        other.file_handle = INVALID_HANDLE_VALUE;
        other.mapping_handle = NULL;
#else
        fd = other.fd;
        other.fd = -1;
#endif
        other.mmap_data = nullptr;
    }
    return *this;
}

NeollmDataType_t SafeTensorFile::parse_dtype(const std::string &dtype_str) {
    static const std::unordered_map<std::string, NeollmDataType_t> dtype_map = {
        {"BYTE", NEOLLM_DTYPE_BYTE},
        {"BOOL", NEOLLM_DTYPE_BOOL},
        {"I8", NEOLLM_DTYPE_I8},
        {"I16", NEOLLM_DTYPE_I16},
        {"I32", NEOLLM_DTYPE_I32},
        {"I64", NEOLLM_DTYPE_I64},
        {"U8", NEOLLM_DTYPE_U8},
        {"U16", NEOLLM_DTYPE_U16},
        {"U32", NEOLLM_DTYPE_U32},
        {"U64", NEOLLM_DTYPE_U64},
        {"F16", NEOLLM_DTYPE_F16},
        {"F32", NEOLLM_DTYPE_F32},
        {"F64", NEOLLM_DTYPE_F64},
        {"BF16", NEOLLM_DTYPE_BF16}};

    auto it = dtype_map.find(dtype_str);
    if (it != dtype_map.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown dtype: " + dtype_str);
}

void SafeTensorFile::load_metadata() {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    // Read 8-byte little-endian header size
    uint64_t header_size;
    file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    if (!file) {
        throw std::runtime_error("Failed to read header size from: " + filepath);
    }

    // Read header JSON
    std::vector<char> header_bytes(header_size);
    file.read(header_bytes.data(), header_size);
    if (!file) {
        throw std::runtime_error("Failed to read header from: " + filepath);
    }

    std::string header_str(header_bytes.begin(), header_bytes.end());
    json header = json::parse(header_str);

    data_offset = 8 + header_size; // Header + size prefix

    // Parse tensor metadata
    for (auto &[key, value] : header.items()) {
        if (key == "__metadata__") {
            continue;
        }

        TensorInfo info;
        info.name = key;
        info.dtype = parse_dtype(value["dtype"].get<std::string>());
        info.shape = value["shape"].get<std::vector<size_t>>();

        auto data_offsets = value["data_offsets"].get<std::vector<size_t>>();
        info.data_offset = data_offsets[0];
        info.num_bytes = data_offsets[1] - data_offsets[0];

        tensors.emplace(std::move(key), std::move(info));
    }
}

void SafeTensorFile::open_mmap() {
#ifdef _WIN32
    file_handle = CreateFileA(
        filepath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file for mmap: " + filepath);
    }

    LARGE_INTEGER file_size_li;
    if (!GetFileSizeEx(file_handle, &file_size_li)) {
        CloseHandle(file_handle);
        throw std::runtime_error("Failed to get file size: " + filepath);
    }
    file_size = static_cast<size_t>(file_size_li.QuadPart);

    mapping_handle = CreateFileMappingA(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping_handle == NULL) {
        CloseHandle(file_handle);
        throw std::runtime_error("Failed to create file mapping: " + filepath);
    }

    mmap_data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (mmap_data == NULL) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        throw std::runtime_error("Failed to map view of file: " + filepath);
    }
#else
    fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file for mmap: " + filepath);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to stat file: " + filepath);
    }
    file_size = sb.st_size;

    mmap_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + filepath);
    }
#endif
}

void SafeTensorFile::close_mmap() {
    if (!mmap_data) {
        return;
    }

#ifdef _WIN32
    UnmapViewOfFile(mmap_data);
    if (mapping_handle != NULL) {
        CloseHandle(mapping_handle);
    }
    if (file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle);
    }
#else
    munmap(mmap_data, file_size);
    if (fd != -1) {
        close(fd);
    }
#endif

    mmap_data = nullptr;
}

const void *SafeTensorFile::get_tensor_data(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        return nullptr;
    }

    const TensorInfo &info = it->second;
    // data_offset is the start of tensor data section; info.data_offset is relative to that
    return static_cast<const char *>(mmap_data) + data_offset + info.data_offset;
}

const TensorInfo *SafeTensorFile::get_tensor_info(const std::string &name) const {
    auto it = tensors.find(name);
    return (it != tensors.end()) ? &it->second : nullptr;
}

std::vector<std::string> SafeTensorFile::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors.size());
    for (const auto &[name, _] : tensors) {
        names.push_back(name);
    }
    return names;
}

bool SafeTensorFile::has_tensor(const std::string &name) const {
    return tensors.count(name) > 0;
}

// ============================================
// SafeTensorsLoader Implementation
// ============================================

SafeTensorsLoader::SafeTensorsLoader(const std::string &model_path) {
    load(model_path);
}

void SafeTensorsLoader::load(const std::string &model_path) {
    if (!fs::exists(model_path) || !fs::is_directory(model_path)) {
        throw std::runtime_error("Model path does not exist or is not a directory: " + model_path);
    }

    // Collect all .safetensors files
    std::vector<std::string> safetensor_files;
    for (const auto &entry : fs::directory_iterator(model_path)) {
        if (entry.is_regular_file()) {
            auto path = entry.path();
            if (path.extension() == ".safetensors") {
                safetensor_files.push_back(path.string());
            }
        }
    }

    if (safetensor_files.empty()) {
        throw std::runtime_error("No .safetensors files found in: " + model_path);
    }

    // Sort to ensure consistent loading order (e.g., for sharded models)
    std::sort(safetensor_files.begin(), safetensor_files.end());

    files.reserve(safetensor_files.size());
    for (const auto &filepath : safetensor_files) {
        auto file = std::make_unique<SafeTensorFile>(filepath);
        for (const auto &name : file->get_tensor_names()) {
            tensor_to_file[name] = files.size();
        }
        files.push_back(std::move(file));
    }
}

const void *SafeTensorsLoader::get_tensor_data(const std::string &name) const {
    auto it = tensor_to_file.find(name);
    if (it == tensor_to_file.end()) {
        return nullptr;
    }
    return files[it->second]->get_tensor_data(name);
}

const TensorInfo *SafeTensorsLoader::get_tensor_info(const std::string &name) const {
    auto it = tensor_to_file.find(name);
    if (it == tensor_to_file.end()) {
        return nullptr;
    }
    return files[it->second]->get_tensor_info(name);
}

std::vector<std::string> SafeTensorsLoader::get_all_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_to_file.size());
    for (const auto &[name, _] : tensor_to_file) {
        names.push_back(name);
    }
    return names;
}

size_t SafeTensorsLoader::get_num_files() const {
    return files.size();
}

} // namespace neollm::loader