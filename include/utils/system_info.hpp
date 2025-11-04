#pragma once

#include <string>

namespace neollm::utils {

// Get the number of physical CPU cores
int get_physical_cores();

// Get runtime environment information (CPU ISA & OpenMP)
std::string get_runtime_info();

// Dump NUMA memory maps from /proc/self/numa_maps
// @param max_lines Maximum number of anonymous memory regions to include
// @return Formatted string containing NUMA mapping information
std::string get_numa_maps_info(int max_lines = 20);

} // namespace neollm::utils