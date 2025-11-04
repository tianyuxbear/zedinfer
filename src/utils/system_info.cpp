#include "utils/system_info.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <cpuid.h>
#endif

namespace neollm::utils {

int get_physical_cores() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::set<std::pair<int, int>> unique_cores; // (physical_id, core_id)

    int phys_id = -1, core_id = -1;
    while (std::getline(cpuinfo, line)) {
        if (line.find("physical id") == 0) {
            phys_id = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("core id") == 0) {
            core_id = std::stoi(line.substr(line.find(":") + 1));
            if (phys_id >= 0 && core_id >= 0) {
                unique_cores.insert({phys_id, core_id});
            }
        }
    }
    return unique_cores.empty() ? std::thread::hardware_concurrency() / 2
                                : unique_cores.size();
#else
    return std::thread::hardware_concurrency() / 2;
#endif
}

std::string get_runtime_info() {
    std::ostringstream oss;
    oss << "\n🚀 =============== Runtime Info ===============\n";

    // Set OpenMP threads to physical core count
    int physical_cores = get_physical_cores();
#ifdef _OPENMP
    omp_set_num_threads(physical_cores);
#endif

    // Detect CPU instruction sets
    oss << "\n📊 CPU Instruction Sets: \n";

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    unsigned int eax, ebx, ecx, edx;

    // Check AVX512 and AVX2 (CPUID leaf 7, subleaf 0)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        bool avx512f = (ebx & bit_AVX512F) != 0;
        bool avx512bw = (ebx & bit_AVX512BW) != 0;
        bool avx512vl = (ebx & bit_AVX512VL) != 0;
        bool avx512vnni = (ecx & bit_AVX512VNNI) != 0;
        bool avx512ifma = (ebx & bit_AVX512IFMA) != 0;
        bool avx2 = (ebx & bit_AVX2) != 0;

        oss << "   AVX512F:    " << (avx512f ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   AVX512BW:   " << (avx512bw ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   AVX512VL:   " << (avx512vl ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   AVX512VNNI: " << (avx512vnni ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   AVX512IFMA: " << (avx512ifma ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   AVX2:       " << (avx2 ? "✅ ON" : "❌ OFF") << std::endl;
    }

    // Check SSE/AVX/FMA/F16C (CPUID leaf 1)
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        bool avx = (ecx & bit_AVX) != 0;
        bool fma = (ecx & bit_FMA) != 0;
        bool f16c = (ecx & bit_F16C) != 0;
        bool sse42 = (ecx & bit_SSE4_2) != 0;

        oss << "   AVX:        " << (avx ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   FMA:        " << (fma ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   F16C:       " << (f16c ? "✅ ON" : "❌ OFF") << std::endl;
        oss << "   SSE4.2:     " << (sse42 ? "✅ ON" : "❌ OFF") << std::endl;
    }
#else
    oss << "   ⚠️  CPU instruction detection not available" << std::endl;
#endif

    // Display threading info
    oss << "\n🧵 Threading Configuration:" << std::endl;
    oss << "   Physical cores:  " << physical_cores << std::endl;
    oss << "   Logical cores:   " << std::thread::hardware_concurrency() << std::endl;

#ifdef _OPENMP
    oss << "   OpenMP:          ✅ ENABLED" << std::endl;
    oss << "   OpenMP threads:  " << omp_get_max_threads() << std::endl;
    oss << "   OpenMP version:  " << _OPENMP << std::endl;
#else
    oss << "   OpenMP:          ❌ DISABLED" << std::endl;
#endif

    oss << "\n================================================\n"
        << std::endl;

    return oss.str();
}

std::string get_numa_maps_info(int max_lines) {
    std::ostringstream oss;
    std::ifstream file("/proc/self/numa_maps");

    if (!file.is_open()) {
        oss << "Failed to open /proc/self/numa_maps\n";
        return oss.str();
    }

    oss << "\n===== NUMA maps (top " << max_lines << " anon regions) =====\n";

    std::string line;
    int count = 0;
    while (std::getline(file, line) && count < max_lines) {
        if (line.find("anon") != std::string::npos) {
            oss << line << '\n';
            ++count;
        }
    }
    return oss.str();
}

} // namespace neollm::utils