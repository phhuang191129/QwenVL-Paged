#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace qwenvl_bench {

inline std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream out;
    out << in.rdbuf();
    std::string text = out.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

struct EnvSnapshot {
    std::string governor;
    std::uint64_t freq_khz{0};
    std::int64_t thermal_mC{0};
    std::string model_name;
};

inline EnvSnapshot capture_env() {
    EnvSnapshot env;
    env.governor = read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    const std::string freq = read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    if (!freq.empty()) {
        env.freq_khz = static_cast<std::uint64_t>(std::stoull(freq));
    }
    const std::string temp = read_file("/sys/class/thermal/thermal_zone0/temp");
    if (!temp.empty()) {
        env.thermal_mC = static_cast<std::int64_t>(std::stoll(temp));
    }
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                env.model_name = line.substr(colon + 1);
                while (!env.model_name.empty() && env.model_name.front() == ' ') {
                    env.model_name.erase(env.model_name.begin());
                }
            }
            break;
        }
    }
    return env;
}

inline bool pin_to_cpu0() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    return false;
#endif
}

inline void warmup_clock() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    volatile std::uint64_t sink = 0;
    while (std::chrono::duration<double, std::milli>(Clock::now() - start).count() < 500.0) {
        for (std::uint64_t i = 0; i < 1000000; ++i) {
            sink += i;
        }
    }
    asm volatile("" : : "g"(sink) : "memory");
}

/**
 * Reject a run when the clock *drops* by more than `max_rel` from the start
 * frequency. A ramp-up from idle/powersave is expected and is not a reject.
 */
inline bool freq_stable(const EnvSnapshot& before, const EnvSnapshot& after, double max_rel) {
    if (before.freq_khz == 0 || after.freq_khz == 0) {
        return true;
    }
    if (after.freq_khz >= before.freq_khz) {
        return true;
    }
    const double drop = static_cast<double>(before.freq_khz - after.freq_khz) /
        static_cast<double>(before.freq_khz);
    return drop <= max_rel;
}

} // namespace qwenvl_bench
