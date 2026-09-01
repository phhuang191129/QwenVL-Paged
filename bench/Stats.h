#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace qwenvl_bench {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::duration<double, std::nano>;

struct SampleStats {
    std::size_t n{0};
    double median_ns{0.0};
    double p95_ns{0.0};
    double mad_ns{0.0};
};

inline double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(index);
    const std::size_t hi = std::min(lo + 1, values.size() - 1);
    const double weight = index - static_cast<double>(lo);
    return values[lo] * (1.0 - weight) + values[hi] * weight;
}

inline SampleStats summarize(std::vector<double> samples) {
    SampleStats stats;
    stats.n = samples.size();
    if (samples.empty()) {
        return stats;
    }
    stats.median_ns = percentile(samples, 0.50);
    stats.p95_ns = percentile(samples, 0.95);

    std::vector<double> deviations;
    deviations.reserve(samples.size());
    for (double sample : samples) {
        deviations.push_back(std::abs(sample - stats.median_ns));
    }
    stats.mad_ns = percentile(std::move(deviations), 0.50);
    return stats;
}

/**
 * Warm up `warmup` times, then time `repeats` calls of `body`.
 * Each call of `body` is one sample.
 */
template <typename Body>
SampleStats measure_body(Body&& body, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        body();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const Clock::time_point start = Clock::now();
        body();
        samples.push_back(Nanoseconds(Clock::now() - start).count());
    }
    return summarize(std::move(samples));
}

inline std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

} // namespace qwenvl_bench
