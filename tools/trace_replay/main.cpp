#include "TraceReplay.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using qwenvl_paged::replay::ReplayConfig;

void print_usage() {
    std::cerr
        << "Usage: qwenvl_trace_replay [options] <trace.jsonl> [<trace.jsonl> ...]\n\n"
           "Replays Qwen3-VL request traces through the paged KV cache and through a\n"
           "contiguous worst-case-reservation baseline over an identically sized pool.\n\n"
           "Options:\n"
           "  --pool-blocks N          Physical blocks in the pool. Default 0, meaning\n"
           "                           size it per trace so the batch never exhausts\n"
           "                           the pool mid-decode.\n"
           "  --tokens-per-block N     Tokens per block (default 16)\n"
           "  --max-active N           Concurrent requests in a batch (default 16)\n"
           "  --max-batch-tokens N     Prefill token budget per step (default 65536)\n"
           "  --watermark N            Free blocks to keep in reserve; 0 disables\n"
           "                           automatic preemption (default 0)\n"
           "  --swap-slots N           Swap capacity in blocks; 0 disables swap (default 0)\n"
           "  --baseline-max-context N Context the contiguous baseline reserves per\n"
           "                           request (default 8192)\n"
           "  --paged-only             Skip the contiguous baseline\n"
           "  --csv PATH               Append CSV rows to PATH instead of stdout\n";
}

bool parse_u32(const char* text, std::uint32_t& out) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint32_t>(value);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    ReplayConfig config;
    config.max_batch_tokens = 65536;

    std::vector<std::string> traces;
    std::string csv_path;
    bool paged_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg == "--paged-only") {
            paged_only = true;
            continue;
        }
        if (!has_value && arg.rfind("--", 0) == 0) {
            std::cerr << "missing value for " << arg << "\n";
            return 2;
        }

        bool ok = true;
        if (arg == "--pool-blocks") {
            ok = parse_u32(argv[++i], config.max_blocks);
        } else if (arg == "--tokens-per-block") {
            ok = parse_u32(argv[++i], config.tokens_per_block);
        } else if (arg == "--max-active") {
            ok = parse_u32(argv[++i], config.max_active_requests);
        } else if (arg == "--max-batch-tokens") {
            ok = parse_u32(argv[++i], config.max_batch_tokens);
        } else if (arg == "--watermark") {
            ok = parse_u32(argv[++i], config.preemption_watermark_blocks);
        } else if (arg == "--swap-slots") {
            ok = parse_u32(argv[++i], config.swap_slots);
        } else if (arg == "--baseline-max-context") {
            ok = parse_u32(argv[++i], config.baseline_max_context);
        } else if (arg == "--csv") {
            csv_path = argv[++i];
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "unknown option " << arg << "\n";
            return 2;
        } else {
            traces.push_back(arg);
        }

        if (!ok) {
            std::cerr << "invalid value for " << arg << "\n";
            return 2;
        }
    }

    if (traces.empty()) {
        print_usage();
        return 2;
    }

    std::vector<std::string> rows;
    for (const std::string& path : traces) {
        std::string error;
        const auto trace = qwenvl_paged::replay::load_trace(path, error);
        if (!trace.has_value()) {
            std::cerr << error << "\n";
            return 1;
        }

        ReplayConfig run_config = config;
        if (run_config.max_blocks == 0) {
            run_config.max_blocks = qwenvl_paged::replay::suggested_pool_blocks(*trace, run_config);
        }

        std::cerr << path << ": " << trace->requests.size() << " requests, mix "
                  << trace->header.mix << ", budget " << trace->header.pixel_budget
                  << ", max prompt " << trace->header.max_prompt_tokens << " tokens, pool "
                  << run_config.max_blocks << " blocks\n";

        const auto paged = qwenvl_paged::replay::run_paged(*trace, run_config);
        rows.push_back(qwenvl_paged::replay::to_csv_row(*trace, run_config, paged));
        if (!paged_only) {
            const auto contiguous = qwenvl_paged::replay::run_contiguous(*trace, run_config);
            rows.push_back(qwenvl_paged::replay::to_csv_row(*trace, run_config, contiguous));
        }
    }

    if (csv_path.empty()) {
        std::cout << qwenvl_paged::replay::csv_header() << "\n";
        for (const std::string& row : rows) {
            std::cout << row << "\n";
        }
        return 0;
    }

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) {
        std::cerr << "cannot write " << csv_path << "\n";
        return 1;
    }
    if (csv.tellp() == 0) {
        csv << qwenvl_paged::replay::csv_header() << "\n";
    }
    for (const std::string& row : rows) {
        csv << row << "\n";
    }
    return 0;
}
