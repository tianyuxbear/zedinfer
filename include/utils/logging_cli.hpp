#pragma once

#include "utils/logging.hpp"

#include <argparse/argparse.hpp>
#include <stdexcept>
#include <string>

namespace zedinfer::utils {

inline void addLoggingArguments(argparse::ArgumentParser& program, const std::string& default_log_file,
                                plog::Severity default_level = plog::info, bool default_to_console = true) {
    program.add_argument("--log-level")
        .help("Log level: none, fatal, error, warning, info, debug, verbose")
        .default_value(logSeverityToString(default_level));

    program.add_argument("--log-file").help("Path to the log file").default_value(default_log_file);

    program.add_argument("--log-to-console")
        .help("Mirror logs to stdout/stderr")
        .default_value(default_to_console)
        .implicit_value(true);

    program.add_argument("--no-log-to-console")
        .help("Disable stdout/stderr log mirroring")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--log-append")
        .help("Append to --log-file instead of replacing it")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--log-overwrite")
        .help("Replace --log-file on startup (default)")
        .default_value(false)
        .implicit_value(true);
}

inline LogConfig logConfigFromArguments(argparse::ArgumentParser& program) {
    const bool append = program.get<bool>("--log-append");
    const bool overwrite = program.get<bool>("--log-overwrite");
    if (append && overwrite) {
        throw std::invalid_argument("--log-append and --log-overwrite are mutually exclusive");
    }

    LogConfig config;
    config.level = parseLogSeverity(program.get<std::string>("--log-level"));
    config.file_path = program.get<std::string>("--log-file");
    config.to_console = program.get<bool>("--log-to-console") && !program.get<bool>("--no-log-to-console");
    config.append = append;
    return config;
}

inline bool initLoggerFromArguments(argparse::ArgumentParser& program) {
    return initLogger(logConfigFromArguments(program));
}

} // namespace zedinfer::utils
