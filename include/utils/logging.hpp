#pragma once

#include <filesystem>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

namespace neollm::utils {

// Define logger IDs
enum {
    FILE_ONLY = 0,    // Default logger: File only logger
    CONSOLE_ONLY = 1, // Console only logger
    BOTH = 2          // Both File and Console logger
};

inline bool initLoggerWithOverwrite(plog::Severity level, const std::string &filepath) {
    // Extract directory path
    std::filesystem::path logPath(filepath);
    std::filesystem::path logDir = logPath.parent_path();

    if (!logDir.empty()) {
        std::filesystem::create_directories(logDir);
    }

    // Remove existing file for overwrite mode
    if (std::filesystem::exists(filepath)) {
        std::filesystem::remove(filepath);
    }

    // Initialize logger
    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender(filepath.c_str());
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;

    plog::init<FILE_ONLY>(level, &fileAppender);
    plog::init<BOTH>(level, &fileAppender).addAppender(&consoleAppender);

    PLOG_INFO << "Log file: " << std::filesystem::absolute(filepath);
    return true;
}

} // namespace neollm::utils