#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Appenders/IAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace zedinfer::utils {

// Logger IDs. FILE_ONLY is the plog default instance kept for compatibility;
// the active sinks are controlled by LogConfig.
enum { FILE_ONLY = 0, CONSOLE_ONLY = 1, BOTH = 2 };

struct LogConfig {
    plog::Severity level = plog::info;
    std::string file_path = "logs/zedinfer.log";
    bool to_console = true;
    bool append = false;
};

inline std::string logSeverityToString(plog::Severity severity) {
    switch (severity) {
        case plog::fatal:
            return "fatal";
        case plog::error:
            return "error";
        case plog::warning:
            return "warning";
        case plog::info:
            return "info";
        case plog::debug:
            return "debug";
        case plog::verbose:
            return "verbose";
        case plog::none:
        default:
            return "none";
    }
}

inline plog::Severity parseLogSeverity(std::string level) {
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (level == "none" || level == "off") {
        return plog::none;
    }
    if (level == "fatal") {
        return plog::fatal;
    }
    if (level == "error" || level == "err") {
        return plog::error;
    }
    if (level == "warning" || level == "warn") {
        return plog::warning;
    }
    if (level == "info") {
        return plog::info;
    }
    if (level == "debug") {
        return plog::debug;
    }
    if (level == "verbose" || level == "verb" || level == "trace") {
        return plog::verbose;
    }

    throw std::invalid_argument("invalid --log-level: " + level
                                + " (expected none, fatal, error, warning, info, debug, verbose)");
}

inline long currentProcessId() {
#if defined(_WIN32)
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}

class ZedLogFormatter {
public:
    static plog::util::nstring header() { return plog::util::nstring(); }

    static plog::util::nstring format(const plog::Record& record) { return format(record, false); }

    static plog::util::nstring format(const plog::Record& record, bool color) {
        tm t;
        plog::util::localtime_s(&t, &record.getTime().time);

        const char* blue = color ? "\x1B[94m" : "";
        const char* gray = color ? "\x1B[90m" : "";
        const char* reset = color ? "\x1B[0m" : "";

        plog::util::nostringstream ss;
        ss << blue << "pid=" << currentProcessId() << reset << " ";
        ss << levelColor(record.getSeverity(), color) << std::setfill(PLOG_NSTR(' ')) << std::setw(5) << std::left
           << plog::severityToString(record.getSeverity()) << reset << " ";
        ss << gray << std::right << t.tm_year + 1900 << "-" << std::setfill(PLOG_NSTR('0')) << std::setw(2)
           << t.tm_mon + 1 << PLOG_NSTR("-") << std::setw(2) << t.tm_mday << PLOG_NSTR(" ") << std::setw(2) << t.tm_hour
           << PLOG_NSTR(":") << std::setw(2) << t.tm_min << PLOG_NSTR(":") << std::setw(2) << t.tm_sec << PLOG_NSTR(".")
           << std::setw(3) << static_cast<int>(record.getTime().millitm) << reset << " ";
        ss << gray << "[tid=" << record.getTid() << " " << record.getFunc() << "@" << record.getLine() << "] " << reset;
        ss << record.getMessage() << reset << PLOG_NSTR("\n");
        return ss.str();
    }

private:
    static const char* levelColor(plog::Severity severity, bool color) {
        if (!color) {
            return "";
        }

        switch (severity) {
            case plog::fatal:
                return "\x1B[97m\x1B[41m";
            case plog::error:
                return "\x1B[91m";
            case plog::warning:
                return "\x1B[93m";
            case plog::info:
                return "\x1B[92m";
            case plog::debug:
                return "\x1B[96m";
            case plog::verbose:
                return "\x1B[90m";
            default:
                return "";
        }
    }
};

class SeverityColorConsoleAppender final : public plog::ConsoleAppender<ZedLogFormatter> {
public:
    explicit SeverityColorConsoleAppender(plog::OutputStream out_stream)
        : plog::ConsoleAppender<ZedLogFormatter>(out_stream) {}

    void write(const plog::Record& record) override {
        const plog::util::nstring str = ZedLogFormatter::format(record, this->m_isatty);
        plog::util::MutexLock lock(this->m_mutex);

        this->writestr(str);
        if (this->m_isatty) {
            this->m_outputStream << "\x1B[0K";
        }
    }
};

class ConfigurableLogAppender final : public plog::IAppender {
public:
    ConfigurableLogAppender()
        : file_appender_("logs/zedinfer.log"),
          stdout_appender_(plog::streamStdOut),
          stderr_appender_(plog::streamStdErr) {}

    void configure(const LogConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);

        console_enabled_ = config.to_console;
        file_enabled_ = !config.file_path.empty();

        if (file_enabled_) {
            std::filesystem::path log_path(config.file_path);
            std::filesystem::path log_dir = log_path.parent_path();
            if (!log_dir.empty()) {
                std::filesystem::create_directories(log_dir);
            }

            if (!config.append && std::filesystem::exists(config.file_path)) {
                std::filesystem::remove(config.file_path);
            }

            file_appender_.setFileName(config.file_path.c_str());
        }
    }

    void write(const plog::Record& record) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (file_enabled_) {
            file_appender_.write(record);
        }

        if (console_enabled_) {
            if (record.getSeverity() <= plog::warning) {
                stderr_appender_.write(record);
            } else {
                stdout_appender_.write(record);
            }
        }
    }

private:
    std::mutex mutex_;
    bool file_enabled_ = true;
    bool console_enabled_ = true;
    plog::RollingFileAppender<ZedLogFormatter> file_appender_;
    SeverityColorConsoleAppender stdout_appender_;
    SeverityColorConsoleAppender stderr_appender_;
};

inline ConfigurableLogAppender& logAppender() {
    static ConfigurableLogAppender appender;
    return appender;
}

inline bool initLogger(const LogConfig& config) {
    auto& appender = logAppender();
    appender.configure(config);

    static bool initialized = false;
    if (!initialized) {
        plog::init<FILE_ONLY>(config.level, &appender);
        plog::init<BOTH>(config.level, &appender);
        initialized = true;
    } else {
        plog::get<FILE_ONLY>()->setMaxSeverity(config.level);
        plog::get<BOTH>()->setMaxSeverity(config.level);
    }

    if (config.file_path.empty()) {
        PLOG_INFO << "Log file disabled";
    } else {
        PLOG_INFO << "Log file: " << std::filesystem::absolute(config.file_path);
    }
    PLOG_INFO << "Log level: " << logSeverityToString(config.level)
              << ", console: " << (config.to_console ? "on" : "off")
              << ", mode: " << (config.append ? "append" : "overwrite");
    return true;
}

inline bool initLoggerWithOverwrite(plog::Severity level, const std::string& filepath) {
    LogConfig config;
    config.level = level;
    config.file_path = filepath;
    config.to_console = true;
    config.append = false;
    return initLogger(config);
}

} // namespace zedinfer::utils
