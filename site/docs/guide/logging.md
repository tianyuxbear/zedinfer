# Logging Guide

This document defines the runtime logging contract for ZedInfer command-line tools and server processes.

---

## CLI Options

Every CLI entry point exposes the same logging controls:

| Option | Default | Description |
|--------|---------|-------------|
| `--log-level` | `info` | Minimum log level. Accepted values: `none`, `fatal`, `error`, `warning`, `info`, `debug`, `verbose`. Aliases: `off`, `err`, `warn`, `verb`, `trace`. |
| `--log-file` | Tool-specific path under `logs/` | File sink path. Pass an empty value to disable file logging. |
| `--log-to-console` | Enabled except for `chat` | Mirror logs to stdout/stderr. |
| `--no-log-to-console` | Disabled | Disable console mirroring even when `--log-to-console` would otherwise be enabled. |
| `--log-append` | Disabled | Append to `--log-file`. |
| `--log-overwrite` | Default behavior | Replace `--log-file` on startup. Mutually exclusive with `--log-append`. |

Default log files:

| Tool | Default log file | Console logs by default |
|------|------------------|-------------------------|
| `serve` | `logs/serve.log` | Yes |
| `chat` | `logs/chat.log` | No |
| `ping` | `logs/ping.log` | Yes |
| `bench` | `logs/bench.log` | Yes |
| `batch_bench` | `logs/batch_bench.log` | Yes |
| `ppl` | `logs/ppl.log` | Yes |

Examples:

```bash
xmake run serve /path/to/model --nvidia --log-level debug
xmake run serve /path/to/model --nvidia --log-file /var/log/zedinfer/serve.log --log-append
xmake run chat /path/to/model --nvidia --log-to-console
xmake run bench /path/to/model --nvidia --log-file "" --no-log-to-console
```

---

## Console Routing

Console logs are split by severity so container runtimes and shell users can consume them naturally:

| Severity | Stream |
|----------|--------|
| `fatal` | `stderr` |
| `error` | `stderr` |
| `warning` | `stderr` |
| `info` | `stdout` |
| `debug` | `stdout` |
| `verbose` | `stdout` |

This means `docker logs` sees all server logs through the standard output streams. When the output stream is not a TTY, ANSI colors are not emitted.

---

## Log Format

File and console logs use the same field order:

```text
pid=<pid> <LEVEL> <YYYY-MM-DD HH:MM:SS.mmm> [tid=<tid> <function>@<line>] <message>
```

When the console supports color:

- `pid` is blue.
- The level is colored by severity.
- Timestamp and source metadata are gray.
- Message text uses the normal terminal foreground.

Files never contain ANSI color escapes.

---

## Startup Banner

CLI tools print the shared ZedInfer banner during startup after logging is initialized and before model/server work begins.

The banner is intentionally written to stdout instead of the logger. It is a startup marker, not a log record. When stdout is a TTY, the `Zed` part is white and the `Infer` part is blue. A blank line is printed before the banner.

The previous CPU instruction set and thread runtime dump is not printed by default.

---

## Interactive Chat

`chat` is interactive, so console logs are disabled by default. This prevents log lines from interleaving with prompts, user input, streamed model output, and in-session command responses.

`chat` still writes to `logs/chat.log` by default. Use `--log-to-console` only when debugging and when interleaving with the terminal UI is acceptable.

---

## HTTP Access Logs

`serve` logs every HTTP request through the logger, including static assets and API endpoints.

Access log fields:

- remote address and port
- local address and port
- HTTP method
- target path/query
- HTTP version
- response status
- response body byte count
- request `Content-Length`
- request `User-Agent`

Inference request/response logs are separate from HTTP access logs and should keep request-level fields such as request id, session id, token counts, and finish reason.

---

## Developer Rules

Use the logger for runtime diagnostics, state transitions, warnings, errors, and debug-only dumps. Do not write those directly with `std::cout`, `std::cerr`, `printf`, `fprintf(stderr, ...)`, `puts`, or `perror`.

Direct stdout/stderr is still appropriate for user-facing output:

- argparse parse errors and usage text
- benchmark and PPL final reports
- `ping` generated text
- `chat` prompts, command responses, and streamed model output
- explicit debug APIs such as `Tensor::debug()`
- terminal progress UI
- low-level fatal paths that cannot safely depend on logger initialization, such as CUDA utility macros

New CLI tools should:

1. Include `utils/logging_cli.hpp`.
2. Call `utils::addLoggingArguments()` before parsing arguments.
3. Call `utils::initLoggerFromArguments()` after parsing succeeds.
4. Call `utils::printZedInferBanner()` when the tool should show the shared startup banner.
5. Use `default_to_console = false` only for interactive terminal tools where logs would corrupt user interaction.
