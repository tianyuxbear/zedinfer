#pragma once

#define ZEDINFER_VERSION "0.1.0"

// Injected by xmake at build time via on_config (see xmake.lua).
// Falls back to "unknown" when git is unavailable (e.g., Docker build without .git/).
#ifndef ZEDINFER_GIT_HASH
#define ZEDINFER_GIT_HASH "unknown"
#endif

#ifndef ZEDINFER_BUILD_DATE
#define ZEDINFER_BUILD_DATE "unknown"
#endif
