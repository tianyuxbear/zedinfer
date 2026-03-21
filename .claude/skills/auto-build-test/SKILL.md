---
name: auto-build-test
description: After modifying code, automatically run build verification commands, write output to logs/build.log, analyze failures, fix issues, retry until build passes, and document root cause and fix method in docs/debug/current.md.
---

# Auto Build Test

## Purpose
Use this skill after changing code that may affect build or test compilation.

This skill provides a closed-loop workflow:
1. clean and rebuild
2. capture output into `logs/build.log`
3. inspect errors if build fails
4. fix the issue
5. rerun from a clean state
6. continue until build succeeds

If a bug is encountered during this process, document the bug's root cause and fix method in `docs/debug/current.md`.

---

## When to use
Use this skill whenever you have modified code and need to verify compilation for the following targets:

- `zedinfer_ops`
- `ping`

Also use it when:
- changing build files
- changing test-related code
- changing headers or shared libraries
- changing code that may impact dependency resolution, symbols, or compilation behavior

---

## Build workflow

Always run the build through the provided script:

```bash
bash auto-build-test/scripts/run_build_check.sh
```

The script will:

- ensure `logs/` and `docs/debug/` exist
- overwrite `logs/build.log` at the start of every build round
- run each build command in order
- append each command's output to `logs/build.log`
- stop immediately on the first failing command with a non-zero exit code

---

## Commands covered by this skill

The build script runs these commands in order:

```bash
xmake clean --all
xmake f -m release --nv-gpu=y --pytest=y
xmake build zedinfer_ops
xmake build ping
xmake run ping
```

All stdout and stderr are appended to:

```text
logs/build.log
```

Before each new round, the old log is cleared.

---

## Required behavior

When using this skill, follow this loop:

1. Modify the code as needed.
2. Run:
   ```bash
   bash auto-build-test/scripts/run_build_check.sh
   ```
3. If the build succeeds:
   - stop
   - summarize success
   - if any issue was fixed during this session, ensure `docs/debug/current.md` reflects the final root cause and fix method
4. If the build fails:
   - inspect `logs/build.log`
   - identify the earliest meaningful error
   - determine the likely root cause
   - make the smallest correct fix
   - update `docs/debug/current.md`
   - rerun the script from a clean state
5. Repeat until all commands pass

---

## Failure analysis rules

When reading `logs/build.log`, always:

- focus on the first real error, not later cascaded errors
- distinguish build stage:
  - clean/configure
  - compile
  - link
  - target registration / xmake configuration
- prioritize root cause over symptom

Common failure types include:

- missing include/header
- symbol rename mismatch
- namespace/type mismatch
- template instantiation failure
- API signature drift after refactor
- missing source file in target
- linker undefined reference
- incompatible xmake config or option usage

Do not paper over errors with broad or unrelated refactors.

---

## Logging and documentation rules

### Build log
- The build log file is:
  ```text
  logs/build.log
  ```
- It must be overwritten at the start of each new retry round
- Each command should be clearly separated in the log

### Debug record
If a bug or build issue is found, write the findings to:

```text
docs/debug/current.md
```

Use the template in:

```text
auto-build-test/references/debug_template.md
```

At minimum, record:

- timestamp
- build round
- failing command
- error summary
- root cause
- fix method
- affected files
- current status

If multiple retry rounds happen, update the document so it reflects the latest active issue and the final resolution.

---

## Constraints
- Continue until the build passes unless blocked by environment or external dependency issues
- Prefer minimal, targeted fixes
- Do not make unrelated cleanup changes
- Always rerun the full build flow after a fix
- Do not reuse old logs from previous rounds
- If the failure is environmental rather than code-related, document that clearly in `docs/debug/current.md`

---

## Success criteria
This skill is complete only when all of the following succeed:

```bash
xmake clean --all
xmake f -m release --nv-gpu=y --pytest=y
xmake build zedinfer_ops
xmake build ping
xmake run ping
```

and the final build round exits successfully.

---

## Expected outputs
- Fresh build log at `logs/build.log`
- Debug notes at `docs/debug/current.md` if any issue occurred
- Code fixes applied until compilation succeeds