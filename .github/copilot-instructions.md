# Copilot Instructions

## Working style

- Prioritize static analysis before expensive execution.
- Read Dockerfile, docker-compose files, lockfiles, dependency manifests, and relevant source files before proposing long-running commands.
- Treat builds, installs, and large container operations as expensive steps.
- Do not use repeated trial-and-error builds as the main debugging method.

## Expensive commands

- If a command may take several minutes, first narrow the problem using file inspection and dependency analysis.
- Only propose a full build, reinstall, or container recreation when there is a strong hypothesis.
- Use expensive commands mainly for final verification, not for exploration.
- When a previous build error already identifies a dependency conflict or config error, fix that root cause before suggesting another full build.

## Docker and dependency debugging

- For Docker issues, inspect Dockerfile ordering, cache behavior, bind mounts, and dependency resolution first.
- Check for conflicts between package pins, lockfiles, overrides, and transitive dependencies before rebuilding.
- Prefer minimal, focused edits over broad container rewrites.
- Keep development containers simple when they are used only for execution.

## Collaboration expectations

- Minimize costly iterations.
- Explain the hypothesis before suggesting a long command.
- Prefer one well-reasoned change over multiple speculative changes.
- Preserve user time: avoid proposing 15-20 minute commands unless clearly justified.
