# Config Profiles

Rayn keeps runnable TOML profiles here instead of the repository root.

## Active

Current simulated-monitoring and actively compared profiles.

- `config.rayn-momentum-paper.toml` - conservative long-only simulated profile.
- `config.rayn-short-bounce-paper.toml` - short bounce-failure observer.
- `config.rayn-short-paper.toml` - baseline short observer.
- `config.rayn-short-paper-aggressive.toml` - higher-risk short observer variant.
- `config.rayn-optimized-v3.toml` - latest bounded-grid research profile.

## Research

Stress-test and experiment profiles that are still useful for local research.

## Baselines

Small baseline and example configs used by older backtests and docs.

Legacy commands such as `--config config.optimized.toml` still work through the
repository path resolver, but new commands should use the explicit paths above.
