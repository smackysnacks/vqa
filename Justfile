set positional-arguments

alias t := test
alias c := check
alias b := build
alias l := lint

help:
    @just --list

# Run cargo check on workspace
check:
    cargo check --workspace --all-targets

# Run cargo build on workspace
build:
    cargo build --workspace

# Run cargo nextest on workspace
test *args:
    #!/usr/bin/env bash
    if ! command -v cargo-nextest >/dev/null; then
        echo "cargo-nextest not found. You can install it by running: cargo install cargo-nextest"
        exit 1
    fi
    cargo nextest run --no-tests=warn --workspace "$@"

# Test and produce a code coverage report
coverage:
    #!/usr/bin/env bash
    if ! command -v cargo-llvm-cov &>/dev/null || ! command -v cargo-nextest &>/dev/null; then
        echo "cargo-nextest or cargo-llvm-cov not found. You can install them by running: cargo install cargo-llvm-cov cargo-nextest"
        exit 1
    fi
    cargo llvm-cov nextest --workspace

# Run cargo clippy on workspace, as strict as CI
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Play a VQA movie in a window; scale is 1, 2, or 4
play file scale="2":
    cargo run --release --example player -- "{{file}}" "{{scale}}"

# Fuzz a cargo-fuzz target (parser | adpcm | lcw); extra args go to libFuzzer, e.g. `just fuzz parser -max_total_time=60`
fuzz target="parser" *args:
    #!/usr/bin/env bash
    if ! command -v cargo-fuzz >/dev/null; then
        echo "cargo-fuzz not found. You can install it by running: cargo install cargo-fuzz"
        exit 1
    fi
    if ! rustup toolchain list | grep -q '^nightly'; then
        echo "cargo-fuzz needs a nightly toolchain. You can install one by running: rustup toolchain install nightly"
        exit 1
    fi
    # Seed the parser corpus with the bundled sample movie
    if [ "$1" = "parser" ] && [ ! -e fuzz/corpus/parser/wwlogo.vqa ]; then
        mkdir -p fuzz/corpus/parser
        cp examples/wwlogo.vqa fuzz/corpus/parser/
    fi
    # Pass the host triple explicitly: cargo-fuzz defaults to the triple it
    # was itself built for (e.g. musl), which breaks ASan on a gnu host
    host=$(rustc +nightly -vV | sed -n 's/^host: //p')
    cargo +nightly fuzz run "$1" --target "$host" -- "${@:2}"

# Scan Cargo.lock for known vulnerabilities in dependencies
audit:
    #!/usr/bin/env bash
    if ! command -v cargo-audit >/dev/null; then
        echo "cargo-audit not found. You can install it by running: cargo install cargo-audit"
        exit 1
    fi
    cargo audit

# Show outdated dependencies
show-outdated:
    cargo outdated --workspace
