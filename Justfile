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

# Fuzz a cargo-fuzz target (parser | adpcm | lcw | snd1 | video); extra args go to libFuzzer, e.g. `just fuzz parser -max_total_time=60`
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

# Download the sample movies in tests/samples.txt into samples/, checking each md5
samples:
    #!/usr/bin/env bash
    set -euo pipefail
    grep -v '^#' tests/samples.txt | while read -r md5 file url; do
        [ -n "$md5" ] || continue
        if ! echo "$md5  samples/$file" | md5sum --check --status 2>/dev/null; then
            mkdir -p "samples/$(dirname "$file")"
            curl -fsSL --retry 3 -o "samples/$file" "$url"
            echo "$md5  samples/$file" | md5sum --check --quiet
        fi
    done

# Run the tests that decode the sample movies (download them first with `just samples`)
test-samples *args:
    cargo test --release --test samples -- --ignored "$@"

# Run the bench example: `just bench <time|hash> <native|generic|wasm|wasm-simd> [file...]`,
# over every sample movie and wwlogo.vqa when no file is given. `generic` builds for
# baseline x86-64 in target/x86-64, as crates.io users get it; `wasm` runs wasm32-wasip1
# under Node's WASI, and `wasm-simd` adds simd128 (in target/wasm-simd128)
bench cmd="time" target="native" *files:
    #!/usr/bin/env bash
    set -euo pipefail
    files=("${@:3}")
    if [ ${#files[@]} -eq 0 ]; then
        files=($(grep -v '^#' tests/samples.txt | awk 'NF { print "samples/" $2 }') examples/wwlogo.vqa)
    fi
    case "$2" in
        native)
            cargo build -q --release --example bench
            ./target/release/examples/bench "$1" "${files[@]}" ;;
        generic)
            RUSTFLAGS="-C target-cpu=x86-64" cargo build -q --release --example bench --target-dir target/x86-64
            ./target/x86-64/release/examples/bench "$1" "${files[@]}" ;;
        wasm | wasm-simd)
            dir=target
            if [ "$2" = wasm-simd ]; then
                dir=target/wasm-simd128
                export RUSTFLAGS="-C target-feature=+simd128"
            fi
            cargo build -q --release --example bench --target wasm32-wasip1 --target-dir "$dir"
            node --no-warnings --input-type=module -e '
                import { readFile } from "node:fs/promises";
                import { WASI } from "node:wasi";
                const [wasm, ...args] = process.argv.slice(1);
                const wasi = new WASI({ version: "preview1", args: [wasm, ...args], preopens: { ".": "." } });
                const module = await WebAssembly.compile(await readFile(wasm));
                wasi.start(await WebAssembly.instantiate(module, wasi.getImportObject()));
            ' "$dir/wasm32-wasip1/release/examples/bench.wasm" "$1" "${files[@]}" ;;
        *)
            echo "unknown target $2: use native, generic, wasm or wasm-simd" >&2
            exit 1 ;;
    esac

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
