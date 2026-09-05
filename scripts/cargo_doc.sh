#!/bin/bash

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

cargo clean --doc
cargo doc --no-deps --open
