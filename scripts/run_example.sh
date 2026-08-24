#!/bin/bash

EXAMPLE="gauss_seidel"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

# RUST_LOG=jm_lib=debug cargo run -r --example "$EXAMPLE"
RUST_LOG=info cargo run -r --example "$EXAMPLE"
