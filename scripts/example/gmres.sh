#!/bin/bash

EXAMPLE="gmres_demo"

BASE=$(dirname "$0")
cd "$BASE/../.." || exit && pwd

RUST_LOG=info cargo run -r --example "$EXAMPLE"
