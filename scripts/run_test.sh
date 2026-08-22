#!/bin/bash

TEST_FUNC="csr_from_mtx_test"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

# RUST_LOG=jm_lib=info cargo test "$TEST_FUNC"
RUST_LOG=jm_lib=debug cargo test "$TEST_FUNC" -- --nocapture
