#!/bin/bash

TEST_FUNC="csr_from_mtx_test"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

# RUST_LOG=jm_lib=debug cargo test "$TEST_FUNC" -- --nocapture
RUST_LOG=jm_lib=debug cargo test -r "$TEST_FUNC" -- --nocapture
