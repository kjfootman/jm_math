#!/bin/bash

# TEST_FUNC="vector_from_mtx_test"
# TEST_FUNC="gauss_seidel_test"
TEST_FUNC="csr_diagonal_test"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

RUST_LOG=jm_lib=debug cargo test -r "$TEST_FUNC" -- --nocapture
