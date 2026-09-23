#!/bin/bash

# TEST_NAME="dot_product_bench"
# TEST_NAME="vector"
# TEST_NAME="vector_calc_residual"
TEST_NAME="conjugate_gradient"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

cargo bench -- "$TEST_NAME"
