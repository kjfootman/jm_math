#!/bin/bash

# TEST_NAME="dot_product_bench"
# TEST_NAME="vector"
# TEST_NAME="vector_calc_residual"
# TEST_NAME="conjugate_gradient"
TEST_NAME="gmres"
# TEST_NAME="hgmres"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

RAYON_NUM_THREADS=8 cargo bench -- "$TEST_NAME"
