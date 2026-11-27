#!/bin/bash

# TEST_NAME="dot_product_bench"
# TEST_NAME="vector"
# TEST_NAME="vector_calc_residual"
# TEST_NAME="conjugate_gradient"
TEST_NAME="gmres"
# TEST_NAME="hgmres"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

# 홈 PC 테스트 용
RAYON_NUM_THREADS=4 cargo bench -- "$TEST_NAME"

# 서버 테스트 용 (2 socket)
# RAYON_NUM_THREADS=4 \
# taskset -c 0-3 \
# cargo bench -- "$TEST_NAME"
