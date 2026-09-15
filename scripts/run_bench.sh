#!/bin/bash

TEST_NAME="dot_product_bench"

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

cargo bench -- "$TEST_NAME"
