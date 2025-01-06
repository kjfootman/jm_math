#!/bin/bash
BASE=$(dirname "$0")

cd $BASE &&
cargo clean --doc &&
cargo test --doc -- --nocapture