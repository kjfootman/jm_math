#!/bin/bash

BASE=$(dirname "$0")
cd "$BASE/.." || exit && pwd

cd "docs/manual/"
mdbook serve --open
