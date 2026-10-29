#!/bin/bash
frontend=robohead_baseline/frontend
destination=${1:?Usage: setup_frontend.sh INSTALL_PREFIX}
destination=$(realpath -m -- "$destination")
source_dir="$destination/src/odas"
mkdir -p "$source_dir" || exit $?
archive=$(mktemp) || exit $?
trap 'rm -f "$archive"' EXIT
curl \
  -fL https://github.com/introlab/odas/archive/bcb845434495e293df3d48f1203b7a86e1852449.tar.gz \
  -o "$archive" || exit $?
tar \
  -xzf "$archive" \
  --strip-components=1 \
  -C "$source_dir" || exit $?
cmake \
  -S "$source_dir" \
  -B "$destination/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$destination" \
  -DODAS_INSTALL_EXECUTABLES=ON || exit $?
cmake \
  --build "$destination/build" \
  -j 2 || exit $?
cmake \
  --install "$destination/build" || exit $?
cc \
  -shared \
  -fPIC \
  -O2 \
  "$frontend/fifo.c" \
  -ldl \
  -o "$frontend/fifo.so"
