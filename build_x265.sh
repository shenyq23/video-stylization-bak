#!/bin/bash
# Build script for x265 encoder
# Reference: deps/x265/build/linux/make-Makefiles.bash
#
# This script builds:
# 1. Static-linked x265 CLI binary -> bin/x265
# 2. Shared library (.so/.dylib) for native Python wrapper -> bin/libx265.so
#
# Usage:
#   ./build_x265.sh              # Build latest version
#   ./build_x265.sh <commit>     # Build specific commit/tag

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
X265_DIR="$PROJECT_ROOT/deps/x265"
X265_SOURCE="$X265_DIR/source"
BUILD_DIR="$X265_DIR/build_output"
BIN_DIR="$PROJECT_ROOT/bin"

# Handle commit hash argument
COMMIT_HASH="${1:-}"
if [ -n "$COMMIT_HASH" ]; then
    echo "Checking out x265 to commit: $COMMIT_HASH"
    cd "$X265_DIR"
    # Only fetch if commit doesn't exist locally
    if ! git cat-file -t "$COMMIT_HASH" &>/dev/null; then
        echo "Commit not found locally, fetching..."
        git fetch --all
    else
        echo "Commit found locally, skipping fetch"
    fi
    git checkout "$COMMIT_HASH"
    cd "$PROJECT_ROOT"
fi

# Detect OS
UNAME=$(uname)
if [ "$UNAME" = "Darwin" ]; then
    SHARED_EXT="dylib"
    # NPROC=$(sysctl -n hw.ncpu)
else
    SHARED_EXT="so"
    # NPROC=$(nproc)
fi

NPROC=16  # Fixed number of threads for consistency

echo "=========================================="
echo "Building x265 on $UNAME"
echo "Source: $X265_SOURCE"
echo "Build dir: $BUILD_DIR"
echo "Using $NPROC threads"
echo "=========================================="

# Create directories
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"/{static,shared}
mkdir -p "$BIN_DIR"

# ==========================================
# Part 1: Build static x265 CLI
# ==========================================
echo ""
echo "[1/2] Building static x265 CLI..."
cd "$BUILD_DIR/static"

if [ "$UNAME" = "Linux" ]; then
    # Linux: fully static linked binary
    cmake -G "Unix Makefiles" "$X265_SOURCE" \
        -DCMAKE_CXX_STANDARD=11 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DENABLE_SHARED=OFF \
        -DSTATIC_LINK_CRT=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-static" \
        -DCMAKE_BUILD_TYPE=Release
else
    # macOS: static linking of x265 lib only (full static not supported)
    cmake -G "Unix Makefiles" "$X265_SOURCE" \
        -DCMAKE_CXX_STANDARD=11 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DENABLE_SHARED=OFF \
        -DCMAKE_BUILD_TYPE=Release
fi

make

# Copy static binary
cp x265 "$BIN_DIR/x265"
echo "Static x265 CLI installed to: $BIN_DIR/x265"

# ==========================================
# Part 2: Build shared library for native wrapper
# ==========================================
echo ""
echo "[2/2] Building shared library..."
cd "$BUILD_DIR/shared"

cmake -G "Unix Makefiles" "$X265_SOURCE" \
    -DCMAKE_CXX_STANDARD=11 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DENABLE_SHARED=ON \
    -DCMAKE_BUILD_TYPE=Release

make

# Copy shared library
cp libx265.$SHARED_EXT "$BIN_DIR/"
echo "Shared library installed to: $BIN_DIR/libx265.$SHARED_EXT"

rm -rf "$BUILD_DIR"

# ==========================================
# Done
# ==========================================
echo ""
echo "=========================================="
echo "Build complete!"
echo "  Static CLI: $BIN_DIR/x265"
echo "  Shared lib: $BIN_DIR/libx265.$SHARED_EXT"
echo "=========================================="
