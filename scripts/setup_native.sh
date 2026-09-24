#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ort_version=${ONNXRUNTIME_VERSION:-1.22.0}
deps_dir="$repo_root/.deps"
install_dir="$deps_dir/onnxruntime"
archive="$deps_dir/onnxruntime-linux-x64-${ort_version}.tgz"
url="https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/onnxruntime-linux-x64-${ort_version}.tgz"

mkdir -p "$deps_dir"
if [[ ! -f "$install_dir/include/onnxruntime_cxx_api.h" ]]; then
    echo "Downloading ONNX Runtime ${ort_version} from ${url}"
    curl --fail --location --retry 3 --output "$archive" "$url"
    extracted="$deps_dir/onnxruntime-linux-x64-${ort_version}"
    rm -rf "$extracted"
    tar -xzf "$archive" -C "$deps_dir"
    rm -rf "$install_dir"
    mv "$extracted" "$install_dir"
fi

python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found at $python_bin" >&2
    echo "Set PYTHON_BIN to a Python 3.12 interpreter with pybind11 installed." >&2
    exit 1
fi

cmake -S "$repo_root" -B "$repo_root/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT="$install_dir" \
    -DPython_EXECUTABLE="$python_bin" \
    -Dpybind11_DIR="$($python_bin -m pybind11 --cmakedir)"
cmake --build "$repo_root/build" --parallel

echo "Native module built. Use: PYTHONPATH=$repo_root/build $python_bin -c 'import adaptive_core'"
