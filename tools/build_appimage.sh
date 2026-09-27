#!/usr/bin/env bash
# Build dist/yonga-<version>-x86_64.AppImage: one file that runs yonga with no
# Python, pip or OpenVINO installed on the host.
#
#   tools/build_appimage.sh [--wheel dist/yonga-X.Y.Z-py3-none-any.whl]
#
# Why an AppImage around a real interpreter, not a frozen executable: yonga
# starts child processes with its own interpreter (`python -m
# yonga.adapters.gguf_import`, the exporter-range lookup) and builds managed
# export environments with `python -m venv`. A PyInstaller/Nuitka binary has no
# `python` to hand them; an embedded relocatable CPython does. The image holds
#
#   usr/python/   python-build-standalone CPython with yonga[openvino,serve]
#   AppRun        runs `python -m yonga`
#
# Conversion (optimum-intel, torch) is deliberately not bundled: it is ~1.5 GB
# and model-specific. `prepare` builds a managed export environment for it on
# first use, from a lasting copy of the bundled interpreter.
#
# Every download is pinned and checked against its SHA-256. Host needs: bash,
# curl, tar, sha256sum and a Python with `build` when no wheel is given. The
# result runs on x86_64 Linux with glibc 2.28 or newer.
set -euo pipefail

PBS_RELEASE=20260924
PYTHON_VERSION=3.14.7
PYTHON_SHA256=bd0d0568ccded07bbf1c87727230dc5dd0187e706a87da23c4de78388a229b78
APPIMAGETOOL_VERSION=1.9.1
APPIMAGETOOL_SHA256=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
RUNTIME_RELEASE=20251108
RUNTIME_SHA256=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d
#: The newest glibc symbol any bundled library may need; see the check below.
MAX_GLIBC=2.28

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${YONGA_APPIMAGE_WORK:-$root/build/appimage}"
dist="${YONGA_APPIMAGE_DIST:-$root/dist}"
downloads="$work/downloads"
wheel=""

while [ $# -gt 0 ]; do
    case "$1" in
        --wheel) wheel="$(realpath "$2")"; shift 2 ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

fetch() {  # fetch URL FILE SHA256
    local url="$1" file="$downloads/$2" sum="$3"
    if [ ! -f "$file" ] || ! echo "$sum  $file" | sha256sum --check --status; then
        echo "downloading $2"
        curl -fsSL --retry 3 -o "$file.part" "$url"
        mv "$file.part" "$file"
    fi
    echo "$sum  $file" | sha256sum --check --quiet
}

mkdir -p "$downloads" "$dist"

python_tar="cpython-${PYTHON_VERSION}+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
fetch "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/${python_tar/+/%2B}" \
    "$python_tar" "$PYTHON_SHA256"
fetch "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_VERSION}/appimagetool-x86_64.AppImage" \
    appimagetool-x86_64.AppImage "$APPIMAGETOOL_SHA256"
fetch "https://github.com/AppImage/type2-runtime/releases/download/${RUNTIME_RELEASE}/runtime-x86_64" \
    runtime-x86_64 "$RUNTIME_SHA256"
chmod +x "$downloads/appimagetool-x86_64.AppImage"

if [ -z "$wheel" ]; then
    rm -rf "$work/wheel"
    python3 -m build --wheel --outdir "$work/wheel" "$root"
    wheel="$(ls "$work"/wheel/yonga-*.whl)"
fi
version="$(basename "$wheel" | cut -d- -f2)"
echo "building yonga $version from $(basename "$wheel")"

appdir="$work/AppDir"
rm -rf "$appdir"
mkdir -p "$appdir/usr"
tar -xzf "$downloads/$python_tar" -C "$appdir/usr"   # unpacks as usr/python/
python="$appdir/usr/python/bin/python3"

platform_flags() {  # every manylinux tag up to MAX_GLIBC; pip does not expand one itself
    local minor
    for minor in $(seq 17 "${MAX_GLIBC#2.}"); do
        printf -- '--platform manylinux_2_%s_x86_64 ' "$minor"
    done
    printf -- '--platform manylinux2014_x86_64 --platform manylinux2010_x86_64 --platform manylinux1_x86_64'
}

# Wheels are chosen for the oldest supported glibc, not for the build host:
# left alone, pip takes e.g. cryptography's manylinux_2_34 build on a new host.
site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
py_tag="$("$python" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
"$python" -m pip install --no-cache-dir --disable-pip-version-check --no-input --no-warn-script-location \
    --only-binary=:all: $(platform_flags) \
    --implementation cp --python-version "$py_tag" --abi "cp$py_tag" \
    --target "$site" --upgrade \
    "yonga[openvino,serve] @ file://$wheel"
rm -rf "$site/bin"   # console scripts with a build-time shebang; AppRun is the entry point
"$python" -m pip uninstall --yes --disable-pip-version-check pip >/dev/null
# The image is read-only at run time, so bytecode is compiled now or never.
"$python" -m compileall -q -j 0 "$appdir/usr/python/lib" >/dev/null || true

install -m 0755 "$root/packaging/appimage/AppRun" "$appdir/AppRun"
install -m 0644 "$root/packaging/appimage/yonga.desktop" "$appdir/yonga.desktop"
install -m 0644 "$root/packaging/appimage/yonga.svg" "$appdir/yonga.svg"
ln -sf yonga.svg "$appdir/.DirIcon"

# Every shared object must load on the oldest glibc the image claims to support.
newest="$(find "$appdir/usr/python" -name '*.so*' -type f -print0 \
    | xargs -0 objdump -T 2>/dev/null \
    | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 | cut -d_ -f2)"
echo "newest glibc symbol required: ${newest:-none}"
if [ -n "$newest" ] && [ "$(printf '%s\n%s\n' "$MAX_GLIBC" "$newest" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
    echo "a bundled library needs glibc $newest, newer than the supported $MAX_GLIBC" >&2
    exit 1
fi

output="$dist/yonga-${version}-x86_64.AppImage"
rm -f "$output"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$downloads/appimagetool-x86_64.AppImage" \
    --no-appstream --runtime-file "$downloads/runtime-x86_64" "$appdir" "$output"
(cd "$dist" && sha256sum "$(basename "$output")" > "$(basename "$output").sha256")

# Smoke test: the image starts, reports its version and finds its runtime.
APPIMAGE_EXTRACT_AND_RUN=1 "$output" --version
APPIMAGE_EXTRACT_AND_RUN=1 "$output" registry validate >/dev/null
echo "built $output ($(du -h "$output" | cut -f1))"
