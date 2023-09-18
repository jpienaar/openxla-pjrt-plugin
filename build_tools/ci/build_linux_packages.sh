#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_packages.sh
# One stop build of the project's Python packages for Linux. The Linux build is
# complicated because it has to be done via a docker container that has
# an LTS glibc version, all Python packages and other deps.
# This script handles all of those details.
#
# Usage:
# Build everything (all packages, all python versions):
#   ./build_tools/build_linux_packages.sh
#
# Build specific Python versions and packages to custom directory:
#   override_python_versions="cp38-cp38" \
#   packages="plugins" \
#   output_dir="/tmp/wheelhouse" \
#   ./build_tools/ci/python_deploy/build_linux_packages.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp38-cp38 cp39-cp39 cp310-cp310
# Note that our Python packages are version independent so it is typical to
# build with the oldest supported Python vs multiples.
#
# Valid packages:
#   plugins
#   plugins_instrumented (currently does not work)
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories (under runtime/ and
# compiler/) with docker created, root owned builds. Sorry - there is
# no good way around it.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -xeu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
repo_root="$(cd "${this_dir}" && git rev-parse --show-toplevel)"
manylinux_docker_image="${manylinux_docker_image:-}"
python_versions="${override_python_versions:-cp310-cp310 cp311-cp311}"
output_dir="${output_dir:-${repo_root}/bindist}"
plugins="${plugins:-iree/integrations/pjrt/cpu/pjrt_plugin_iree_cpu.so iree/integrations/pjrt/cuda/pjrt_plugin_iree_cuda.so}"
packages="${packages:-plugins python-cpu-wheel python-cuda-wheel}"
package_suffix="${package_suffix:-}"
write_caches="${write_caches:-0}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${manylinux_docker_image}"

  if [ -z "${manylinux_docker_image}" ]; then
    manylinux_docker_image="${manylinux_docker_image:-$(uname -m | awk '{print ($1 == "aarch64") ? "quay.io/pypa/manylinux_2_28_aarch64" : "gcr.io/iree-oss/manylinux2014_x86_64-release@sha256:e83893d35be4ce3558c989e9d5ccc4ff88d058bc3e74a83181059cc76e2cf1f8" }')}"
    if [ -z "${manylinux_docker_image}" ]; then
      echo "ERROR: Could not determine manylinux docker image"
      exit 1
    fi
    echo "Using default docker image: $manylinux_docker_image"
  fi

  # Strip off the rest of the URL. Only the digest is load bearing anyway. If
  # someone's not specifying the digest they may have a bad time, but the most
  # likely scenario is just that the cache hits won't return anything. Actually
  # writing to the cache requires access. It's surprisingly hard to get docker
  # to tell you "give me the fully qualified name to the image you would use if
  # I told you to `docker run` this", which is what we'd want to do this
  # properly.
  cache_key="${manylinux_docker_image##*/}"

  # Canonicalize paths.
  mkdir -p "${output_dir}"
  output_dir="$(cd "${output_dir}" && pwd)"
  echo "Outputting to ${output_dir}"
  mkdir -p "${output_dir}"
  # Mount one level up to get the entire workspace.
  mount_dir="$(cd $repo_root/.. && pwd)"
  docker run --rm \
    -v "${mount_dir}:${mount_dir}" \
    -v "${output_dir}:${output_dir}" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "override_python_versions=${python_versions}" \
    -e "plugins=${plugins}" \
    -e "packages=${packages}" \
    -e "package_suffix=${package_suffix}" \
    -e "output_dir=${output_dir}" \
    -e "write_caches=${write_caches}" \
    -e "cache_key=${cache_key}" \
    -e "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-}" \
    "${manylinux_docker_image}" \
    -- ${this_dir}/${script_name}

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l "${output_dir}"
}

function run_in_docker() {
  echo "Running in docker"
  echo "Using python versions: ${python_versions}"

  local orig_path="${PATH}"

  # Build phase.
  for package in ${packages}; do
    echo "******************** BUILDING PACKAGE ${package} ********************"
    for python_version in ${python_versions}; do
      python_dir="/opt/python/${python_version}"
      if ! [ -x "${python_dir}/bin/python" ]; then
        echo "ERROR: Could not find python: ${python_dir} (skipping)"
        continue
      fi
      cd $repo_root
      export PATH="${python_dir}/bin:${orig_path}"
      echo ":::: Python version $(python --version)"
      echo "::: Running from $(pwd)"
      echo "::: Installing CUDA SDK..."
      cuda_sdk_dir="$($repo_root/../iree/third_party/nvidia_sdk_download/fetch_cuda_toolkit.py /tmp/cuda_sdk)"
      echo "CUDA SDK installed at $cuda_sdk_dir"
      echo "::: Installing python dependencies"
      pip install -r requirements.txt
      echo "::: Configuring bazel"
      python configure.py --cuda-sdk-dir=${cuda_sdk_dir}
      # replace dashes with underscores
      declare -a bazel_flags=(
        --compilation_mode=opt
        --remote_cache="https://storage.googleapis.com/openxla-bazel-cache/${cache_key}"
      )
      if (( write_caches == 1 )); then
        bazel_flags+=(--google_default_credentials)
      else
        bazel_flags+=(--noremote_upload_local_results)
      fi
      package_suffix="${package_suffix//-/_}"
      case "${package}" in
        plugins)
          build_plugins
          ;;

        plugins_instrumented)
          build_plugins_instrumented
          ;;

        python-cpu-wheel)
          build_python_cpu_wheel
          ;;

        python-cuda-wheel)
          build_python_cuda_wheel
          ;;

        *)
          echo "Unrecognized package '${package}'"
          exit 1
          ;;
      esac
    done
  done
}

function build_plugins() {
  local f
  local dest="${output_dir}/pjrt_plugins"
  mkdir -p $dest
  bazel build "${bazel_flags[@]}" ${plugins}
  for f in ${plugins}; do
    cp -fv bazel-bin/$f $dest
  done
}

function build_plugins_instrumented() {
  # TODO: Currently does not compile.
  local f
  local dest="${output_dir}/pjrt_plugins_instrumented"
  mkdir -p $dest
  bazel build "${bazel_flags[@]}" --iree_enable_runtime_tracing ${plugins}
  for f in ${plugins}; do
    cp -fv bazel-bin/$f $dest
  done
}

function build_python_cpu_wheel() {
  # Note that these wheels are Python version independent. We build them
  # with each listed Python version to make sure that the setup machinery
  # works, but the most recent one persists.
  bazel build "${bazel_flags[@]}" iree/integrations/pjrt/cpu/pjrt_plugin_iree_cpu.so
  clean_wheels "openxla_pjrt_plugin_cpu${package_suffix}" "py3-none"
  build_wheel python_packages/openxla_cpu_plugin
  run_audit_wheel "openxla_pjrt_plugin_cpu${package_suffix}" "py3-none"
}

function build_python_cuda_wheel() {
  # Note that these wheels are Python version independent. We build them
  # with each listed Python version to make sure that the setup machinery
  # works, but the most recent one persists.
  bazel build -c opt iree/integrations/pjrt/cuda/pjrt_plugin_iree_cuda.so
  clean_wheels "openxla_pjrt_plugin_cuda${package_suffix}" "py3-none"
  build_wheel python_packages/openxla_cuda_plugin
  run_audit_wheel "openxla_pjrt_plugin_cuda${package_suffix}" "py3-none"
}

function build_wheel() {
  python -m pip wheel --disable-pip-version-check \
    -f https://openxla.github.io/iree/pip-release-links.html \
    -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html \
    -v -w "${output_dir}" "${repo_root}/$@"
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  # Force wildcard expansion here
  generic_wheel="$(echo "${output_dir}/${wheel_basename}-"*"-${python_version}-linux_x86_64.whl")"
  ls "${generic_wheel}"
  echo ":::: Auditwheel ${generic_wheel}"
  auditwheel repair -w "${output_dir}" "${generic_wheel}"
  rm -v "${generic_wheel}"
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels ${wheel_basename} ${python_version}"
  rm -f -v "${output_dir}/${wheel_basename}-"*"-${python_version}-"*".whl"
}

# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
