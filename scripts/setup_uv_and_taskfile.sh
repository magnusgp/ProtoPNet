#!/usr/bin/env bash
set -euo pipefail

echo "== setup: astral 'uv' and Taskfile runner =="

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Ensure python3 and pip3 are available
if ! command_exists python3; then
  echo "python3 not found. Please install Python 3 and re-run this script." >&2
  exit 1
fi

if ! command_exists pip3; then
  echo "pip3 not found. Trying to use python3 -m pip" >&2
  PIP_CMD="python3 -m pip"
else
  PIP_CMD="pip3"
fi

echo "1) Install astral 'uv' (attempting common package names)"
if command_exists uv; then
  echo " - 'uv' already installed. Skipping."
else
  echo " - trying: $PIP_CMD install --user uv"
  if $PIP_CMD install --user uv; then
    echo " - installed 'uv' (package name: uv)"
  else
    echo " - first attempt failed, trying alternative package name 'astral-uv'"
    if $PIP_CMD install --user astral-uv; then
      echo " - installed 'astral-uv'"
    else
      echo " - could not install 'uv' automatically. Please install astral's uv manually and ensure the 'uv' command is on your PATH." >&2
    fi
  fi
fi

echo ""
echo "2) Install Taskfile runner (go-task 'task' CLI)"
if command_exists task; then
  echo " - 'task' already installed. Skipping."
else
  if command_exists brew; then
    echo " - Homebrew detected, installing go-task via brew"
    brew install go-task/tap/go-task || echo " - brew install failed; please install go-task manually: https://taskfile.dev/"
  else
    echo " - Homebrew not found. Please install the Taskfile runner (go-task) manually." 
    echo "   Visit https://taskfile.dev/ for installation instructions."
  fi
fi

echo ""
echo "Finish: ensure user local bin is on your PATH if you used --user installs. Common locations:" 
echo "  - ~/.local/bin"
echo "  - ~/Library/Python/\$(python3 -c 'import sys; print("%d.%d"%(sys.version_info.major,sys.version_info.minor))')/bin"
echo "If you installed 'uv' with pip --user add the appropriate bin directory to your PATH and restart your shell." 

echo "Done. You can now use the Taskfile (Taskfile.yml) with the 'task' command. Example:"
echo "  task prepare:cub200"
