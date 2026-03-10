#!/usr/bin/env bash
set -euo pipefail

# brane installer
# Usage: ./install.sh [--big]

BIG=false
for arg in "$@"; do
    case $arg in
        --big) BIG=true ;;
        -h|--help)
            echo "Usage: ./install.sh [--big]"
            echo ""
            echo "Install brane and its dependencies."
            echo ""
            echo "Options:"
            echo "  --big    Also pull the 72B model (~48GB download, needs 48GB+ VRAM)"
            echo "  -h       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

info() { echo -e "\033[1;34m==>\033[0m \033[1m$1\033[0m"; }
ok()   { echo -e "\033[1;32m==>\033[0m \033[1m$1\033[0m"; }
err()  { echo -e "\033[1;31m==>\033[0m \033[1m$1\033[0m" >&2; }

# --- Check prerequisites ---

info "Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    err "Python 3 is required but not found."
    err "Install it with: sudo apt install python3 python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    err "Python 3.12+ is required (found $PYTHON_VERSION)."
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    err "NVIDIA GPU drivers not found. brane requires an NVIDIA GPU."
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 8000 ]; then
    err "GPU has ${GPU_MEM}MB VRAM. The 7B model needs at least 8GB."
    exit 1
fi

ok "Prerequisites met: Python $PYTHON_VERSION, ${GPU_MEM}MB VRAM"

# --- Install Ollama ---

if command -v ollama &>/dev/null; then
    ok "Ollama already installed: $(ollama --version)"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sudo sh
    ok "Ollama installed: $(ollama --version)"
fi

# Wait for Ollama to be ready
if ! ollama list &>/dev/null 2>&1; then
    info "Starting Ollama service..."
    if command -v systemctl &>/dev/null && systemctl is-enabled ollama &>/dev/null 2>&1; then
        sudo systemctl start ollama
    else
        ollama serve &>/dev/null &
        sleep 2
    fi
fi

# --- Pull models ---

info "Pulling Qwen2.5-VL 7B model (~5GB)..."
ollama pull qwen2.5vl:7b

if [ "$BIG" = true ]; then
    info "Pulling Qwen2.5-VL 72B model (~48GB)..."
    ollama pull qwen2.5vl:72b
fi

# --- Set up Python environment ---

info "Creating virtual environment..."

if [ -d "$VENV_DIR" ]; then
    info "Existing .venv found, reusing it."
else
    python3 -m venv "$VENV_DIR"
fi

info "Installing brane..."

if command -v uv &>/dev/null; then
    uv pip install --python "$VENV_DIR/bin/python" -e "$SCRIPT_DIR"
else
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"
fi

# --- Verify ---

if "$VENV_DIR/bin/brane" --help &>/dev/null; then
    ok "brane installed successfully!"
else
    err "Installation failed. Check the output above for errors."
    exit 1
fi

# --- Print usage instructions ---

SHELL_RC=""
if [ -n "${BASH_VERSION:-}" ] || [ "$(basename "$SHELL")" = "bash" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

echo ""
ok "Done! To use brane, either:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Or add it to your PATH permanently:"
if [ -n "$SHELL_RC" ]; then
    echo "     echo 'export PATH=\"$VENV_DIR/bin:\$PATH\"' >> $SHELL_RC"
    echo "     source $SHELL_RC"
else
    echo "     Add $VENV_DIR/bin to your PATH"
fi
echo ""
echo "  3. Or run it directly:"
echo "     $VENV_DIR/bin/brane image.png"
echo ""
echo "  Try it out:"
echo "     brane --help"
