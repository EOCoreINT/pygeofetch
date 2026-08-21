#!/usr/bin/env bash
# PyGeoFetch — Local Development Setup Script
# Usage: bash setup_dev.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   PyGeoFetch v2.4.0 Setup       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
echo ""

# ── 1. Check Python ───────────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3 || command -v python || error "Python not found")
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    error "Python 3.9+ required. Found: $PY_VER"
fi
success "Python $PY_VER"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv .venv
    success "Created .venv/"
else
    info "Using existing .venv/"
fi

# Activate
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    error "Could not activate virtual environment"
fi
success "Virtual environment activated"

# ── 3. Install package ────────────────────────────────────────────────────────
info "Installing pygeofetch in editable mode..."
pip install --upgrade pip -q
pip install -e . -q
success "pygeofetch installed"

# ── 4. Install dev + test deps ────────────────────────────────────────────────
info "Installing dev dependencies..."
pip install pytest pytest-cov pytest-mock -q
success "Dev dependencies installed"

# ── 5. Verify CLI ─────────────────────────────────────────────────────────────
info "Verifying CLI..."
pygeofetch --version
success "CLI working"

# ── 6. Run tests ──────────────────────────────────────────────────────────────
info "Running unit tests..."
python -m pytest tests/unit/ -q --tb=short
success "Tests passed"

# ── 7. Run doctor ─────────────────────────────────────────────────────────────
info "Running diagnostics..."
pygeofetch doctor

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup complete! Try these commands:            ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                  ║${NC}"
echo -e "${GREEN}║  pygeofetch status                          ║${NC}"
echo -e "${GREEN}║  pygeofetch providers list                  ║${NC}"
echo -e "${GREEN}║  pygeofetch search run \\                    ║${NC}"
echo -e "${GREEN}║    --bbox \"-74,40,-73,41\" \\                      ║${NC}"
echo -e "${GREEN}║    --providers aws_earth \\                       ║${NC}"
echo -e "${GREEN}║    --cloud-cover 0-20                            ║${NC}"
echo -e "${GREEN}║                                                  ║${NC}"
echo -e "${GREEN}║  See QUICKSTART.md for full docs                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
