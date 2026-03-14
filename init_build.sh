#!/usr/bin/env bash
set -euo pipefail

# Build TensileLite client and install Origami analytical model.
# Run once after cloning / updating submodules.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building TensileLite client ==="
cd hipblaslt/tensilelite/
pip3 install invoke
invoke build-client
cd ../..

echo "=== Installing Origami ==="
pip3 install origami/shared/origami/python/

echo "=== Done ==="
