#!/bin/bash
# Setup script for IOWarp CEI MCP Server with Gemini CLI

set -e

echo "========================================================================"
echo "IOWarp CEI MCP Server - Gemini CLI Setup"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "------------------------------------------------------------------------"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if IOWarp is built
if [ -f "/workspace/build/bin/wrp_cee.cpython-312-aarch64-linux-gnu.so" ] || [ -f "/workspace/build/bin/wrp_cee.so" ]; then
    echo -e "${GREEN}✓${NC} IOWarp Python bindings found"
else
    echo -e "${RED}✗${NC} IOWarp Python bindings not found"
    echo "  Please build with: cmake --preset=debug -DWRP_CORE_ENABLE_PYTHON=ON && cmake --build build -j8"
    exit 1
fi

echo ""

# Step 2: Install Python packages
echo "Step 2: Installing Python packages..."
echo "------------------------------------------------------------------------"

pip install -q pyyaml mcp google-generativeai 2>/dev/null || {
    echo -e "${YELLOW}⚠${NC}  Some packages may already be installed"
}

# Verify MCP SDK
if python3 -c "import mcp" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} MCP SDK installed"
else
    echo -e "${RED}✗${NC} MCP SDK not installed. Installing..."
    pip install mcp
fi

echo ""

# Step 3: Set environment variables
echo "Step 3: Setting up environment..."
echo "------------------------------------------------------------------------"

export PYTHONPATH="/workspace/build/bin:$PYTHONPATH"
export CHI_REPO_PATH="/workspace/build/bin"
export LD_LIBRARY_PATH="/workspace/build/bin:${LD_LIBRARY_PATH}"

echo "export PYTHONPATH=/workspace/build/bin:\$PYTHONPATH" >> ~/.bashrc
echo "export CHI_REPO_PATH=/workspace/build/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/workspace/build/bin:\${LD_LIBRARY_PATH}" >> ~/.bashrc

echo -e "${GREEN}✓${NC} Environment variables set and added to ~/.bashrc"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CHI_REPO_PATH: $CHI_REPO_PATH"

echo ""

# Step 4: Test MCP server
echo "Step 4: Testing MCP server..."
echo "------------------------------------------------------------------------"

cd /workspace/context-exploration-engine/iowarp-cei-mcp

# Test import
if python3 -c "import sys; sys.path.insert(0, 'src'); from iowarp_cei_mcp import server; print('Import successful')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} MCP server module imports successfully"
else
    echo -e "${RED}✗${NC} MCP server import failed"
    exit 1
fi

# Run quick test
echo ""
echo "Running end-to-end test (this may take a minute)..."
if timeout 120 python3 test_mcp_end_to_end.py > /tmp/mcp_test.log 2>&1; then
    echo -e "${GREEN}✓${NC} End-to-end test PASSED"
else
    echo -e "${RED}✗${NC} End-to-end test FAILED"
    echo "  Check /tmp/mcp_test.log for details"
    tail -20 /tmp/mcp_test.log
    exit 1
fi

echo ""

# Step 5: Create MCP configuration for Gemini
echo "Step 5: Creating Gemini CLI configuration..."
echo "------------------------------------------------------------------------"

mkdir -p ~/.config/gemini-cli

cat > ~/.config/gemini-cli/mcp_servers.json << 'ENDCONFIG'
{
  "mcpServers": {
    "iowarp-cei": {
      "command": "python3",
      "args": [
        "-m",
        "iowarp_cei_mcp.server"
      ],
      "env": {
        "PYTHONPATH": "/workspace/build/bin:/workspace/context-exploration-engine/iowarp-cei-mcp/src",
        "CHI_REPO_PATH": "/workspace/build/bin",
        "LD_LIBRARY_PATH": "/workspace/build/bin"
      }
    }
  }
}
ENDCONFIG

echo -e "${GREEN}✓${NC} MCP server config created at ~/.config/gemini-cli/mcp_servers.json"

echo ""

# Step 6: Setup instructions
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set your Gemini API key:"
echo "   ${YELLOW}export GEMINI_API_KEY='your-api-key-here'${NC}"
echo ""
echo "2. Install Gemini CLI (choose one):"
echo "   ${YELLOW}pip install google-gemini-cli${NC}"
echo "   ${YELLOW}npm install -g @google/gemini-cli${NC}"
echo ""
echo "3. Configure Gemini CLI:"
echo "   ${YELLOW}gemini config set apiKey \$GEMINI_API_KEY${NC}"
echo "   ${YELLOW}gemini config set mcpEnabled true${NC}"
echo ""
echo "4. Start Gemini CLI with IOWarp MCP server:"
echo "   ${YELLOW}gemini chat --mcp-server iowarp-cei${NC}"
echo ""
echo "5. Test IOWarp commands in Gemini:"
echo "   User: Store /tmp/test.txt in IOWarp context 'my_data'"
echo "   User: Query all contexts"
echo "   User: Retrieve data from 'my_data'"
echo "   User: Delete context 'my_data'"
echo ""
echo "========================================================================"
echo ""
echo "For detailed instructions, see:"
echo "  ${GREEN}GEMINI_SETUP.md${NC}"
echo ""
echo "To test the MCP server directly (without Gemini):"
echo "  ${YELLOW}python3 test_mcp_end_to_end.py${NC}"
echo ""
echo "========================================================================"
