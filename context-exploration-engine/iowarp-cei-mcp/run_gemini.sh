#!/bin/bash
# Simple script to run Gemini CLI with IOWarp MCP

set -e

echo "Setting up IOWarp MCP for Gemini CLI..."

# Install packages if needed
pip install --break-system-packages -q mcp pyyaml 2>/dev/null || true

# Set environment
export PYTHONPATH=/workspace/build/bin:/workspace/context-exploration-engine/iowarp-cei-mcp/src
export CHI_REPO_PATH=/workspace/build/bin
export LD_LIBRARY_PATH=/workspace/build/bin:${LD_LIBRARY_PATH}

# Create MCP config
mkdir -p ~/.config/gemini-cli

cat > ~/.config/gemini-cli/mcp_servers.json << 'MCPCONF'
{
  "mcpServers": {
    "iowarp-cei": {
      "command": "/workspace/context-exploration-engine/iowarp-cei-mcp/venv/bin/python3",
      "args": ["-m", "iowarp_cei_mcp.server"],
      "env": {
        "PYTHONPATH": "/workspace/build/bin:/workspace/context-exploration-engine/iowarp-cei-mcp/src",
        "CHI_REPO_PATH": "/workspace/build/bin",
        "LD_LIBRARY_PATH": "/workspace/build/bin"
      }
    }
  }
}
MCPCONF

echo "✓ MCP config created at ~/.config/gemini-cli/mcp_servers.json"

# Start runtime in background
if ! pgrep -f chimaerad > /dev/null; then
    echo "Starting IOWarp runtime..."
    /workspace/build/bin/chimaerad > /dev/null 2>&1 &
    sleep 3
    echo "✓ Runtime started"
else
    echo "✓ Runtime already running"
fi

echo ""
echo "Ready! Starting Gemini CLI with IOWarp MCP..."
echo ""
echo "Try asking:"
echo "  - List the available tools"
echo "  - Store /tmp/test.txt in context 'my_data'"
echo "  - Query all contexts"
echo ""

# Start Gemini
gemini chat --mcp-server iowarp-cei
