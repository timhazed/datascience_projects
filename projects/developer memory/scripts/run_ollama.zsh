#!/bin/zsh

set -euo pipefail

# 1. PARAMETER HANDLING
MODE="${1:-}"
# Input name from the benchmark script (e.g., gemma4:e4b)
MODEL="${2:-gemma4:26b}"

if [[ -z "${MODE}" ]]; then
    echo "Usage: ./run_ollama.zsh [hybrid|model-runner] [model_tag]"
    exit 1
fi

# 2. UTILITY FUNCTIONS
stop_native_ollama() {
    echo "Stopping native Ollama processes..."
    osascript -e 'tell app "Ollama" to quit' 2>/dev/null || true
    pkill -9 Ollama 2>/dev/null || true
    pkill -9 ollama 2>/dev/null || true
}

wait_for_api() {
    local port="$1"
    echo "Waiting for server to start on http://localhost:${port}..."
    for i in {1..15}; do
        # Check if the port is open and responding
        if curl -sSf "http://localhost:${port}" >/dev/null 2>&1; then
            echo "Successfully connected to http://localhost:${port}"
            return 0
        fi
        sleep 1
    done
    return 1
}

# 3. MODE SELECTION
if [[ "${MODE}" == "hybrid" ]]; then
    PORT=11434
    echo "Starting HYBRID MODE on port $PORT..."
    stop_native_ollama
    sleep 2 
    ollama serve > /dev/null 2>&1 &
    
elif [[ "${MODE}" == "model-runner" ]]; then
    PORT=12434 
    echo "Switching to MODEL-RUNNER MODE on port $PORT..."
    
    # CRITICAL: We stop native Ollama but do NOT try to 'enable' Docker via CLI
    # This avoids the 'unknown settings keys' error
    stop_native_ollama
    sleep 2 

    # MAP THE MODEL NAME FOR DOCKER PULLS
    # This allows you to pass 'gemma4:e4b' to the script
    if [[ "${MODEL}" == "gemma4:e4b" ]]; then
        DOCKER_MODEL="huggingface.co/lmstudio-community/gemma-3n-e4b-it-mlx-4bit" # Updated
    elif [[ "${MODEL}" == "gemma4:26b" ]]; then
        DOCKER_MODEL="huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
    else
        DOCKER_MODEL="${MODEL}"
    fi

else
    echo "Invalid mode. Use 'hybrid' or 'model-runner'."
    exit 1
fi

# 4. READINESS & MODEL CHECK
if ! wait_for_api "${PORT}"; then
    echo "Error: Server failed to start on port ${PORT}."
    echo "For model-runner, ensure 'Host-side TCP' is checked in Docker Settings > AI."
    exit 1
fi

if [[ "${MODE}" == "model-runner" ]]; then
    echo "Checking Docker store for: ${DOCKER_MODEL}..."
    if ! docker model ls | grep -q "${DOCKER_MODEL}"; then
        echo "Model not found in Docker. Pulling now..."
        docker model pull "${DOCKER_MODEL}"
    fi
else
    echo "Checking Ollama store for: ${MODEL}..."
    if ! ollama list | grep -q "${MODEL}"; then
        ollama pull "${MODEL}"
    fi
fi

echo "READY: Mode='${MODE}' Model='${MODEL}' URL='http://localhost:${PORT}'"