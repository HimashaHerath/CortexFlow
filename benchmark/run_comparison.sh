#!/bin/bash
# Run benchmark comparison with all implemented systems

# Set Python path to include parent directory
export PYTHONPATH=$(pwd):$(pwd)/..

# Run the path fixer first
echo "Checking and fixing Python path..."
python fix_path.py || exit 1

# Install main dependencies directly in the current environment
echo "Installing requirements..."
pip install -r requirements.txt

# Fix potential dependency conflicts
echo "Installing additional dependencies with specific versions..."
pip install langchain==0.1.1 langchain-community==0.1.1 langchain-core==0.1.15
pip install llama-index==0.9.48 llama-index-core==0.10.39 llama-index-llms-ollama==0.1.1
pip install qdrant-client==1.7.0
pip install deeplake==3.8.3

# Check if llama3.2 model exists in Ollama
if ! ollama list | grep -q "llama3.2"; then
  echo "llama3.2 model not found. Please make sure Ollama has access to llama3.2."
  echo "Continuing with parameter configuration only..."
fi

# Display available systems
echo "Available benchmark systems:"
python -c "from benchmark.registry import get_available_systems; print('\n'.join(get_available_systems()))" || exit 1

# Create benchmark_results directory if it doesn't exist
mkdir -p ../benchmark_results

# Run benchmark with all systems
echo "Running benchmark comparison with all systems..."
python run_all_comparison.py --all-systems --model llama3.2 --output-dir ../benchmark_results --verbose

echo "Benchmark complete. Check benchmark_results directory for results." 