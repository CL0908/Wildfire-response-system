# Moving Firefighter Problem (MFP) Implementation

This repository contains a complete implementation of the Moving Firefighter Problem with MIQCP optimization, comprehensive visualizations, and batch processing capabilities.

## 📦 Installation

Install the required dependencies:

```bash
pip install qiskit==0.44.0 qiskit-optimization[cvx]==0.6.1 qiskit-algorithms==0.3.1
pip install numpy==1.26.1
pip install matplotlib
pip install plotly
pip install pyscipopt
```

## 🚀 Usage

### 1. Problem Generation and Solving (`mff_test.py`)

Run the main script to generate a problem instance, solve it, and create visualizations:

```bash
python mff_test.py
```

This script will:
- Generate a random graph with fire spread
- Calculate defense rounds upper bound (D)
- Solve the problem using MIQCP optimization
- Create comprehensive visualizations
- Save problem and solution data to JSON files
- Generate interactive timeline visualization (if Plotly is available)

### 2. Problem Inference (`mff_inference.py`)

Load and solve saved problem instances from JSON files:

```bash
# Basic usage
python mff_inference.py problem_file.json

# With custom time limit and solver
python mff_inference.py problem_file.json --time-limit 5000 --use-scip

# Quiet mode for batch processing
python mff_inference.py problem_file.json --quiet --no-plots

# Custom output prefix
python mff_inference.py problem_file.json --output-prefix my_solution
```

#### Command-Line Options:

- `--time-limit`: Time limit in seconds (default: 3000)
- `--use-scip` / `--no-scip`: Choose solver (default: SCIP)
- `--no-plots`: Skip visualization (for batch processing)
- `--no-save`: Skip saving solution to JSON
- `--output-prefix`: Prefix for output files
- `--verbose` / `--quiet`: Control output verbosity

### 3. Programmatic Usage

You can also use the scripts as modules in your own code:

```python
# Generate and solve a problem
from mff_test import *

# Load and solve a saved problem
from mff_inference import load_problem_from_json, solve_mff_problem, save_solution_to_json

# Load problem
problem_data, graph, params = load_problem_from_json("problem.json")

# Solve problem
solution_data = solve_mff_problem(graph, params, time_limit=3000, use_scip=True)

# Save solution
save_solution_to_json(problem_data, solution_data, "my_solution")
```

## 📊 Output Files

The scripts generate several types of output files:

### Problem Files:
- `mfp_n{n}_lambda{λ}_b{fires}_{timestamp}_problem.json` - Problem instance data
- `mfp_n{n}_lambda{λ}_b{fires}_{timestamp}_solution.json` - Solution data

### Solution Files:
- `mff_solution_{timestamp}.json` - Inference solution data
- `mff_solution_n{n}_lambda{λ}_{timestamp}.png` - Visualization plots

## 🎯 Features

### Problem Generation (`mff_test.py`):
- ✅ Random graph generation with reproducible seeds
- ✅ Distance matrix calculation with λ multiplier
- ✅ Firefighter starting position selection
- ✅ Defense rounds upper bound calculation
- ✅ MIQCP optimization with SCIP/Gurobi solvers
- ✅ Comprehensive error handling
- ✅ Interactive timeline visualization
- ✅ JSON save/load system

### Problem Inference (`mff_inference.py`):
- ✅ Load problem instances from JSON files
- ✅ Solve with configurable parameters
- ✅ Batch processing capabilities
- ✅ Command-line interface
- ✅ Solution saving and analysis
- ✅ Programmatic API

## 🔧 Configuration

### Problem Parameters:
- `n`: Number of vertices (default: 10)
- `λ`: Distance multiplier (default: 2)
- `burnt_nodes`: Number of initial fires (default: 1)
- `D`: Defense rounds upper bound (calculated)
- `B`: Burning rounds (default: 3)

### Solver Options:
- **SCIP**: Free, open-source solver (default)
- **Gurobi**: Commercial solver (requires license)
- **Fallback**: Automatic switching on license limits

## 📈 Visualization Features

- **Graph Structure**: Distance matrix-based node positioning
- **Firefighter Path**: Complete movement strategy visualization
- **Timeline Analysis**: Defense actions across burning rounds
- **Movement Costs**: Travel time analysis
- **Interactive Timeline**: Time-based simulation with Plotly
- **Solution Summary**: Comprehensive analytics

## 🎨 Color Legend

- 🔴 **Red**: Initially burning vertices
- 🔵 **Blue**: Firefighter current position
- 🟢 **Green**: Defended vertices
- 🟠 **Orange**: Burned during fire spread
- 💙 **Light Blue**: Saved by isolation
- 📍 **Gray**: Safe vertices

## 💡 Examples

### Basic Problem Generation:
```bash
python mff_test.py
```

### Solve Saved Problem:
```bash
python mff_inference.py mfp_n10_lambda2_b1_20240101_120000_problem.json
```

### Batch Processing:
```bash
# Solve multiple problems
for file in *.json; do
    python mff_inference.py "$file" --quiet --no-plots
done
```

### Custom Parameters:
```bash
python mff_inference.py problem.json --time-limit 10000 --use-scip --output-prefix custom_solution
```

## 🔍 Troubleshooting

### Common Issues:

1. **SCIP not available**: Install with `pip install pyscipopt`
2. **Plotly not available**: Install with `pip install plotly`
3. **Gurobi license limits**: Script automatically falls back to SCIP
4. **Memory issues**: Reduce problem size or increase time limits

### Dependencies:
- **Required**: numpy, matplotlib, pyscipopt
- **Optional**: plotly (for interactive visualizations)
- **Optional**: gurobi (for commercial solver)

## 📚 Research Applications

This implementation is suitable for:
- **Algorithm Comparison**: Test different optimization approaches
- **Parameter Studies**: Analyze sensitivity to λ, B, D values
- **Scalability Analysis**: Test performance on larger instances
- **Real-world Applications**: Network protection scenarios
- **Educational Purposes**: Understanding combinatorial optimization

## 🤝 Contributing

Feel free to extend the implementation with:
- Additional solver interfaces
- New visualization types
- Performance optimizations
- Real-world problem instances