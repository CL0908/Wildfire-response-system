# Moving Firefighter Problem Inference Script
# Loads problem instances from JSON files and solves them

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import datetime
import os
import types
import argparse
from pathlib import Path

# Import the MFP modules
try:
    import moving_firefighter_problem_generator.movingfp.gen as mfp
    from movingff_paper.get_D_value import start_recursion
    from movingff_paper.gens_for_paper import get_seed
    from movingff_paper.miqcp_scip_alternative import mfp_constraints_scip
    SCIP_AVAILABLE = True
    print("✅ SCIP alternative available")
except ImportError as e:
    print(f"⚠️  Some MFP modules not available: {e}")
    SCIP_AVAILABLE = False

# For interactive visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj


def load_problem_from_json(problem_file):
    """
    Load a Moving Firefighter Problem instance from JSON file.
    
    Parameters:
    - problem_file: Path to the problem JSON file
    
    Returns:
    - problem_data: Dictionary containing the problem instance
    - graph: Recreated graph object
    """
    print(f"📂 Loading problem from: {problem_file}")
    
    with open(problem_file, 'r') as f:
        problem_data = json.load(f)
    
    # Recreate graph object
    graph = types.SimpleNamespace()
    graph.A = np.array(problem_data['graph']['adjacency_matrix'])
    graph.D = np.array(problem_data['graph']['distance_matrix'])
    graph.burnt_nodes = set(problem_data['graph']['burnt_nodes'])
    
    if problem_data['graph'].get('coordinates'):
        graph.xyz = np.array(problem_data['graph']['coordinates'])
    
    # Extract parameters
    params = problem_data['parameters']
    n = params['n']
    lambda_d = params['lambda_d']
    burnt_nodes = params['burnt_nodes']
    instance = params['instance']
    dim = params['dimension']
    p = params['edge_probability']
    D = params['D']
    B = params['B']
    
    print(f"✅ Problem loaded successfully!")
    print(f"   • Vertices: {n}")
    print(f"   • Initial fires: {list(graph.burnt_nodes)}")
    print(f"   • Lambda: {lambda_d}")
    print(f"   • Defense rounds (D): {D}")
    print(f"   • Burning rounds (B): {B}")
    print(f"   • Edges: {int(np.sum(graph.A)/2)}")
    
    return problem_data, graph, params


def solve_mff_problem(graph, params, time_limit=3000, use_scip=True, verbose=True):
    """
    Solve the Moving Firefighter Problem using MIQCP.
    
    Parameters:
    - graph: Graph object with A, D, and burnt_nodes attributes
    - params: Dictionary of problem parameters
    - time_limit: Time limit in seconds
    - use_scip: Whether to use SCIP solver
    - verbose: Whether to print detailed output
    
    Returns:
    - solution_data: Dictionary containing solution results
    """
    n = params['n']
    D = params['D']
    B = params['B']
    
    if verbose:
        print(f"\n🔧 SOLVING MOVING FIREFIGHTER PROBLEM")
        print("=" * 50)
        print(f"Parameters:")
        print(f"  • Vertices: {n}")
        print(f"  • Defense rounds (D): {D}")
        print(f"  • Burning rounds (B): {B}")
        print(f"  • Initial fires: {list(graph.burnt_nodes)}")
        print(f"  • Time limit: {time_limit} seconds")
        print(f"  • Solver: {'SCIP' if use_scip else 'Gurobi'}")
    
    # Choose firefighter starting position (avoid burnt nodes)
    firefighter_start = None
    for i in range(n):
        if i not in graph.burnt_nodes:
            firefighter_start = i
            break
    
    if firefighter_start is None:
        firefighter_start = 0
    
    if verbose:
        print(f"  • Firefighter starting position: {firefighter_start}")
    
    # Solve the problem
    if use_scip:
        if verbose:
            print("🔄 Using SCIP solver...")
        
        if SCIP_AVAILABLE:
            try:
                feasible, runtime, not_interrupted, objective, defense_sequence, distances = mfp_constraints_scip(
                    D=D, 
                    B=B, 
                    n=n, 
                    graph=graph, 
                    time=time_limit, 
                    firefighters=1
                )
                if verbose:
                    print("✅ SCIP solver completed successfully!")
            except Exception as scip_error:
                if verbose:
                    print(f"❌ SCIP solver failed: {scip_error}")
                raise
        else:
            if verbose:
                print("❌ SCIP not available. Install with: pip install pyscipopt")
            raise ImportError("SCIP solver not available but use_scip=True")
    else:
        # Try Gurobi first, fallback to SCIP on license issues
        try:
            if verbose:
                print("🔄 Trying Gurobi MIQCP solver...")
            
            # Import Gurobi solver
            from movingff_paper.miqcp import mfp_constraints
            
            feasible, runtime, not_interrupted, objective, defense_sequence, distances = mfp_constraints(
                D=D, 
                B=B, 
                n=n, 
                graph=graph, 
                time=time_limit, 
                firefighters=1
            )
            if verbose:
                print("✅ Gurobi solver completed successfully!")
                
        except Exception as e:
            error_msg = str(e)
            if "size-limited license" in error_msg or "Model too large" in error_msg:
                if verbose:
                    print("❌ Gurobi license limit exceeded!")
                    print(f"   Error: {error_msg}")
                    print("🔄 Switching to free SCIP solver...")
                
                if SCIP_AVAILABLE:
                    try:
                        feasible, runtime, not_interrupted, objective, defense_sequence, distances = mfp_constraints_scip(
                            D=D, 
                            B=B, 
                            n=n, 
                            graph=graph, 
                            time=time_limit, 
                            firefighters=1
                        )
                        if verbose:
                            print("✅ SCIP solver completed successfully!")
                    except Exception as scip_error:
                        if verbose:
                            print(f"❌ SCIP solver also failed: {scip_error}")
                        raise
                else:
                    if verbose:
                        print("❌ SCIP not available. Install with: pip install pyscipopt")
                    raise ImportError("SCIP solver not available")
            else:
                if verbose:
                    print(f"❌ Gurobi solver failed with different error: {error_msg}")
                raise
    
    # Prepare solution data
    solution_data = {
        'feasible': feasible,
        'objective': objective,
        'runtime': runtime,
        'not_interrupted': not_interrupted,
        'defense_sequence': defense_sequence,
        'distances': distances,
        'solver': 'SCIP' if use_scip else 'Gurobi',
        'firefighter_start': firefighter_start
    }
    
    if verbose:
        print(f"\n" + "="*50)
        print(f"SOLUTION RESULTS")
        print(f"="*50)
        
        if not_interrupted:
            if not feasible:
                print(f"✅ Problem solved successfully!")
                print(f"Runtime: {runtime:.2f} seconds")
                print(f"Objective (burned vertices): {objective}")
                print(f"Defense sequence length: {len(defense_sequence)}")
                print(f"Firefighter path: {[v for v, _, _ in defense_sequence]}")
            else:
                print(f"❌ Problem is infeasible with B = {B}")
                print(f"Try increasing the number of burning rounds B")
        else:
            print(f"⏰ Time limit exceeded ({time_limit}s)")
            print(f"Runtime: {runtime:.2f} seconds")
    
    return solution_data


def save_solution_to_json(problem_data, solution_data, output_prefix="mff_solution"):
    """
    Save the solution to a JSON file.
    
    Parameters:
    - problem_data: Original problem data
    - solution_data: Solution results
    - output_prefix: Prefix for output filename
    
    Returns:
    - solution_filename: Name of the saved solution file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare solution data with NumPy type conversion
    solution_json = {
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "problem_file": problem_data.get('metadata', {}).get('created_at', 'unknown'),
            "solver": solution_data.get('solver', 'MIQCP'),
            "version": "1.0"
        },
        "solution": {
            "feasible": convert_numpy_types(solution_data.get('feasible')),
            "objective": convert_numpy_types(solution_data.get('objective')),
            "runtime": convert_numpy_types(solution_data.get('runtime')),
            "not_interrupted": convert_numpy_types(solution_data.get('not_interrupted')),
            "defense_sequence": convert_numpy_types(solution_data.get('defense_sequence')),
            "distances": convert_numpy_types(solution_data.get('distances')),
            "firefighter_start": convert_numpy_types(solution_data.get('firefighter_start'))
        },
        "analysis": {
            "total_vertices": convert_numpy_types(problem_data['parameters']['n']),
            "initially_burning": convert_numpy_types(len(problem_data['graph']['burnt_nodes'])),
            "final_burned": convert_numpy_types(solution_data.get('objective')),
            "vertices_saved": convert_numpy_types(problem_data['parameters']['n'] - (solution_data.get('objective') if solution_data.get('objective') is not None else problem_data['parameters']['n'])),
            "defended_vertices": convert_numpy_types(len(set([v for v, _, _ in solution_data.get('defense_sequence', [])[1:]])) if solution_data.get('defense_sequence') else 0)
        }
    }
    
    # Save file
    solution_filename = f"{output_prefix}_{timestamp}.json"
    
    with open(solution_filename, 'w') as f:
        json.dump(solution_json, f, indent=2)
    
    print(f"✅ Solution saved to: {solution_filename}")
    
    return solution_filename


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Solve Moving Firefighter Problem from JSON file')
    parser.add_argument('problem_file', help='Path to the problem JSON file')
    parser.add_argument('--time-limit', type=int, default=3000, help='Time limit in seconds (default: 3000)')
    parser.add_argument('--use-scip', action='store_true', default=True, help='Use SCIP solver (default: True)')
    parser.add_argument('--no-scip', dest='use_scip', action='store_false', help='Use Gurobi solver instead')
    parser.add_argument('--no-plots', action='store_true', help='Skip visualization')
    parser.add_argument('--no-save', action='store_true', help='Skip saving solution to JSON')
    parser.add_argument('--output-prefix', default='mff_solution', help='Prefix for output files')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print detailed output')
    parser.add_argument('--quiet', dest='verbose', action='store_false', help='Minimal output')
    
    args = parser.parse_args()
    
    # Check if problem file exists
    if not os.path.exists(args.problem_file):
        print(f"❌ Problem file not found: {args.problem_file}")
        return 1
    
    try:
        # Load problem
        problem_data, graph, params = load_problem_from_json(args.problem_file)
        
        # Solve problem
        solution_data = solve_mff_problem(
            graph, params, 
            time_limit=args.time_limit, 
            use_scip=args.use_scip, 
            verbose=args.verbose
        )
        
        # Save solution
        if not args.no_save:
            save_solution_to_json(problem_data, solution_data, args.output_prefix)
        
        print(f"\n🎯 Moving Firefighter Problem Inference Complete!")
        print("=" * 50)
        print("✅ Problem loaded and solved successfully")
        if not args.no_save:
            print("✅ Solution saved to JSON")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    # Example usage as a module
    if len(sys.argv) == 1:
        print("🎯 Moving Firefighter Problem Inference Script")
        print("=" * 50)
        print("Usage:")
        print("  python mff_inference.py <problem_file.json>")
        print("  python mff_inference.py <problem_file.json> --time-limit 5000 --use-scip")
        print("  python mff_inference.py <problem_file.json> --no-plots --quiet")
        print("\nExample:")
        print("  python mff_inference.py mfp_n10_lambda2_b1_20240101_120000_problem.json")
        print("=" * 50)
    else:
        import sys
        sys.exit(main()) 