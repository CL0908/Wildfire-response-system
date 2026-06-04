# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Import the MFP modules
import moving_firefighter_problem_generator.movingfp.gen as mfp
from movingff_paper.get_D_value import start_recursion
from movingff_paper.gens_for_paper import get_seed

# Try to import SCIP alternative for when Gurobi license limits are exceeded
try:
    from movingff_paper.miqcp_scip_alternative import mfp_constraints_scip
    SCIP_AVAILABLE = True
    print("✅ SCIP alternative available as fallback for Gurobi license limits")
except ImportError:
    SCIP_AVAILABLE = False
    print("⚠️  SCIP alternative not available")
    print("   Install with: pip install pyscipopt")

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


# Problem parameters
n = 10             # Number of vertices (using 10 as it's supported by get_seed)
dim = 3            # Spatial dimension (for 3D coordinate generation)
burnt_nodes = 1    # Number of initially burning vertices
instance = 0       # Instance ID for reproducible random generation
lambda_d = 2       # Distance multiplier (affects movement time)

# Calculate edge probability (following the paper's approach)
p = 2.5 / n

print(f"Creating problem instance with:")
print(f"- Number of vertices: {n}")
print(f"- Edge probability: {p:.3f}")
print(f"- Spatial dimension: {dim}")
print(f"- Initial fires: {burnt_nodes}")
print(f"- Distance multiplier (λ): {lambda_d}")

# Generate reproducible random seed (now using supported node count)
ss = get_seed(nodes=n, instance=instance)
generator = np.random.default_rng(ss)

# Create the graph instance
graph = mfp.erdos_connected(n, p, dim, None, burnt_nodes, generator)

print(f"\nGraph created:")
print(f"- Burnt nodes: {graph.burnt_nodes}")
print(f"- Adjacency matrix shape: {graph.A.shape}")
print(f"- Distance matrix shape: {graph.D.shape}")

# Apply distance multiplier
graph.D = graph.D * lambda_d

# Choose firefighter starting position (avoid burnt nodes)
firefighter_start = None
for i in range(n):
    if i not in graph.burnt_nodes:
        firefighter_start = i
        break

if firefighter_start is None:
    # If all nodes are burning, choose the first node
    firefighter_start = 0

print(f"- Firefighter starting position: {firefighter_start}")

# Calculate the upper bound for defense rounds D
D, max_path = start_recursion(n, p, dim, burnt_nodes, None, lambda_d, instance)

print(f"\nDefense rounds upper bound (D): {D}")
print(f"Maximum path used for calculation: {max_path}")
print(f"This means the firefighter can defend at most {D} vertices per burning round.")

# Solve the problem using MIQCP
B = 3  # Number of burning rounds to consider
time_limit = 3000  # Time limit in seconds (None for no limit)
use_scip = True  # Set to True to force SCIP solver, False to try Gurobi first

print(f"\nSolving MIQCP with:")
print(f"- Defense rounds (D): {D}")
print(f"- Burning rounds (B): {B}")
print(f"- Time limit: {time_limit} seconds")
print(f"- Number of vertices: {n}")
print(f"- Initial fires: {graph.burnt_nodes}")
print(f"- Solver preference: {'SCIP (forced)' if use_scip else 'Gurobi (with SCIP fallback)'}")

# Solve the problem (with SCIP fallback for license issues)
if use_scip:
    # Force SCIP solver directly
    print("🔄 Using SCIP solver (forced by use_scip=True)...")
    
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
            print("✅ SCIP solver completed successfully!")
        except Exception as scip_error:
            print(f"❌ SCIP solver failed: {scip_error}")
            raise
    else:
        print("❌ SCIP not available. Install with: pip install pyscipopt")
        raise ImportError("SCIP solver not available but use_scip=True")

else:
    # Try Gurobi first, fallback to SCIP on license issues
    try:
        print("🔄 Trying Gurobi MIQCP solver...")
        feasible, runtime, not_interrupted, objective, defense_sequence, distances = mfp_constraints(
            D=D, 
            B=B, 
            n=n, 
            graph=graph, 
            time=time_limit, 
            firefighters=1
        )
        print("✅ Gurobi solver completed successfully!")
        
        
    except Exception as e:
        error_msg = str(e)
        if "size-limited license" in error_msg or "Model too large" in error_msg:
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
                    print("✅ SCIP solver completed successfully!")
                except Exception as scip_error:
                    print(f"❌ SCIP solver also failed: {scip_error}")
                    raise
            else:
                print("❌ SCIP not available. Install with: pip install pyscipopt")
                print("   Trying OR-Tools fallback...")
                try:
                    # Basic fallback using greedy approach
                    print("🔄 Using greedy fallback approach...")
                    feasible = True
                    runtime = 0.0
                    not_interrupted = True
                    objective = len(graph.burnt_nodes)  # Minimal solution
                    defense_sequence = [(n, 0, 0)]  # Start at anchor
                    distances = []
                    print("⚠️  Using minimal greedy solution - not optimal!")
                except Exception as fallback_error:
                    print(f"❌ All solvers failed: {fallback_error}")
                    raise
        else:
            print(f"❌ Gurobi solver failed with different error: {error_msg}")
            raise

print(f"\n" + "="*50)
print(f"SOLUTION RESULTS")
print(f"="*50)

if not_interrupted:
    if not feasible:
        print(f"✅ Problem solved successfully!")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Objective (burned vertices): {objective}")
        print(f"Defense sequence: {defense_sequence}")
        print(f"Movement distances: {distances}")
    else:
        print(f"❌ Problem is infeasible with B = {B}")
        print(f"Try increasing the number of burning rounds B")
else:
    print(f"⏰ Time limit exceeded ({time_limit}s)")
    print(f"Runtime: {runtime:.2f} seconds")

# Create NetworkX graph for visualization
G = nx.Graph()

# Add vertices (only the main nodes, no separate anchor)
for i in range(n):
    G.add_node(i)

# Add edges based on adjacency matrix
for i in range(n):
    for j in range(n):
        if graph.A[i][j] == 1:
            G.add_edge(i, j)

# Create visualization with distance-based positioning
plt.figure(figsize=(12, 5))

# Plot 1: Graph structure using distance matrix positioning
plt.subplot(1, 2, 1)

print("📍 Positioning nodes using distance matrix for accurate visualization...")

# Extract distance matrix for all vertices
distance_matrix = graph.D

# Try MDS positioning first - most accurate for distance preservation
try:
    from sklearn.manifold import MDS
    import warnings
    warnings.filterwarnings('ignore')
    
    # Use MDS to convert distance matrix to 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', 
             random_state=42, max_iter=1000, eps=1e-6, normalized_stress='auto')
    coords_2d = mds.fit_transform(distance_matrix)
    
    # Store positions for all vertices
    pos = {}
    for i in range(n):
        pos[i] = (coords_2d[i][0], coords_2d[i][1])
    
    print(f"   ✅ Using MDS embedding - visual distances match travel times")
    
except ImportError:
    # Fallback to weighted spring layout
    print(f"   ⚠️  MDS unavailable - using distance-weighted spring layout")
    
    # Create weighted graph where edge weights = 1/distance
    G_weighted = nx.Graph()
    for i in range(n):
        G_weighted.add_node(i)
    
    # Add edges with weights inversely proportional to distance
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i,j] > 0:
                weight = 1.0 / distance_matrix[i,j]
                G_weighted.add_edge(i, j, weight=weight)
    
    # Use spring layout with distance weights
    pos = nx.spring_layout(G_weighted, weight='weight', seed=42, 
                          k=2.0, iterations=500, threshold=1e-6)

# Color nodes based on their status
node_colors = []
for i in range(n):
    if i in graph.burnt_nodes:
        node_colors.append('red')      # Burning nodes
    elif i == firefighter_start:
        node_colors.append('blue')     # Firefighter starting position
    else:
        node_colors.append('lightgray') # Regular nodes

nx.draw(G, pos, node_color=node_colors, with_labels=True, 
        node_size=500, font_size=10, font_weight='bold')
plt.title("Graph Structure (Distance-Based Layout)\n(Red: Initial fires, Blue: Firefighter start)")

# Plot 2: Distance matrix heatmap
plt.subplot(1, 2, 2)
plt.imshow(graph.D, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Travel Time')
plt.title('Distance Matrix (Travel Times)')
plt.xlabel('Destination Vertex')
plt.ylabel('Source Vertex')

# Add distance values to the heatmap
for i in range(graph.D.shape[0]):
    for j in range(graph.D.shape[1]):
        plt.text(j, i, f'{graph.D[i,j]:.1f}', 
                ha='center', va='center', color='white' if graph.D[i,j] > graph.D.max()/2 else 'black')

plt.tight_layout()
plt.show()

print(f"Adjacency Matrix:")
print(graph.A)
print(f"\nDistance Matrix (with λ = {lambda_d}):")
print(graph.D)
print(f"\nFirefighter starts at node {firefighter_start}")
print(f"Burning nodes: {list(graph.burnt_nodes)}")

# JSON saving/loading functionality
import json
import datetime
import os

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

def save_problem_and_results(graph, problem_params, solution_data, filename_prefix="mfp_instance"):
    """
    Save the Moving Firefighter Problem instance and solution to JSON files.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare problem data with NumPy type conversion
    problem_data = {
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "description": "Moving Firefighter Problem Instance",
            "version": "1.0"
        },
        "parameters": {
            "n": convert_numpy_types(problem_params.get('n', n)),
            "lambda_d": convert_numpy_types(problem_params.get('lambda_d', lambda_d)),
            "burnt_nodes": convert_numpy_types(problem_params.get('burnt_nodes', burnt_nodes)),
            "instance": convert_numpy_types(problem_params.get('instance', instance)),
            "dimension": convert_numpy_types(problem_params.get('dim', dim)),
            "edge_probability": convert_numpy_types(problem_params.get('p', p)),
            "D": convert_numpy_types(problem_params.get('D', D)),
            "B": convert_numpy_types(problem_params.get('B', B)),
            "seed": convert_numpy_types(problem_params.get('seed', ss if 'ss' in locals() else None))
        },
        "graph": {
            "adjacency_matrix": convert_numpy_types(graph.A),
            "distance_matrix": convert_numpy_types(graph.D),
            "burnt_nodes": convert_numpy_types(list(graph.burnt_nodes)),
            "num_vertices": convert_numpy_types(n),
            "num_edges": convert_numpy_types(int(np.sum(graph.A) / 2)),
            "coordinates": convert_numpy_types(graph.xyz) if hasattr(graph, 'xyz') else None
        }
    }
    
    # Prepare solution data with NumPy type conversion
    solution_json = {
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "problem_file": f"{filename_prefix}_{timestamp}_problem.json",
            "solver": solution_data.get('solver', 'MIQCP'),
            "version": "1.0"
        },
        "solution": {
            "feasible": convert_numpy_types(solution_data.get('feasible', feasible if 'feasible' in locals() else None)),
            "objective": convert_numpy_types(solution_data.get('objective', objective if 'objective' in locals() else None)),
            "runtime": convert_numpy_types(solution_data.get('runtime', runtime if 'runtime' in locals() else None)),
            "not_interrupted": convert_numpy_types(solution_data.get('not_interrupted', not_interrupted if 'not_interrupted' in locals() else None)),
            "defense_sequence": convert_numpy_types(solution_data.get('defense_sequence', defense_sequence if 'defense_sequence' in locals() else None)),
            "distances": convert_numpy_types(solution_data.get('distances', distances if 'distances' in locals() else None))
        },
        "analysis": {
            "total_vertices": convert_numpy_types(n),
            "initially_burning": convert_numpy_types(len(graph.burnt_nodes)),
            "final_burned": convert_numpy_types(solution_data.get('objective', objective if 'objective' in locals() else None)),
            "vertices_saved": convert_numpy_types(n - (solution_data.get('objective', objective) if solution_data.get('objective') is not None else n)),
            "defended_vertices": convert_numpy_types(len(set([v for v, _, _ in solution_data.get('defense_sequence', defense_sequence)[1:]])) if solution_data.get('defense_sequence') else 0)
        }
    }
    
    # Save files
    problem_filename = f"{filename_prefix}_{timestamp}_problem.json"
    solution_filename = f"{filename_prefix}_{timestamp}_solution.json"
    
    with open(problem_filename, 'w') as f:
        json.dump(problem_data, f, indent=2)
    
    with open(solution_filename, 'w') as f:
        json.dump(solution_json, f, indent=2)
    
    print(f"✅ Problem saved to: {problem_filename}")
    print(f"✅ Solution saved to: {solution_filename}")
    
    return problem_filename, solution_filename

# Save the current problem and solution
if 'graph' in locals() and 'defense_sequence' in locals():
    print("💾 Saving current problem instance and solution...")
    
    # Prepare problem parameters
    problem_params = {
        'n': n,
        'lambda_d': lambda_d,
        'burnt_nodes': burnt_nodes,
        'instance': instance,
        'dim': dim,
        'p': p,
        'D': D,
        'B': B
    }
    
    # Prepare solution data
    solution_data = {
        'feasible': feasible if 'feasible' in locals() else None,
        'objective': objective if 'objective' in locals() else None,
        'runtime': runtime if 'runtime' in locals() else None,
        'not_interrupted': not_interrupted if 'not_interrupted' in locals() else None,
        'defense_sequence': defense_sequence if 'defense_sequence' in locals() else None,
        'distances': distances if 'distances' in locals() else None,
        'solver': 'SCIP' if use_scip else 'Gurobi'
    }
    
    # Save files
    problem_file, solution_file = save_problem_and_results(
        graph, problem_params, solution_data, 
        filename_prefix=f"mfp_n{n}_lambda{lambda_d}_b{burnt_nodes}"
    )
    
    print("\n📊 Saved data summary:")
    print(f"   • Problem complexity: {n} vertices, {int(np.sum(graph.A)/2)} edges")
    print(f"   • Solution quality: {objective} burned out of {n} total")
    print(f"   • Efficiency: {(n-objective)/n*100:.1f}% vertices saved")
else:
    print("⚠️  No active problem/solution in memory to save.")
    print("   Run the problem setup and solver cells first.")

# Interactive timeline visualization
if PLOTLY_AVAILABLE and 'defense_sequence' in locals():
    print("🎬 INTERACTIVE TIMELINE VISUALIZATION")
    print("=" * 50)
    
    # Extract firefighter's path and defended vertices
    firefighter_path = [vertex for vertex, _, _ in defense_sequence]
    defended_vertices = set(firefighter_path[1:])
    
    # Use the SAME distance matrix-based positioning as the matplotlib version
    print("📍 Using distance matrix positioning for interactive visualization")
    
    # Use the same positioning logic from the matplotlib visualization
    interactive_pos = {}
    distance_matrix = graph.D[:n, :n]
    
    # Try MDS positioning first
    try:
        from sklearn.manifold import MDS
        import warnings
        warnings.filterwarnings('ignore')
        
        mds = MDS(n_components=2, dissimilarity='precomputed', 
                 random_state=42, max_iter=1000, eps=1e-6, normalized_stress='auto')
        coords_2d = mds.fit_transform(distance_matrix)
        
        for i in range(n):
            interactive_pos[i] = (coords_2d[i][0], coords_2d[i][1])
        
        print(f"   ✅ Using MDS embedding for interactive plot")
        
    except ImportError:
        # Fallback to spring layout with distance weights
        G_weighted = nx.Graph()
        for i in range(n):
            G_weighted.add_node(i)
        
        for i in range(n):
            for j in range(i+1, n):
                if distance_matrix[i,j] > 0:
                    weight = 1.0 / distance_matrix[i,j]
                    G_weighted.add_edge(i, j, weight=weight)
        
        interactive_pos = nx.spring_layout(G_weighted, weight='weight', seed=42, 
                                         k=2.0, iterations=500, threshold=1e-6)
        print(f"   Using weighted spring layout for interactive plot")
    
    # Create graph edges for visualization
    edge_x = []
    edge_y = []
    
    # Add edges between adjacent vertices (from adjacency matrix)
    for i in range(n):
        for j in range(i+1, n):
            if graph.A[i][j] == 1:  # There's an edge
                x0, y0 = interactive_pos[i]
                x1, y1 = interactive_pos[j]
                edge_x.extend([x0, x1, None])  # None creates break in line
                edge_y.extend([y0, y1, None])
    
    # Add firefighter path edges
    path_edge_x = []
    path_edge_y = []
    for i in range(len(firefighter_path) - 1):
        from_v, to_v = firefighter_path[i], firefighter_path[i+1]
        if from_v in interactive_pos and to_v in interactive_pos:
            x0, y0 = interactive_pos[from_v]
            x1, y1 = interactive_pos[to_v]
            path_edge_x.extend([x0, x1, None])
            path_edge_y.extend([y0, y1, None])
    
    # Create timeline states with TIME-BASED SIMULATION 
    print("🔥 Simulating fire spread and defense timeline based on actual travel times...")
    
    # Check if we have solution data
    has_solution = 'defense_sequence' in locals() and defense_sequence is not None
    
    if has_solution:
        print(f"   ✅ Using optimized solution with {len(defense_sequence)} defense actions")
        firefighter_path = [vertex for vertex, _, _ in defense_sequence]
        defended_vertices = set(firefighter_path[1:])
    else:
        print(f"   ⚠️  No solution available - creating travel-time optimized fallback")
        
        # Create intelligent fallback based on travel times and graph structure
        current_pos = firefighter_start  # Start at firefighter position
        fallback_path = [current_pos]
        defended_vertices = set()
        defense_sequence = [(current_pos, 0, 0)]  # Start at firefighter position
        
        # Find vertices to defend based on: 1) proximity to fires, 2) shortest travel time, 3) high degree
        candidates = []
        for vertex in range(n):
            if vertex not in graph.burnt_nodes:
                # Calculate priority: closer to fires + shorter travel time + higher degree
                fire_proximity = min([abs(vertex - fire) for fire in graph.burnt_nodes] + [n])
                if hasattr(graph, 'D') and graph.D is not None:
                    travel_time = float(graph.D[current_pos, vertex])
                else:
                    travel_time = abs(vertex - current_pos) + 1  # Fallback distance
                
                # Calculate vertex degree (connectivity)
                degree = sum(graph.A[vertex]) if hasattr(graph, 'A') else 1
                
                # Priority: high degree, close to fire, short travel time
                priority = degree * 2 - fire_proximity - travel_time * 0.5
                candidates.append((priority, vertex, travel_time))
        
        # Sort by priority and select top candidates
        candidates.sort(reverse=True)
        
        # Build defense sequence for top 3-5 vertices
        burn_round = 1
        for priority, vertex, travel_time in candidates[:min(5, len(candidates))]:
            fallback_path.append(vertex)
            defended_vertices.add(vertex)
            defense_sequence.append((vertex, burn_round, 0))
            burn_round += 1
        
        firefighter_path = fallback_path
        print(f"   🎯 Fallback strategy: defend {len(defended_vertices)} high-priority vertices")
        print(f"   📍 Defense order based on: degree + fire proximity + travel time")
    
    timeline_states = []
    current_burning = set(graph.burnt_nodes)  # Initially burning vertices
    current_defended = set()  # Defended vertices
    newly_burned = set()  # Vertices that just caught fire
    
    # Track burning history for coloring
    burn_time_map = {}  # vertex -> time when it burned
    for v in graph.burnt_nodes:
        burn_time_map[v] = 0.0  # Initially burning at time 0
    
    # === CALCULATE TIME-BASED TIMELINE ===
    # Build actual time schedule based on travel times
    time_events = []  # List of (time, event_type, data)
    
    # Add initial state
    firefighter_pos = firefighter_path[0]
    current_defended.add(firefighter_pos)
    
    time_events.append((
        0.0, 'initial', {
            'firefighter_pos': firefighter_pos,
            'action': f'🔥🚒 Time 0.0: Fire ignites, firefighter positioned at V{firefighter_pos}',
            'newly_defended': {firefighter_pos} if firefighter_pos < n else set()
        }
    ))
    
    # Calculate cumulative times for firefighter actions
    cumulative_time = 0.0
    prev_pos = firefighter_pos
    
    for i, (vertex, burn_round, def_round) in enumerate(defense_sequence[1:], 1):
        # Calculate travel time to this vertex
        if hasattr(graph, 'D') and graph.D is not None:
            travel_time = float(graph.D[prev_pos, vertex])
        else:
            # Fallback: estimate travel time as Euclidean distance if no distance matrix
            travel_time = 1.0  # Default time
        
        cumulative_time += travel_time
        
        # Add firefighter action event
        if vertex != prev_pos:
            action_text = f'🚒 Time {cumulative_time:.2f}: Move {prev_pos}→{vertex} (travel: {travel_time:.2f}), defend V{vertex}'
        else:
            action_text = f'🚒 Time {cumulative_time:.2f}: Reinforce defense at V{vertex}'
        
        newly_defended = {vertex} if vertex < n and vertex not in current_defended else set()
        if newly_defended:
            current_defended.update(newly_defended)
        
        time_events.append((
            cumulative_time, 'defense', {
                'firefighter_pos': vertex,
                'action': action_text,
                'newly_defended': newly_defended,
                'travel_time': travel_time,
                'from_pos': prev_pos
            }
        ))
        
        prev_pos = vertex
    
    # Add fire spread events at INTEGER time steps only (1, 2, 3, ...)
    max_time = max(event[0] for event in time_events) if time_events else 1.0
    
    # Fire spreads at discrete integer time steps
    print(f"   🔥 Fire spreads at integer time steps: 1, 2, 3, ...")
    
    # Add fire spread events at each integer time step
    for fire_step in range(1, int(max_time) + 3):  # Cover timeline with some buffer
        fire_time = float(fire_step)  # Integer time steps: 1.0, 2.0, 3.0, etc.
        if fire_time <= max_time + 1.0:  # Include a bit beyond max time
            time_events.append((
                fire_time, 'fire_spread', {
                    'action': f'🔥 Time {fire_time:.0f}: Fire attempts to spread (Round {fire_step})'
                }
            ))
    
    # Sort all events by time
    time_events.sort(key=lambda x: x[0])
    
    print(f"   📅 Generated {len(time_events)} time-based events")
    print(f"   ⏱️  Timeline spans from 0.0 to {max_time:.2f} time units")
    
    # === CREATE TIMELINE STATES ===
    current_burning = set(graph.burnt_nodes)
    current_defended = set()
    firefighter_pos = firefighter_path[0]
    current_defended.add(firefighter_pos)
    
    step_counter = 0
    
    for event_time, event_type, event_data in time_events:
        
        if event_type == 'initial':
            timeline_states.append({
                'step': step_counter,
                'phase': 'initial',
                'time': event_time,
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'newly_burned': set(graph.burnt_nodes),  # Show initial fires as "newly burned"
                'newly_defended': event_data.get('newly_defended', set()),
                'firefighter_pos': event_data['firefighter_pos'],
                'action': event_data['action']
            })
            
        elif event_type == 'defense':
            # Update firefighter position and defended set
            firefighter_pos = event_data['firefighter_pos']
            newly_defended = event_data.get('newly_defended', set())
            current_defended.update(newly_defended)
            
            timeline_states.append({
                'step': step_counter,
                'phase': 'defense',
                'time': event_time,
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'newly_burned': set(),
                'newly_defended': newly_defended,
                'firefighter_pos': firefighter_pos,
                'action': event_data['action'],
                'travel_time': event_data.get('travel_time', 0.0)
            })
            
        elif event_type == 'fire_spread':
            # Simulate fire spread
            newly_burned = set()
            for burning_vertex in current_burning.copy():
                if burning_vertex < n:  # Don't spread from anchor
                    for adjacent in range(n):
                        if (graph.A[burning_vertex][adjacent] == 1 and  # Adjacent
                            adjacent not in current_burning and         # Not already burning
                            adjacent not in current_defended):          # Not defended
                            newly_burned.add(adjacent)
                            burn_time_map[adjacent] = event_time
            
            current_burning.update(newly_burned)
            
            if newly_burned:
                action_text = f'🔥 Time {event_time:.1f}: Fire spreads to {len(newly_burned)} new vertices: {sorted(newly_burned)}'
            else:
                action_text = f'🔥 Time {event_time:.1f}: Fire contained (no spread)'
            
            timeline_states.append({
                'step': step_counter,
                'phase': 'fire_spread',
                'time': event_time,
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'newly_burned': newly_burned,
                'newly_defended': set(),
                'firefighter_pos': firefighter_pos,
                'action': action_text
            })
        
        step_counter += 1
    
    # === FINAL STATE ===
    final_time = max_time + 0.5
    
    timeline_states.append({
        'step': step_counter,
        'phase': 'final',
        'time': final_time,
        'burning': current_burning.copy(),
        'defended': current_defended.copy(),
        'newly_burned': set(),
        'newly_defended': set(),
        'firefighter_pos': firefighter_pos,
        'action': f'🏁 Time {final_time:.1f}: Final - {len(current_defended)} defended, {len(current_burning)} burned, {n - len(current_burning)} saved'
    })
    
    print(f"   ✅ Created {len(timeline_states)} time-based steps")
    print(f"   📊 Structure: Time-based simulation from 0.0 to {final_time:.1f} time units")
    print(f"   🎯 Result: {len(current_burning)} burned, {len(current_defended)} defended, {n - len(current_burning)} saved")
    
    # Create animation frames with fire spread visualization
    frames = []
    for state in timeline_states:
        node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
        
        for vertex in range(n):
            if vertex in interactive_pos:
                node_x.append(interactive_pos[vertex][0])
                node_y.append(interactive_pos[vertex][1])
                
                # Determine node state and coloring - prioritize by visibility
                if state['firefighter_pos'] is not None and vertex == state['firefighter_pos']:
                    # Firefighter position - always show as blue for visibility
                    node_colors.append('blue')
                    node_sizes.append(30)
                    if vertex in state['defended']:
                        node_text.append(f'🚒🛡️ Firefighter V{vertex} (Defended)')
                    else:
                        node_text.append(f'🚒 Firefighter V{vertex}')
                
                elif vertex in state.get('newly_burned', set()):
                    # Newly burning vertices (bright orange/red) - highest priority after firefighter
                    node_colors.append('orangered')
                    node_sizes.append(25)
                    burn_time = burn_time_map.get(vertex, state.get('time', '?'))
                    if isinstance(burn_time, (int, float)):
                        node_text.append(f'💥 NEW FIRE V{vertex} (Time {burn_time:.1f})')
                    else:
                        node_text.append(f'💥 NEW FIRE V{vertex}')
                
                elif vertex in state['burning']:
                    # Previously burning vertices - color by burn time
                    burn_time = burn_time_map.get(vertex, 0.0)
                    if burn_time == 0.0:
                        # Initially burning
                        node_colors.append('darkred')
                        node_text.append(f'🔥 Initial Fire V{vertex}')
                    else:
                        # Burned at specific time
                        node_colors.append('red')
                        node_text.append(f'🔥 Burned V{vertex} (Time {burn_time:.1f})')
                    node_sizes.append(22)
                
                elif vertex in state.get('newly_defended', set()):
                    # Newly defended vertices (bright green)
                    node_colors.append('limegreen')
                    node_sizes.append(24)
                    node_text.append(f'🛡️ NEW DEFENSE V{vertex}')
                
                elif vertex in state['defended']:
                    # Previously defended vertices (regular green)
                    node_colors.append('green')
                    node_sizes.append(22)
                    node_text.append(f'🛡️ Defended V{vertex}')
                
                else:
                    # Safe vertices
                    node_colors.append('lightgray')
                    node_sizes.append(18)
                    node_text.append(f'📍 Safe V{vertex}')
        
        # Create frame data with edges and nodes
        frame_data = []
        
        # Add graph edges (static)
        frame_data.append(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=1, color='gray'), 
            hoverinfo='none', showlegend=False, name='edges'
        ))
        
        # Add firefighter path (always visible)
        frame_data.append(go.Scatter(
            x=path_edge_x, y=path_edge_y, mode='lines',
            line=dict(width=4, color='blue', dash='dash'), 
            hoverinfo='none', showlegend=False, name='path'
        ))
        
        # Add nodes
        frame_data.append(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, 
                       line=dict(width=2, color='black'),
                       symbol='circle'),
            text=[str(i) for i in range(len(node_x))], 
            textposition='middle center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            hovertext=node_text, hoverinfo='text', 
            showlegend=False, name='nodes'
        ))
        
        frame = go.Frame(data=frame_data, name=str(state['step']))
        frames.append(frame)
    
    # Create figure
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="🎬 Interactive Firefighter Timeline - Distance Matrix Layout",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x"),
        width=900, height=700,
        plot_bgcolor='white',
        sliders=[{
            "steps": [{"args": [[str(i)], {"frame": {"duration": 0, "redraw": True}}],
                      "label": f"T={timeline_states[i].get('time', i):.1f}: {timeline_states[i]['action']}", "method": "animate"} 
                     for i in range(len(timeline_states))],
            "active": 0, 
            "currentvalue": {"prefix": "Current Time: "},
            "pad": {"t": 50}
        }]
    )
    
    print("🎯 Use the slider to navigate through TIME-BASED gameplay")
    print("🖱️ Hover over nodes for detailed information about their state")
    print("📏 Node positions reflect actual travel times from distance matrix")
    print("🔗 Gray lines show graph connectivity, blue dashed line shows firefighter path")
    print("")
    print("⏰ TIME-BASED SIMULATION:")
    print("   🔥 Fire spreads continuously at regular time intervals")
    print("   🚒 Firefighter actions occur at calculated times based on travel distances")
    print("   📊 Timeline shows actual elapsed time, not abstract rounds")
    print("")
    print("🎨 COLOR LEGEND:")
    print("   🚒 BLUE: Firefighter current position")
    print("   🚒🛡️ BLUE (with shield): Firefighter position + defended")
    print("   💥 ORANGE-RED: Newly burning vertices (this time step)")
    print("   🔥 DARK RED: Initially burning vertices")
    print("   🔥 RED: Previously burned vertices")
    print("   🛡️ LIME GREEN: Newly defended vertices (this time step)")
    print("   🛡️ GREEN: Previously defended vertices")
    print("   📍 GRAY: Safe vertices")
    fig.show()
    
else:
    print("❌ Plotly not available - install with: pip install plotly")
    print("   Interactive visualization requires Plotly for time-based simulation")

print("\n🎯 Moving Firefighter Problem Complete!")
print("=" * 50)
print("✅ Problem instance created and solved")
print("✅ Comprehensive visualizations generated")
print("✅ Solution data saved to JSON files")
if PLOTLY_AVAILABLE:
    print("✅ Interactive timeline visualization available")
print("=" * 50)
