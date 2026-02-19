
import sys
from typing import List, Tuple

def solve_ibsa(
    current_layers: List[int],
    execution_times: List[float],
    threshold_ratio: float = 0.05
) -> Tuple[List[int], bool, int, int]:
    """
    Inter-VG Bottleneck Smoothing Algorithm (IBSA)
    
    Dynamically rebalances layers between adjacent stages (Virtual Groups) to reduce
    the maximum execution time (bottleneck) in the pipeline.
    
    Args:
        current_layers: List of number of layers assigned to each stage.
        execution_times: List of measured execution times for each stage.
        threshold_ratio: Minimum gain ratio required to trigger a rebalance (hysteresis).
        
    Returns:
        Tuple containing:
        - new_layers: The updated layer distribution (or original if no change).
        - changed: Boolean indicating if a change occurred.
        - src_idx: Index of the source stage (bottleneck) from which a layer was moved (-1 if no change).
        - dst_idx: Index of the destination stage to which a layer was moved (-1 if no change).
    """
    
    # 1. Input Validation
    if not current_layers or not execution_times:
        return current_layers, False, -1, -1
        
    if len(current_layers) != len(execution_times):
        print("Error: Sizes of layers and times mismatch.", file=sys.stderr)
        return current_layers, False, -1, -1
        
    n = len(current_layers)
    
    # 2. Calculate Unit Cost (mu) - Average time per layer for each stage
    mu = [0.0] * n
    for i in range(n):
        if current_layers[i] > 0:
            mu[i] = execution_times[i] / current_layers[i]
        else:
            # Fallback: If a stage has 0 layers, we assume 0 cost to add one (based on C++ logic).
            # In a real system, you might want to use a historical average or profile data here.
            mu[i] = 0.0
            
    # 3. Identify Bottleneck
    max_idx = -1
    t_max = -1.0
    
    for i in range(n):
        if execution_times[i] > t_max:
            t_max = execution_times[i]
            max_idx = i
            
    # Defensive check: if no valid max found or max stage has 0 layers (shouldn't happen for max time > 0)
    if max_idx == -1 or current_layers[max_idx] == 0:
        return current_layers, False, -1, -1
        
    # 4. Neighbor Search (Find the faster neighbor)
    target_idx = -1
    min_neighbor_time = float('inf')
    
    # Check left neighbor
    if max_idx > 0:
        left_idx = max_idx - 1
        if execution_times[left_idx] < min_neighbor_time:
            min_neighbor_time = execution_times[left_idx]
            target_idx = left_idx
            
    # Check right neighbor
    if max_idx < n - 1:
        right_idx = max_idx + 1
        # Prefer the emptier/faster neighbor
        if execution_times[right_idx] < min_neighbor_time:
            min_neighbor_time = execution_times[right_idx]
            target_idx = right_idx
            
    # If no suitable neighbor found (e.g., single node) or neighbor is not faster
    if target_idx == -1 or min_neighbor_time >= t_max:
        return current_layers, False, -1, -1
        
    # 5. Simulation & Gain Calculation
    # Simulate moving 1 layer: max_idx -> target_idx
    
    # Predicted time for source after removing 1 layer
    t_src_predicted = t_max - mu[max_idx]
    
    # Predicted time for target after adding 1 layer
    # Note: Using target's current average cost. If target has 0 layers, mu is 0, 
    # which assumes adding a layer is free. This is a simplification from the C++ algorithm.
    t_target_predicted = execution_times[target_idx] + mu[target_idx]
    
    # New local bottleneck between these two
    t_bottleneck_predicted = max(t_src_predicted, t_target_predicted)
    
    gain = t_max - t_bottleneck_predicted
    
    # 6. Hysteresis Check
    # Only change if gain is significant enough to avoid oscillation
    required_gain = t_max * threshold_ratio
    
    if gain > required_gain:
        new_layers = list(current_layers)
        new_layers[max_idx] -= 1
        new_layers[target_idx] += 1
        return new_layers, True, max_idx, target_idx
        
    return current_layers, False, -1, -1

# --- Example Usage ---
if __name__ == "__main__":
    # Example scenario: 3 stages pipeline
    # Stage 1 is the bottleneck (high execution time)
    current_layers = [10, 10, 10]
    execution_times = [100.0, 200.0, 120.0] # ms
    
    print("Initial State:")
    print(f"  Layers: {current_layers}")
    print(f"  Times:  {execution_times}")
    
    # Run IBSA
    new_layers, changed, src, dst = solve_ibsa(current_layers, execution_times, threshold_ratio=0.05)
    
    if changed:
        print("\nOptimization Applied:")
        print(f"  Moved 1 layer from Stage {src} to Stage {dst}")
        print(f"  New Layers: {new_layers}")
    else:
        print("\nNo optimization applied (gain too small or no valid move).")
