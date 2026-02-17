import math
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DeviceStatus:
    device_id: int
    execution_time_ms: float
    # Active computing range [start, end)
    current_layer_start: int
    current_layer_end: int
    # KV Cache holding range [start, end)
    kv_holding_start: int
    kv_holding_end: int

    @property
    def num_active_layers(self):
        return self.current_layer_end - self.current_layer_start

def find_bottleneck_device_index(devices: List[DeviceStatus]) -> int:
    src_idx = -1
    max_time = -1.0
    for i, dev in enumerate(devices):
        if dev.execution_time_ms > max_time:
            max_time = dev.execution_time_ms
            src_idx = i
    return src_idx

def rebalance_layers(devices: List[DeviceStatus]) -> List[int]:
    """
    Implements the layer rebalancing algorithm.
    Returns a list of new layer counts for each device.
    """
    # Create a copy of ranges to modify
    new_ranges = []
    for dev in devices:
        new_ranges.append([dev.current_layer_start, dev.current_layer_end])
        
    if not devices:
        return []
        
    # 1. Find Bottleneck (T_max)
    src_idx = find_bottleneck_device_index(devices)
    if src_idx == -1:
        return [r[1] - r[0] for r in new_ranges]
        
    src_dev = devices[src_idx]
    current_layers = src_dev.num_active_layers
    
    # Must keep at least 1 layer
    if current_layers <= 1:
        return [r[1] - r[0] for r in new_ranges]
        
    t_src_unit = src_dev.execution_time_ms / current_layers
    max_movable = current_layers - 1
    
    move_left = 0
    move_right = 0
    
    # --- Calculate Move to Left ---
    if src_idx > 0:
        left_dev = devices[src_idx - 1]
        l_layers = left_dev.num_active_layers
        t_left_unit = (left_dev.execution_time_ms / l_layers) if l_layers > 0 else t_src_unit
        
        if src_dev.execution_time_ms > left_dev.execution_time_ms:
            # delta = (T_src - T_L) / (t_src + t_L)
            delta_raw = (src_dev.execution_time_ms - left_dev.execution_time_ms) / (t_src_unit + t_left_unit)
            move_left = int(math.floor(delta_raw))
            
    # --- Calculate Move to Right ---
    if src_idx < len(devices) - 1:
        right_dev = devices[src_idx + 1]
        r_layers = right_dev.num_active_layers
        t_right_unit = (right_dev.execution_time_ms / r_layers) if r_layers > 0 else t_src_unit
        
        if src_dev.execution_time_ms > right_dev.execution_time_ms:
            delta_raw = (src_dev.execution_time_ms - right_dev.execution_time_ms) / (t_src_unit + t_right_unit)
            move_right = int(math.floor(delta_raw))
            
    # Apply limit based on movable layers
    if move_left + move_right > max_movable:
        total_req = float(move_left + move_right)
        ratio = float(max_movable) / total_req
        
        new_move_left = int(move_left * ratio)
        new_move_right = int(move_right * ratio)
        
        # Rounding fix
        if new_move_left + new_move_right > max_movable:
            if new_move_left > 0:
                new_move_left -= 1
            elif new_move_right > 0:
                new_move_right -= 1
                
        move_left = new_move_left
        move_right = new_move_right

    # 3. Apply KV-Cache Constraints (This must happen AFTER proportional scaling)
    # The C++ code applies constraints at the end, AFTER scaling.
    # But wait, C++ code applies scaling (Step 2) THEN KV constraints (Step 3).
    # This matches my order.
    
    # Move Left
    if move_left > 0 and src_idx > 0:
        left_dev = devices[src_idx - 1]
        # In Python slicing, range [start, end).
        # Src active: [Src.start, Src.end).
        # Left active: [Left.start, Left.end).
        # If we move Left, Left active end increases.
        # It takes [Src.start, Src.start + k).
        # So new Left active end = Left.end + k.
        
        current_L_end = new_ranges[src_idx - 1][1] 
        
        # However, the C++ code says:
        # max_possible = left_kv.end - current_L_end;
        # Left KV holding: [Left.kv_start, Left.kv_end).
        # So yes, max active end is kv_end.
        
        # BUT WAIT: Is the Left neighbor CONTIGUOUS with Src?
        # In PP, usually yes. Stage 0: 0-12, Stage 1: 12-24.
        # Left.end == Src.start.
        # So if we move k layers from Src to Left, Left end becomes Src.start + k.
        # This is correct.
        
        max_possible = left_dev.kv_holding_end - current_L_end
        if max_possible < 0: max_possible = 0
        
        if move_left > max_possible:
            move_left = max_possible
        
        if move_left > 0:
            # Update ranges
            new_ranges[src_idx - 1][1] += move_left
            new_ranges[src_idx][0] += move_left

    # Move Right
    if move_right > 0 and src_idx < len(devices) - 1:
        right_dev = devices[src_idx + 1]
        current_R_start = new_ranges[src_idx + 1][0]
        
        # Right KV holding: [Right.kv_start, Right.kv_end).
        # We want to decrease Right.start.
        # New Right.start = Right.start - k.
        # Constraint: Right.start - k >= Right.kv_start.
        
        max_possible = current_R_start - right_dev.kv_holding_start
        if max_possible < 0: max_possible = 0
        
        if move_right > max_possible:
            move_right = max_possible
            
        if move_right > 0:
            # Update ranges
            new_ranges[src_idx][1] -= move_right
            new_ranges[src_idx + 1][0] -= move_right

            
    # Return new counts
    return [r[1] - r[0] for r in new_ranges]
