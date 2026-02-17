import math
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DeviceStatus:
    device_id: int
    execution_time_ms: float
    # Active computing range for Heads [start, end)
    current_head_start: int
    current_head_end: int
    # Active computing range for FFN [start, end)
    current_ffn_start: int
    current_ffn_end: int
    # KV Cache holding range for Heads [start, end)
    kv_head_holding_start: int
    kv_head_holding_end: int
    # FFN holding range? Assuming FFN is stateless or weights are movable/replicated
    # If FFN weights are sharded, we might need holding constraints too.
    # For now, let's assume FFN can be moved freely (e.g. all-gather weights or weight-streaming?)
    # Or more likely, user implies standard TP rebalancing where weights are static?
    # If weights are static, we CANNOT move FFN.
    # User says "re-partition heads and FFN".
    # This implies we CAN move FFN computation.
    # So either weights are moved, or duplicated.

    @property
    def num_active_heads(self):
        return self.current_head_end - self.current_head_start

    @property
    def num_active_ffn(self):
        return self.current_ffn_end - self.current_ffn_start

def find_bottleneck_device_index(devices: List[DeviceStatus]) -> int:
    src_idx = -1
    max_time = -1.0
    for i, dev in enumerate(devices):
        if dev.execution_time_ms > max_time:
            max_time = dev.execution_time_ms
            src_idx = i
    return src_idx

def rebalance_intra_stage(devices: List[DeviceStatus]) -> tuple[List[int], List[int]]:
    """
    Implements the intra-stage rebalancing algorithm for Heads and FFN.
    Returns (new_head_counts, new_ffn_counts).
    """
    # Create copies of ranges
    new_head_ranges = [[d.current_head_start, d.current_head_end] for d in devices]
    new_ffn_ranges = [[d.current_ffn_start, d.current_ffn_end] for d in devices]
        
    if not devices:
        return [], []
        
    # 1. Find Bottleneck (T_max)
    src_idx = find_bottleneck_device_index(devices)
    if src_idx == -1:
        return ([r[1] - r[0] for r in new_head_ranges], [r[1] - r[0] for r in new_ffn_ranges])
        
    src_dev = devices[src_idx]
    current_heads = src_dev.num_active_heads
    current_ffn = src_dev.num_active_ffn
    
    # Must keep at least 1 head
    if current_heads <= 1:
        return ([r[1] - r[0] for r in new_head_ranges], [r[1] - r[0] for r in new_ffn_ranges])
        
    t_src_unit = src_dev.execution_time_ms / current_heads # Simplified unit cost metric
    max_movable_heads = current_heads - 1
    
    move_left = 0
    move_right = 0
    
    # --- Calculate Move to Left (Heads) ---
    if src_idx > 0:
        left_dev = devices[src_idx - 1]
        l_heads = left_dev.num_active_heads
        t_left_unit = (left_dev.execution_time_ms / l_heads) if l_heads > 0 else t_src_unit
        
        if src_dev.execution_time_ms > left_dev.execution_time_ms:
            delta_raw = (src_dev.execution_time_ms - left_dev.execution_time_ms) / (t_src_unit + t_left_unit)
            move_left = int(math.floor(delta_raw))
            
    # --- Calculate Move to Right (Heads) ---
    if src_idx < len(devices) - 1:
        right_dev = devices[src_idx + 1]
        r_heads = right_dev.num_active_heads
        t_right_unit = (right_dev.execution_time_ms / r_heads) if r_heads > 0 else t_src_unit
        
        if src_dev.execution_time_ms > right_dev.execution_time_ms:
            delta_raw = (src_dev.execution_time_ms - right_dev.execution_time_ms) / (t_src_unit + t_right_unit)
            move_right = int(math.floor(delta_raw))
            
    # Apply limit based on movable heads
    if move_left + move_right > max_movable_heads:
        total_req = float(move_left + move_right)
        ratio = float(max_movable_heads) / total_req
        
        new_move_left = int(move_left * ratio)
        new_move_right = int(move_right * ratio)
        
        if new_move_left + new_move_right > max_movable_heads:
            if new_move_left > 0:
                new_move_left -= 1
            elif new_move_right > 0:
                new_move_right -= 1
                
        move_left = new_move_left
        move_right = new_move_right

    # Calculate FFN moves based on Head ratio
    ffn_move_left = 0
    ffn_move_right = 0
    if current_heads > 0:
        ratio_left = move_left / current_heads
        ratio_right = move_right / current_heads
        ffn_move_left = int(current_ffn * ratio_left)
        ffn_move_right = int(current_ffn * ratio_right)

    # 3. Apply KV-Cache Constraints (Heads Only)
    # FFN typically doesn't depend on KV cache, so we only constrain heads
    
    # Move Left Heads
    if move_left > 0 and src_idx > 0:
        left_dev = devices[src_idx - 1]
        current_L_end = new_head_ranges[src_idx - 1][1] 
        max_possible = left_dev.kv_head_holding_end - current_L_end
        if max_possible < 0: max_possible = 0
        
        if move_left > max_possible:
            move_left = max_possible
            # If we constrain heads, should we constrain FFN proportionally?
            # Yes, to maintain balance logic.
            # Recalculate FFN move
            if current_heads > 0:
                ratio_left = move_left / current_heads
                ffn_move_left = int(current_ffn * ratio_left)
        
        if move_left > 0:
            new_head_ranges[src_idx - 1][1] += move_left
            new_head_ranges[src_idx][0] += move_left
            
    # Move Left FFN
    if ffn_move_left > 0 and src_idx > 0:
        new_ffn_ranges[src_idx - 1][1] += ffn_move_left
        new_ffn_ranges[src_idx][0] += ffn_move_left

    # Move Right Heads
    if move_right > 0 and src_idx < len(devices) - 1:
        right_dev = devices[src_idx + 1]
        current_R_start = new_head_ranges[src_idx + 1][0]
        max_possible = current_R_start - right_dev.kv_head_holding_start
        if max_possible < 0: max_possible = 0
        
        if move_right > max_possible:
            move_right = max_possible
            if current_heads > 0:
                ratio_right = move_right / current_heads
                ffn_move_right = int(current_ffn * ratio_right)
            
        if move_right > 0:
            new_head_ranges[src_idx][1] -= move_right
            new_head_ranges[src_idx + 1][0] -= move_right
            
    # Move Right FFN
    if ffn_move_right > 0 and src_idx < len(devices) - 1:
        new_ffn_ranges[src_idx][1] -= ffn_move_right
        new_ffn_ranges[src_idx + 1][0] -= ffn_move_right
            
    # Return new counts
    head_counts = [r[1] - r[0] for r in new_head_ranges]
    ffn_counts = [r[1] - r[0] for r in new_ffn_ranges]
    return head_counts, ffn_counts
