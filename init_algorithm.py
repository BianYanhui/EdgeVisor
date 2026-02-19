import math
import sys
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import itertools

# --- Data Structures ---

@dataclass
class Device:
    id: int
    compute: float  # TFLOPS or similar unit
    memory: float   # GB
    bandwidth_bps: float = 0.0 # For internal use in CCWF

@dataclass
class Link:
    src_id: int
    dst_id: int
    bandwidth: float # Gbps

@dataclass
class RRAGCConfig:
    K: int # Number of stages (VGs)
    P_min: float # Min compute per VG
    M_min: float # Min memory per VG
    alpha: float = 0.5 # Weight for bandwidth in root selection
    beta: float = 0.5  # Weight for compute in root selection

@dataclass
class ModelConfig:
    total_layers: int
    activation_size_gb: float # Size of activation to transfer between stages

@dataclass
class LayerTask:
    input_bytes: float
    output_bytes: float
    total_flops: float

@dataclass
class RRAGCResult:
    vg_roots: List[int] = field(default_factory=list)
    device_to_vg_map: Dict[int, int] = field(default_factory=dict)
    pipeline_order: List[int] = field(default_factory=list)

@dataclass
class AllocationResult:
    alphas: List[float]
    estimated_latency: float # seconds

@dataclass
class InitResult:
    rragc_result: RRAGCResult
    intravg_allocations: Dict[int, AllocationResult] = field(default_factory=dict)
    intervg_layer_allocation: List[int] = field(default_factory=list)

@dataclass
class WorkerProfile:
    dev_id: int
    compute_flops: float
    bandwidth_bps: float
    max_alpha_mem: float

@dataclass
class VGProfile:
    vg_id: int
    unit_time_ms: float # Time to process 1 layer (estimated)
    max_layers_capacity: int
    next_link_bw_gbps: float

# --- Helper Functions ---

def get_bandwidth(links: List[Link], src: int, dst: int) -> float:
    for l in links:
        if l.src_id == src and l.dst_id == dst:
            return l.bandwidth
    return 0.0

def get_connection_strength(adj: Dict[int, Dict[int, float]], u: int, v: int) -> float:
    # Max of u->v and v->u
    bw_uv = adj.get(u, {}).get(v, 0.0)
    bw_vu = adj.get(v, {}).get(u, 0.0)
    return max(bw_uv, bw_vu)

# --- Algorithms ---

class RRAGC:
    @staticmethod
    def solve(devices: List[Device], links: List[Link], config: RRAGCConfig) -> RRAGCResult:
        result = RRAGCResult()
        if not devices:
            return result
        
        N = len(devices)
        K = min(config.K, N)
        
        # Build Adjacency Matrix
        adj: Dict[int, Dict[int, float]] = {}
        global_max_link_bw = 0.0
        for l in links:
            if l.src_id not in adj: adj[l.src_id] = {}
            adj[l.src_id][l.dst_id] = l.bandwidth
            global_max_link_bw = max(global_max_link_bw, l.bandwidth)
            
        device_map = {d.id: d for d in devices}
        
        # =========================================================
        # Phase 1: Connectivity-Penalized Anchor Identification
        # =========================================================
        
        # 1. Data Aggregation
        total_bw: Dict[int, float] = {}
        max_bw = -1.0
        min_bw = float('inf')
        max_c = -1.0
        min_c = float('inf')
        
        for d in devices:
            bw_sum = 0.0
            # Outgoing
            if d.id in adj:
                bw_sum += sum(adj[d.id].values())
            # Incoming
            for l in links:
                if l.dst_id == d.id:
                    bw_sum += l.bandwidth
            
            total_bw[d.id] = bw_sum
            max_bw = max(max_bw, bw_sum)
            min_bw = min(min_bw, bw_sum)
            max_c = max(max_c, d.compute)
            min_c = min(min_c, d.compute)
            
        if min_bw > max_bw: min_bw = max_bw = 0
        if min_c > max_c: min_c = max_c = 0
        
        # 2 & 3. Normalize and Calculate Raw Score
        score_raw: Dict[int, float] = {}
        for d in devices:
            b_hat = 1.0 if abs(max_bw - min_bw) < 1e-9 else (total_bw[d.id] - min_bw) / (max_bw - min_bw)
            c_hat = 1.0 if abs(max_c - min_c) < 1e-9 else (d.compute - min_c) / (max_c - min_c)
            score_raw[d.id] = config.alpha * b_hat + config.beta * c_hat
            
        # 4. Iteratively Select Roots
        selected_roots: List[int] = []
        selected_roots_set: Set[int] = set()
        
        for _ in range(K):
            best_candidate = -1
            best_adj_score = -1.0
            
            for d in devices:
                if d.id in selected_roots_set:
                    continue
                
                bw_link_u = 0.0
                if selected_roots:
                    for root_id in selected_roots:
                        bw = get_connection_strength(adj, d.id, root_id)
                        bw_link_u = max(bw_link_u, bw)
                
                rho_u = (bw_link_u / global_max_link_bw) if global_max_link_bw > 1e-9 else 0.0
                score_adj = score_raw[d.id] * (1.0 - rho_u)
                
                if score_adj > best_adj_score:
                    best_adj_score = score_adj
                    best_candidate = d.id
            
            if best_candidate != -1:
                selected_roots.append(best_candidate)
                selected_roots_set.add(best_candidate)
            else:
                break
        
        result.vg_roots = selected_roots
        K = len(selected_roots) # Update K if fewer roots found
        
        # =========================================================
        # Phase 2: Star Topology Clustering
        # =========================================================
        
        vg_members: Dict[int, List[int]] = {i: [] for i in range(K)}
        dev_to_vg: Dict[int, int] = {}
        
        # Assign roots
        for i, root_id in enumerate(selected_roots):
            vg_members[i].append(root_id)
            dev_to_vg[root_id] = i
            
        # Assign workers
        isolated_nodes: List[int] = []
        for d in devices:
            if d.id in selected_roots_set:
                continue
            
            best_vg = -1
            max_bw_to_root = -1.0
            
            for i in range(K):
                root_id = selected_roots[i]
                bw = get_connection_strength(adj, d.id, root_id)
                if bw > max_bw_to_root:
                    max_bw_to_root = bw
                    best_vg = i
            
            if max_bw_to_root > 0.0:
                vg_members[best_vg].append(d.id)
                dev_to_vg[d.id] = best_vg
            else:
                isolated_nodes.append(d.id)
                
        # Handle isolated nodes (assign to min compute VG)
        for iso_id in isolated_nodes:
            target_vg = -1
            min_vg_compute = float('inf')
            
            for i in range(K):
                current_vg_compute = sum(device_map[m_id].compute for m_id in vg_members[i])
                if current_vg_compute < min_vg_compute:
                    min_vg_compute = current_vg_compute
                    target_vg = i
            
            if target_vg != -1:
                vg_members[target_vg].append(iso_id)
                dev_to_vg[iso_id] = target_vg
                
        # =========================================================
        # Phase 3: Minimum-Sacrifice Viability Enforcement
        # =========================================================
        
        optimization_active = True
        max_iterations = N * 2
        iter_count = 0
        
        while optimization_active and iter_count < max_iterations:
            optimization_active = False
            iter_count += 1
            
            # Calculate stats
            vg_stats = []
            critical_vgs = []
            for i in range(K):
                c_sum = sum(device_map[m_id].compute for m_id in vg_members[i])
                m_sum = sum(device_map[m_id].memory for m_id in vg_members[i])
                is_critical = (c_sum < config.P_min or m_sum < config.M_min)
                vg_stats.append({'compute': c_sum, 'memory': m_sum, 'is_critical': is_critical})
                if is_critical:
                    critical_vgs.append(i)
            
            if not critical_vgs:
                break
                
            for weak_vg_idx in critical_vgs:
                best_donor_node = -1
                best_donor_vg = -1
                min_delta_loss = float('inf')
                found_candidate = False
                
                weak_root = selected_roots[weak_vg_idx]
                
                for neighbor_vg_idx in range(K):
                    if neighbor_vg_idx == weak_vg_idx:
                        continue
                    if vg_stats[neighbor_vg_idx]['is_critical']:
                        continue
                        
                    for w_id in vg_members[neighbor_vg_idx]:
                        if w_id == selected_roots[neighbor_vg_idx]:
                            continue # Cannot move root
                        
                        bw_to_weak = get_connection_strength(adj, w_id, weak_root)
                        if bw_to_weak <= 1e-9:
                            continue
                            
                        w_dev = device_map[w_id]
                        new_neighbor_compute = vg_stats[neighbor_vg_idx]['compute'] - w_dev.compute
                        new_neighbor_memory = vg_stats[neighbor_vg_idx]['memory'] - w_dev.memory
                        
                        if new_neighbor_compute < config.P_min or new_neighbor_memory < config.M_min:
                            continue
                            
                        bw_to_current = get_connection_strength(adj, w_id, selected_roots[neighbor_vg_idx])
                        delta_loss = bw_to_current - bw_to_weak
                        
                        if delta_loss < min_delta_loss:
                            min_delta_loss = delta_loss
                            best_donor_node = w_id
                            best_donor_vg = neighbor_vg_idx
                            found_candidate = True
                
                if found_candidate:
                    # Move
                    vg_members[best_donor_vg].remove(best_donor_node)
                    vg_members[weak_vg_idx].append(best_donor_node)
                    dev_to_vg[best_donor_node] = weak_vg_idx
                    
                    # Update local stats
                    w_dev = device_map[best_donor_node]
                    vg_stats[best_donor_vg]['compute'] -= w_dev.compute
                    vg_stats[best_donor_vg]['memory'] -= w_dev.memory
                    
                    vg_stats[weak_vg_idx]['compute'] += w_dev.compute
                    vg_stats[weak_vg_idx]['memory'] += w_dev.memory
                    
                    if vg_stats[weak_vg_idx]['compute'] >= config.P_min and vg_stats[weak_vg_idx]['memory'] >= config.M_min:
                        vg_stats[weak_vg_idx]['is_critical'] = False
                        
                    optimization_active = True

        result.device_to_vg_map = dev_to_vg
        
        # =========================================================
        # Phase 4: Max-Min Bottleneck Ordering
        # =========================================================
        
        if K > 0:
            p = list(range(K))
            
            root_adj = [[0.0] * K for _ in range(K)]
            for i in range(K):
                for j in range(K):
                    if i == j: continue
                    root_adj[i][j] = get_connection_strength(adj, selected_roots[i], selected_roots[j])
            
            best_p = list(p)
            max_min_bw = -1.0
            
            if K == 1:
                result.pipeline_order = p
            else:
                for perm in itertools.permutations(p):
                    current_min = float('inf')
                    for k in range(K - 1):
                        bw = root_adj[perm[k]][perm[k+1]]
                        if bw < current_min:
                            current_min = bw
                    
                    if current_min > max_min_bw:
                        max_min_bw = current_min
                        best_p = list(perm)
                
                result.pipeline_order = best_p
        
        return result


def solve_ccwf(workers: List[WorkerProfile], task: LayerTask) -> AllocationResult:
    if not workers:
        return AllocationResult(alphas=[], estimated_latency=0.0)
    
    n = len(workers)
    H = [0.0] * n
    K_val = [0.0] * n
    
    min_H = float('inf')
    max_H_plus_K = 0.0
    
    for i, w in enumerate(workers):
        # H_i = T_recv = input / bandwidth
        # bandwidth is in bps
        if w.bandwidth_bps > 1e-9:
            H[i] = task.input_bytes / w.bandwidth_bps
        else:
            H[i] = 1e9 # Very large penalty if no bandwidth? Or assumed 0 if local?
                       # Logic in CPP: if dev != root, bw = link.
                       # If dev == root, bw = 1000 Gbps.
                       # So H[i] should be small for root.
        
        t_comp_full = task.total_flops / w.compute_flops if w.compute_flops > 0 else 1e9
        t_send_full = task.output_bytes / w.bandwidth_bps if w.bandwidth_bps > 1e-9 else 1e9
        
        K_val[i] = t_comp_full + t_send_full
        
        if H[i] < min_H: min_H = H[i]
        
        t_full = H[i] + K_val[i]
        if t_full > max_H_plus_K:
            max_H_plus_K = t_full
            
    # Binary Search
    low = min_H
    high = max_H_plus_K * 2.0
    max_iter = 100
    best_T = high
    
    for _ in range(max_iter):
        mid = low + (high - low) / 2.0
        sum_alpha = 0.0
        
        for i in range(n):
            alpha = 0.0
            if mid > H[i]:
                alpha = (mid - H[i]) / K_val[i]
            
            if alpha > w.max_alpha_mem:
                alpha = w.max_alpha_mem
            
            sum_alpha += alpha
            
        if sum_alpha >= 1.0:
            best_T = mid
            high = mid
        else:
            low = mid
            
    # Final Alpha
    alphas = [0.0] * n
    final_sum = 0.0
    for i in range(n):
        alpha = 0.0
        if best_T > H[i]:
            alpha = (best_T - H[i]) / K_val[i]
        
        if alpha > workers[i].max_alpha_mem:
            alpha = workers[i].max_alpha_mem
        
        if alpha < 0.0: alpha = 0.0
        
        alphas[i] = alpha
        final_sum += alpha
        
    # Normalize
    if final_sum > 0.0:
        for i in range(n):
            alphas[i] /= final_sum
            
    return AllocationResult(alphas=alphas, estimated_latency=best_T)


def solve_layer_partition(vgs: List[VGProfile], model: ModelConfig) -> List[int]:
    if not vgs or model.total_layers <= 0:
        return []
        
    num_vgs = len(vgs)
    total_layers = model.total_layers
    infinity = float('inf')
    
    # dp[k][l]
    dp = [[infinity] * (total_layers + 1) for _ in range(num_vgs + 1)]
    split_point = [[-1] * (total_layers + 1) for _ in range(num_vgs + 1)]
    
    dp[0][0] = 0.0
    
    for k in range(1, num_vgs + 1):
        vg = vgs[k - 1]
        
        for l in range(total_layers + 1):
            # Try previous split point j
            for j in range(l + 1):
                if dp[k - 1][j] == infinity:
                    continue
                    
                count = l - j
                
                # Check memory capacity
                if count > vg.max_layers_capacity:
                    continue
                    
                t_calc = count * vg.unit_time_ms
                t_comm = 0.0
                
                if count > 0:
                    if vg.next_link_bw_gbps > 1e-9:
                        # Time (sec) = Size (Gb) / BW (Gbps)
                        transfer_seconds = (model.activation_size_gb * 8.0) / vg.next_link_bw_gbps
                        t_comm = transfer_seconds * 1000.0 # ms
                    else:
                        t_comm = 0.0 # Assume valid or last node
                
                stage_cost = t_calc + t_comm
                current_bottleneck = max(dp[k - 1][j], stage_cost)
                
                if current_bottleneck < dp[k][l]:
                    dp[k][l] = current_bottleneck
                    split_point[k][l] = j
                    
    if dp[num_vgs][total_layers] == infinity:
        print("OLP Failed: No valid partition found.")
        return []
        
    # Backtrack
    allocation = [0] * num_vgs
    curr_l = total_layers
    for k in range(num_vgs, 0, -1):
        prev_l = split_point[k][curr_l]
        allocation[k - 1] = curr_l - prev_l
        curr_l = prev_l
        
    return allocation


def run_initialization(
    devices: List[Device],
    links: List[Link],
    rragc_config: RRAGCConfig,
    layer_task_template: LayerTask,
    model_config: ModelConfig
) -> InitResult:
    final_result = InitResult(rragc_result=RRAGCResult())
    
    # Step 1: RRAGC
    print("[Init] Step 1: Running RRAGC...")
    final_result.rragc_result = RRAGC.solve(devices, links, rragc_config)
    
    device_map = {d.id: d for d in devices}
    vg_members: Dict[int, List[int]] = {vg_id: [] for vg_id in range(rragc_config.K)}
    for dev_id, vg_id in final_result.rragc_result.device_to_vg_map.items():
        vg_members[vg_id].append(dev_id)
        
    # Step 2: CCWF
    print("[Init] Step 2: Running CCWF for each VG...")
    vg_profiles_for_olp: List[VGProfile] = []
    
    for vg_id in final_result.rragc_result.pipeline_order:
        root_id = final_result.rragc_result.vg_roots[vg_id]
        member_ids = vg_members[vg_id]
        
        workers: List[WorkerProfile] = []
        total_mem_capacity_layers = 0
        
        for dev_id in member_ids:
            dev = device_map[dev_id]
            # Assume 0.5GB per layer
            total_mem_capacity_layers += int(dev.memory / 0.5)
            
            bw_to_root = 1000.0 # Default high for root itself
            if dev_id != root_id:
                bw_to_root = get_bandwidth(links, root_id, dev_id)
                if bw_to_root <= 1e-9: bw_to_root = 1e-9
            
            workers.append(WorkerProfile(
                dev_id=dev_id,
                compute_flops=dev.compute,
                bandwidth_bps=bw_to_root * 1e9, # Gbps to bps
                max_alpha_mem=1.0 # Simplified
            ))
            
        ccwf_res = solve_ccwf(workers, layer_task_template)
        final_result.intravg_allocations[vg_id] = ccwf_res
        
        # Prepare for OLP
        next_link_bw = 0.0
        try:
            idx = final_result.rragc_result.pipeline_order.index(vg_id)
            if idx < len(final_result.rragc_result.pipeline_order) - 1:
                next_vg_id = final_result.rragc_result.pipeline_order[idx + 1]
                next_root_id = final_result.rragc_result.vg_roots[next_vg_id]
                next_link_bw = get_bandwidth(links, root_id, next_root_id)
        except ValueError:
            pass
            
        vg_profiles_for_olp.append(VGProfile(
            vg_id=vg_id,
            unit_time_ms=ccwf_res.estimated_latency * 1000.0,
            max_layers_capacity=total_mem_capacity_layers,
            next_link_bw_gbps=next_link_bw
        ))
        
    # Step 3: OLP
    print("[Init] Step 3: Running OLP (DP Algorithm)...")
    final_result.intervg_layer_allocation = solve_layer_partition(vg_profiles_for_olp, model_config)
    
    if not final_result.intervg_layer_allocation:
        print("[Init] OLP Failed to find a valid partition!")
        
    return final_result

# --- Example Usage ---
if __name__ == "__main__":
    # Example setup
    devices = [
        Device(id=0, compute=100.0, memory=16.0),
        Device(id=1, compute=50.0, memory=8.0),
        Device(id=2, compute=90.0, memory=16.0),
        Device(id=3, compute=40.0, memory=4.0),
        Device(id=4, compute=30.0, memory=4.0),
        Device(id=5, compute=20.0, memory=2.0)
    ]
    
    links = [
        Link(0, 1, 10.0), Link(1, 0, 10.0),
        Link(0, 2, 5.0), Link(2, 0, 5.0),
        Link(2, 3, 8.0), Link(3, 2, 8.0),
        Link(1, 4, 6.0), Link(4, 1, 6.0),
        Link(0, 5, 2.0), Link(5, 0, 2.0),
        Link(2, 4, 1.0), Link(4, 2, 1.0)
    ]
    
    rragc_config = RRAGCConfig(K=2, P_min=60.0, M_min=10.0, alpha=0.7, beta=0.3)
    
    # Task Template (for one layer)
    # Assume 1GB input, 1GB output, 1 TFLOP compute
    layer_task = LayerTask(
        input_bytes=1.0 * 1024**3, 
        output_bytes=1.0 * 1024**3, 
        total_flops=1.0 * 10**12
    )
    
    model_config = ModelConfig(
        total_layers=24,
        activation_size_gb=1.0
    )
    
    result = run_initialization(devices, links, rragc_config, layer_task, model_config)
    
    print("\n=== Final Initialization Result ===")
    print("1. Pipeline Order (VG IDs):", result.rragc_result.pipeline_order)
    print("2. Stage Allocations:")
    for i, vg_id in enumerate(result.rragc_result.pipeline_order):
        root = result.rragc_result.vg_roots[vg_id]
        num_layers = result.intervg_layer_allocation[i] if i < len(result.intervg_layer_allocation) else 0
        print(f"   Stage {i} (VG {vg_id}, Root {root}): Assigned {num_layers} Layers")
        
        # Print Intra-VG Split
        alloc = result.intravg_allocations.get(vg_id)
        if alloc:
            print(f"     Latency per Layer: {alloc.estimated_latency*1000:.2f} ms")
            print(f"     Device Split (Alpha): {['{:.2f}'.format(a) for a in alloc.alphas]}")
