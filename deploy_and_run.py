import paramiko
import time
import threading
import sys
import os

# Configuration
NODES = [
    {'ip': '192.168.182.11', 'user': 'cc', 'pass': 'cc123', 'role': 'worker', 'python': '/usr/local/bin/python3.9'},
    {'ip': '192.168.182.12', 'user': 'cc', 'pass': 'cc123', 'role': 'worker', 'python': '/usr/local/bin/python3.9'},
    {'ip': '192.168.182.13', 'user': 'cc', 'pass': 'cc123', 'role': 'worker', 'python': '/usr/local/bin/python3.9'},
    {'ip': '192.168.182.15', 'user': 'jetson', 'pass': 'yahboom', 'role': 'worker', 'python': 'python3'},
    {'ip': '192.168.182.17', 'user': 'jetson', 'pass': 'yahboom', 'role': 'worker', 'python': 'python3'},
    {'ip': '192.168.182.16', 'user': 'jetson', 'pass': 'yahboom', 'role': 'master', 'python': 'python3'},
]

LOCAL_FILE = 'distributed_inference_optimized.py'
REMOTE_DIR = 'yanhui/EdgeVisor' # Relative to home
REMOTE_FILE = f'{REMOTE_DIR}/{LOCAL_FILE}'

# Master Command Template (Python path will be prepended)
# Optimize distribution:
# Total Layers: 36 (detected from log: n_kv_heads=36 > n_heads=32? No, n_layers=32 from log)
# Log says: {'dim': 4096, ... 'n_layers': 32}
# Total 32 Layers.
# Devices: 3 Jetsons (Fast), 3 CC (Slow).
# Previous: 1:1@26 * 31:1@9 * 1:1@1 (Total 36? No, 26+9+1=36. Wait, log said n_layers=32)
# If model has 32 layers, and we config 36, the last 4 are empty/skipped or errors.
# Let's target 32 layers.

# Strategy:
# Jetsons (Rank 0, 1, 2) are fast.
# CCs (Rank 3, 4, 5) are slow.
# We should give Jetsons most work, but split evenly among them.
# CCs should take minimal work or act as router/embedding/head if possible?
# But pipeline requires linear sequence.
# Order in IP list:
# 16 (Master/Jetson), 17 (Jetson), 15 (Jetson), 11 (CC), 12 (CC), 13 (CC)
# Rank 0: 16
# Rank 1: 17
# Rank 2: 15
# Rank 3: 11
# Rank 4: 12
# Rank 5: 13

# Stable Multi-Device Config:
# Jetsons (Rank 0, 1, 2) are fast.
# CCs (Rank 3, 4, 5) are slow.
# User Request: '1:1@26*31:1@9*1:1@1' (Total 36 Layers)
# We map IPs to satisfy the ratios:
# Stage 0 (1:1@26): Needs 2 devices. Use Jetson + Jetson. (Strong/Strong)
# Stage 1 (31:1@9): Needs 2 devices. Use Jetson + CC. (Strong/Weak ratio 31:1)
# Stage 2 (1:1@1): Needs 2 devices. Use CC + CC. (Weak/Weak)

# IP Order:
# 1. 192.168.182.16 (Jetson) -> Stage 0, Dev 1
# 2. 192.168.182.17 (Jetson) -> Stage 0, Dev 2
# 3. 192.168.182.15 (Jetson) -> Stage 1, Dev 1 (Ratio 31)
# 4. 192.168.182.11 (CC)     -> Stage 1, Dev 2 (Ratio 1)
# 5. 192.168.182.12 (CC)     -> Stage 2, Dev 1
# 6. 192.168.182.13 (CC)     -> Stage 2, Dev 2

MASTER_ARGS = (
    "distributed_inference_optimized.py "
    "--mode assign "
    "--my_ip 192.168.182.16 "
    "--port 29500 "
    "--config '1:1@26*31:1@9*1:1@1' "
    "--ips '192.168.182.16:29500,192.168.182.17:29500,192.168.182.15:29500,192.168.182.11:29500,192.168.182.12:29500,192.168.182.13:29500' "
    "--model /home/models/dllama_model_qwen3_8b_q40.m "
    "--tokenizer /home/models/dllama_tokenizer_qwen3_8b_q40.t "
    "--num_tasks 3"
)

def create_client(node):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(node['ip'], username=node['user'], password=node['pass'], timeout=10)
        return client
    except Exception as e:
        print(f"[{node['ip']}] Connection failed: {e}")
        return None

def deploy_node(node):
    print(f"[{node['ip']}] Deploying...")
    client = create_client(node)
    if not client: return

    try:
        # 1. Upload File
        sftp = client.open_sftp()
        # Ensure dir exists (simple check)
        try:
            sftp.stat(REMOTE_DIR)
        except FileNotFoundError:
            print(f"[{node['ip']}] Creating directory {REMOTE_DIR}...")
            client.exec_command(f'mkdir -p {REMOTE_DIR}')
        
        local_path = os.path.abspath(LOCAL_FILE)
        remote_path = f"/home/{node['user']}/{REMOTE_FILE}"
        sftp.put(local_path, remote_path)
        print(f"[{node['ip']}] File uploaded to {remote_path}")
        sftp.close()

        # 2. Kill existing processes
        stdin, stdout, stderr = client.exec_command("pkill -f distributed_inference_optimized.py")
        stdout.channel.recv_exit_status() # Wait for it
        print(f"[{node['ip']}] Cleaned up old processes.")

    except Exception as e:
        print(f"[{node['ip']}] Deploy failed: {e}")
    finally:
        client.close()

def start_worker(node):
    print(f"[{node['ip']}] Starting Worker...")
    client = create_client(node)
    if not client: return

    try:
        cmd = (
            f"cd {REMOTE_DIR} && "
            f"nohup {node['python']} -u distributed_inference_optimized.py "
            f"--mode listen --my_ip {node['ip']} --port 29500 "
            f"> worker_{node['ip']}.log 2>&1 &"
        )
        client.exec_command(cmd)
        print(f"[{node['ip']}] Worker started.")
    except Exception as e:
        print(f"[{node['ip']}] Start worker failed: {e}")
    finally:
        client.close()

def run_master(node):
    print(f"[{node['ip']}] Starting Master...")
    client = create_client(node)
    if not client: return

    try:
        cmd = f"cd {REMOTE_DIR} && {node['python']} -u {MASTER_ARGS}"
        print(f"[{node['ip']}] Executing: {cmd}")
        
        # Open a session to stream output
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
        
        for line in iter(stdout.readline, ""):
            print(f"[Master Output] {line}", end="")
            
        exit_status = stdout.channel.recv_exit_status()
        print(f"[{node['ip']}] Master finished with status {exit_status}")
        
    except Exception as e:
        print(f"[{node['ip']}] Run master failed: {e}")
    finally:
        client.close()

def main():
    # 1. Deploy to ALL nodes in parallel
    threads = []
    for node in NODES:
        t = threading.Thread(target=deploy_node, args=(node,))
        t.start()
        threads.append(t)
    for t in threads: t.join()
    
    print("\n--- Deployment Complete ---\n")

    # 2. Start Workers
    for node in NODES:
        if node['role'] == 'worker':
            start_worker(node)
    
    # Give workers a moment to spin up
    print("Waiting 5s for workers to initialize...")
    time.sleep(5)
    
    # 3. Start Master
    master_node = next(n for n in NODES if n['role'] == 'master')
    run_master(master_node)

    # 4. Debug: Check logs if failed
    print("\n--- Checking Worker Logs ---")
    for node in NODES:
        if node['role'] == 'worker':
            check_log(node)

def check_log(node):
    client = create_client(node)
    if not client: return
    try:
        cmd = f"tail -n 5 {REMOTE_DIR}/worker_{node['ip']}.log"
        stdin, stdout, stderr = client.exec_command(cmd)
        output = stdout.read().decode().strip()
        if output:
            print(f"[{node['ip']} Log]\n{output}")
        else:
            print(f"[{node['ip']} Log] (Empty)")
    except:
        pass
    finally:
        client.close()

if __name__ == "__main__":
    main()
