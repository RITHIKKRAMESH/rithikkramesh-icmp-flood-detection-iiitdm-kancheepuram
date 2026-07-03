import os
import sys
import time
import random
import joblib
import pandas as pd
import numpy as np

# Try importing Scapy
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "models/best_model.pkl"
FEATURES_SCHEMA = ['proto', 'total_length', 'src_port', 'dst_port', 'fin', 'syn', 'rst', 'psh', 'ack', 'urg']
PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP"}

# ANSI Colors for gorgeous alerts
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# ==========================================================
# GORGEOUS ARCHITECTURE DIAGRAM
# ==========================================================
def print_architecture():
    print(BLUE + "=" * 80)
    print("                 REAL-TIME DETECTION ADAPTER ARCHITECTURE DIAGRAM")
    print("=" * 80 + RESET)
    print("""
    [ Real Network Traffic ]
               |
               v
    [ Packet Sniffer / Capture Layer ] --(Scapy sniff() / Simulation Engine)
               |
               v
    [ Feature Extractor Module ] -------(Extracts proto, length, TCP flags, ports)
               |
               v
    [ Data Transform Layer ] -----------(Aligns to 10-feature model schema)
               |
               v
    [ Inference Engine ] ---------------(LightGBM Best Binary Model)
               |
               +------------------------+
               v                        v
      [ Prediction: 0 ]        [ Prediction: 1 ]
               |                        |
               v                        v
       [ Normal Traffic ]       [ DDOS ATTACK ALERT! ]
       (Console Logging)        (Red Alert Printed)
    """)
    print(BLUE + "=" * 80 + RESET)

# ==========================================================
# FEATURE EXTRACTION ENGINE
# ==========================================================
def extract_packet_features(pkt):
    # Default feature values
    proto = 0
    total_length = len(pkt)
    src_port = 0
    dst_port = 0
    fin = 0
    syn = 0
    rst = 0
    psh = 0
    ack = 0
    urg = 0

    # Parse IP layer
    if pkt.haslayer(IP):
        proto = pkt[IP].proto
        total_length = pkt[IP].len if pkt[IP].len else len(pkt)
    
    # Parse Transport layers
    if pkt.haslayer(TCP):
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
        # Extract TCP flags
        flags = pkt[TCP].underlayer.sprintf("%TCP.flags%") if hasattr(pkt[TCP], 'underlayer') else str(pkt[TCP].flags)
        if "F" in flags: fin = 1
        if "S" in flags: syn = 1
        if "R" in flags: rst = 1
        if "P" in flags: psh = 1
        if "A" in flags: ack = 1
        if "U" in flags: urg = 1
    elif pkt.haslayer(UDP):
        src_port = pkt[UDP].sport
        dst_port = pkt[UDP].dport
    elif pkt.haslayer(ICMP):
        proto = 1  # Ensure proto maps correctly for ICMP
        src_port = 0
        dst_port = 0

    return {
        'proto': proto,
        'total_length': total_length,
        'src_port': src_port,
        'dst_port': dst_port,
        'fin': fin,
        'syn': syn,
        'rst': rst,
        'psh': psh,
        'ack': ack,
        'urg': urg
    }

# ==========================================================
# INFERENCE & ALERT ENGINE
# ==========================================================
def run_inference(features, model):
    # Prepare input dataframe
    df = pd.DataFrame([features])[FEATURES_SCHEMA]
    
    # Run prediction
    pred = model.predict(df)[0]
    
    proto_name = PROTO_MAP.get(features['proto'], f"Proto({features['proto']})")
    
    if pred == 1:
        print(f"{RED}[ALERT] DDoS Flood Attack Detected! Protocol: {proto_name} | Length: {features['total_length']} bytes | Src Port: {features['src_port']} -> Dst Port: {features['dst_port']}{RESET}")
    else:
        print(f"{GREEN}[INFO] Normal Traffic. Protocol: {proto_name} | Length: {features['total_length']} bytes | Src Port: {features['src_port']} -> Dst Port: {features['dst_port']}{RESET}")

# ==========================================================
# TRAFFIC SIMULATION (fallback or test mode)
# ==========================================================
def run_simulation(model):
    print(YELLOW + "\nStarting Real-Time Packet Simulation Engine (Ctrl+C to stop)..." + RESET)
    print("Press Ctrl+C to terminate the adapter script.\n")
    try:
        while True:
            # Randomly choose normal or attack scenario
            scenario = random.choices(["normal", "tcp_flood", "udp_flood", "icmp_flood"], weights=[60, 15, 15, 10])[0]
            
            if scenario == "normal":
                features = {
                    'proto': random.choice([6, 17]),  # TCP or UDP
                    'total_length': random.randint(40, 1500),
                    'src_port': random.randint(1024, 65535),
                    'dst_port': random.choice([80, 443, 22, 53]),
                    'fin': 0, 'syn': 0, 'rst': 0, 'psh': 0, 'ack': 1, 'urg': 0
                }
            elif scenario == "tcp_flood":
                # SYN flood attack features
                features = {
                    'proto': 6,  # TCP
                    'total_length': 64,  # Small packet length typical of SYN floods
                    'src_port': random.randint(1024, 65535),
                    'dst_port': 80,
                    'fin': 0, 'syn': 1, 'rst': 0, 'psh': 0, 'ack': 0, 'urg': 0
                }
            elif scenario == "udp_flood":
                # UDP flood attack features
                features = {
                    'proto': 17,  # UDP
                    'total_length': 1024,  # Larger payload size to exhaust bandwidth
                    'src_port': random.randint(1024, 65535),
                    'dst_port': 53,
                    'fin': 0, 'syn': 0, 'rst': 0, 'psh': 0, 'ack': 0, 'urg': 0
                }
            elif scenario == "icmp_flood":
                # ICMP Ping flood features
                features = {
                    'proto': 1,  # ICMP
                    'total_length': 84,
                    'src_port': 0,
                    'dst_port': 0,
                    'fin': 0, 'syn': 0, 'rst': 0, 'psh': 0, 'ack': 0, 'urg': 0
                }
            
            # Perform prediction
            run_inference(features, model)
            time.sleep(random.uniform(0.1, 0.8))  # Real-time pacing
            
    except KeyboardInterrupt:
        print(YELLOW + "\nSimulation stopped by user." + RESET)

# ==========================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    print_architecture()
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}.")
        print("Please train your model first before running the adapter.")
        sys.exit(1)
        
    print(f"Loading trained LightGBM binary model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.\n")
    
    # Check command-line arguments or TTY status
    choice = "2"  # Default to simulation
    if len(sys.argv) > 1:
        if sys.argv[1] == "--real":
            choice = "1"
        elif sys.argv[1] == "--simulate":
            choice = "2"
    elif sys.stdin.isatty():
        # Ask for mode in interactive shell
        print("Choose network traffic source:")
        print("1) Capture Real Traffic on Network Interface (Requires Npcap + Admin)")
        print("2) Run Real-Time Attack Simulation Engine (Default)")
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
        except Exception:
            choice = "2"
    else:
        print("Non-interactive run detected. Defaulting to Real-Time Attack Simulation Engine.")

    if choice == "1":
        if not SCAPY_AVAILABLE:
            print("\nError: Scapy is not installed or available. Falling back to Simulation Mode.")
            run_simulation(model)
        else:
            print(YELLOW + "\n[WARNING] Sniffing network packets on default interface..." + RESET)
            print("To capture traffic, make sure you are running as Administrator / root.")
            print("Press Ctrl+C to stop sniffing.\n")
            
            def packet_callback(pkt):
                try:
                    features = extract_packet_features(pkt)
                    run_inference(features, model)
                except Exception as e:
                    pass
            
            try:
                sniff(prn=packet_callback, store=0)
            except Exception as e:
                print(RED + f"Sniffing error: {e}" + RESET)
                print(YELLOW + "Check Admin privileges or Npcap installation. Falling back to Simulation Mode..." + RESET)
                run_simulation(model)
    else:
        run_simulation(model)
        
    print("\nReal-time detection adapter terminated.")
