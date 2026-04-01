# scripts/traci_emulator.py
import socket
import struct
import pandas as pd
import json
import time

TRACI_PORT = 9999
HOST       = "localhost"

def encode_traci_response(vehicle_data):
    """
    Encode vehicle positions in TraCI binary format
    vehicle_data: list of dicts with id, x, y, speed, angle
    """
    # TraCI response: simplified position list
    # Real TraCI uses specific command IDs — 
    # this is a simplified version compatible with Veins
    payload = b''
    
    for veh in vehicle_data:
        vid     = veh['id'].encode('utf-8')
        payload += struct.pack('>H', len(vid)) + vid
        payload += struct.pack('>d', veh['x'])
        payload += struct.pack('>d', veh['y'])
        payload += struct.pack('>d', veh['speed'])
        payload += struct.pack('>d', veh['angle'])
    
    # Prepend vehicle count
    header  = struct.pack('>I', len(vehicle_data))
    return header + payload

class TraCIEmulator:
    def __init__(self, trajectories_path, step_size=1.0):
        print("Loading trajectory data...")
        self.df        = pd.read_csv(trajectories_path)
        self.step_size = step_size
        self.timesteps = sorted(self.df['SIMSEC'].unique())
        self.current   = 0
        print(f"  Loaded {len(self.df)} records")
        print(f"  Timesteps: {len(self.timesteps)}")
        print(f"  Vehicles:  {self.df['NO'].nunique()}")
        print(f"  Duration:  {self.timesteps[-1]}s")
    
    def get_vehicles_at(self, simsec):
        """Get all vehicle positions at a given timestep"""
        snapshot = self.df[self.df['SIMSEC'] == simsec]
        vehicles = []
        for _, row in snapshot.iterrows():
            vehicles.append({
                'id':    str(int(row['NO'])),
                'x':     row['X'],
                'y':     row['Y'],
                'speed': row['SPEED'] / 3.6,  # km/h → m/s
                'angle': 0.0  # can compute from successive positions
            })
        return vehicles
    
    def run(self):
        """Start TraCI server and wait for OMNeT++ to connect"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, TRACI_PORT))
        server.listen(1)
        
        print(f"\nTraCI emulator listening on port {TRACI_PORT}")
        print("Waiting for OMNeT++ to connect...")
        
        conn, addr = server.accept()
        print(f"OMNeT++ connected from {addr}")
        
        for step_idx, simsec in enumerate(self.timesteps):
            vehicles = self.get_vehicles_at(simsec)
            response = encode_traci_response(vehicles)
            
            try:
                conn.sendall(response)
                # Wait for OMNeT++ acknowledgement
                ack = conn.recv(4)
                if not ack:
                    print("OMNeT++ disconnected")
                    break
            except ConnectionResetError:
                print("Connection lost")
                break
            
            print(f"  t={simsec:.1f}s → "
                  f"{len(vehicles)} vehicles sent")
            time.sleep(self.step_size * 0.01)  # pacing
        
        print("\nSimulation complete!")
        conn.close()
        server.close()

if __name__ == "__main__":
    emulator = TraCIEmulator(
        trajectories_path="outputs/vissim/trajectories_xy.csv",
        step_size=1.0
    )
    emulator.run()

