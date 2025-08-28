#!/usr/bin/env python
"""
Restart VLM server script
Kills any existing process on port 3026 before starting the server
"""

import os
import sys
import time
import platform
import subprocess
import signal

def find_and_kill_process_on_port(port):
    """Find and kill process listening on the specified port"""
    system = platform.system()
    
    print(f"Checking for existing processes on port {port}...")
    
    try:
        if system == "Windows":
            # Windows: Use netstat to find the process
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                pids = set()
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        # Get PID (last column)
                        pid = parts[-1]
                        if pid.isdigit() and pid != "0":
                            pids.add(int(pid))
                
                for pid in pids:
                    try:
                        # Get process name
                        name_cmd = f"tasklist /FI \"PID eq {pid}\" /FO CSV"
                        name_result = subprocess.run(name_cmd, shell=True, capture_output=True, text=True)
                        process_name = "unknown"
                        if name_result.stdout:
                            lines = name_result.stdout.strip().split('\n')
                            if len(lines) > 1:
                                # Parse CSV output
                                import csv
                                reader = csv.reader([lines[1]])
                                for row in reader:
                                    if row:
                                        process_name = row[0].strip('"')
                        
                        print(f"Found process '{process_name}' (PID: {pid}) using port {port}")
                        print("Stopping process...")
                        
                        # Kill the process
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                        print("Process stopped successfully")
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error killing process {pid}: {e}")
                
                if not pids:
                    print(f"No process found on port {port}")
            else:
                print(f"No process found on port {port}")
                
        else:  # Linux/Mac
            # Unix-like: Use lsof to find the process
            cmd = f"lsof -i :{port} -t"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.isdigit():
                        try:
                            # Get process name
                            name_result = subprocess.run(f"ps -p {pid} -o comm=", 
                                                       shell=True, capture_output=True, text=True)
                            process_name = name_result.stdout.strip() if name_result.stdout else "unknown"
                            
                            print(f"Found process '{process_name}' (PID: {pid}) using port {port}")
                            print("Stopping process...")
                            
                            # Kill the process
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(1)
                            
                            # Force kill if still running
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass  # Process already terminated
                            
                            print("Process stopped successfully")
                            
                        except Exception as e:
                            print(f"Error killing process {pid}: {e}")
            else:
                print(f"No process found on port {port}")
                
    except Exception as e:
        print(f"Error checking for processes: {e}")
        print("Continuing anyway...")

def start_server():
    """Start the VLM server using uvicorn"""
    print("\nStarting VLM server on port 3026...")
    print("Press Ctrl+C to stop the server\n")
    
    # Change to the VLM server directory
    server_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(server_dir)
    
    # Start uvicorn
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", 
           "--host", "0.0.0.0", "--port", "3026"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def main():
    """Main function"""
    # Kill existing process on port 3026
    find_and_kill_process_on_port(3026)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
