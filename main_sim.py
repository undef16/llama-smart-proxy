#!/usr/bin/env python3
"""
E2E Simulation Script for Llama Smart Proxy

This script starts the proxy server and performs a real end-to-end test
by sending a chat completion request and receiving a response without
using any mock objects.
"""

import subprocess
import time
import requests
import json
import sys
import os
from src.proxy.config import Config
class E2ESimulation:
    """Handles end-to-end simulation of the Llama Smart Proxy."""
    def __init__(self):


        # Load configuration
        self.config = Config.load('config.json')
        if self.config.simulation is None:
            raise ValueError("Simulation configuration is required for this script")
        self.sim = self.config.simulation

    def wait_for_server(self, url: str, timeout: int = 30):
        """Wait for the server to be ready by checking the health endpoint."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                return response
            except requests.RequestException:
                pass
            time.sleep(1)
        return None

    def run(self):
        """Main simulation function."""
        print("Starting Llama Smart Proxy E2E Simulation...")

        # Start the proxy server
        server_process = self.start_server()
        if not server_process:
            return 1

        try:

            # Prepare the request data
            request_data = {
                "model": self.sim.model,
                "messages": [msg.dict() for msg in self.sim.messages]
            }

            # Send the request
            response = requests.post(
                self.sim.server_url + self.sim.chat_endpoint,
                json=request_data,
                timeout=self.sim.request_timeout
            )

            if response.status_code == 200:
                result = response.json()
                print("Request successful!")
                print("Response:")
                print(json.dumps(result, indent=2))

            else:
                print(f"Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return 1

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return 1
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            return 1
        finally:
            # Clean up: terminate the server
            self.stop_server(server_process)

            # Server output is printed directly

        print("E2E simulation completed successfully!")
        return 0

    def stop_server(self, server_process):
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=self.sim.terminate_timeout)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()

    def start_server(self):
        try:
            print("Starting proxy server...")
            server_process = subprocess.Popen(
                [sys.executable, 'main.py'],
                cwd=os.getcwd()
            )
            # Wait for server to be ready
            print("Waiting for server to start...")
            check_response = self.wait_for_server(self.sim.server_url + self.sim.health_endpoint, timeout=self.sim.wait_timeout)
            if not check_response:
                print("Server failed to start within timeout")
                return None

            print(f"Server is ready. Sending chat completion request...")

        except Exception as e:
            print(f"Failed to start server: {e}")
            server_process = None
            
        return server_process


if __name__ == "__main__":
    sim = E2ESimulation()
    sys.exit(sim.run())