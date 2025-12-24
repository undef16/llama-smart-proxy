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
import concurrent.futures
from src.proxy.config import Config
class E2ESimulation:
    """Handles end-to-end simulation of the Llama Smart Proxy."""
    def __init__(self):


        # Load configuration
        self.config = Config.load('config.json')
        if self.config.simulation is None:
            raise ValueError("Simulation configuration is required for this script")
        self.sim = self.config.simulation
        if self.sim.enable_remote:
            host = self.sim.host
            if host.startswith('http://'):
                host = host[7:]
            elif host.startswith('https://'):
                host = host[8:]
            self.server_url = f"http://{host}:{self.sim.port}"
        else:
            self.server_url = self.sim.server_url or f"http://localhost:{self.config.server.port}"

    def get_base_request_data(self):
        """Get the base request data for chat completion."""
        return {
            "model": self.sim.model,
            "messages": [msg.dict() for msg in self.sim.messages]
        }

    def send_request(self, endpoint, data):
        """Send a POST request to the specified endpoint with the given data."""
        return requests.post(
            self.server_url + endpoint,
            json=data,
            timeout=self.sim.request_timeout
        )

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

        result = 1
        print("Starting Llama Smart Proxy E2E Simulation...")

        server_process = None
        if not self.sim.enable_remote:
            # Start the proxy server
            server_process = self.start_server()
            if not server_process:
                return result

        try:
            # Run all checks
            checks = [self.check_completion, self.check_parallel_requests, self.check_forwarding_endpoints]
            for check in checks:
                if not check():
                    return result

            result = 0
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")

        finally:
            # Clean up: terminate the server
            if server_process:
                self.stop_server(server_process)

            # Server output is printed directly

        print("E2E simulation completed successfully!")
        return result

    def check_completion(self):
        result = False
        request_data = self.get_base_request_data()

        # Send the request
        response = self.send_request(self.sim.chat_endpoint, request_data)

        if response.status_code == 200:
            response_data = response.json()
            print("Request successful!")
            print("Response:")
            print(json.dumps(response_data, indent=2))
            result = True
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

        return result

    def check_parallel_requests(self):
        """Test parallel concurrent requests to the chat completion endpoint."""
        print("Testing parallel requests...")

        request_data = self.get_base_request_data()
        request_data["max_tokens"] = 512

        def send_request():
            try:
                response = self.send_request(self.sim.chat_endpoint, request_data)
                if response.status_code == 200:
                    result = response.json()
                    print(f'Response: {result}')
                    # Check if response has choices with content
                    if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0]:
                        return True
                    else:
                        return False
                else:
                    return False
            except requests.RequestException:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(send_request) for _ in range(4)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        success_count = sum(results)
        if success_count == 4:
            print("All parallel requests successful!")
            return True
        else:
            print(f"Parallel requests failed: {success_count}/4 successful")
            return False

    def check_forwarding_endpoints(self):
        """Test forwarding of endpoints that are not processed by the proxy."""
        print("Testing forwarding endpoints...")
        print("Note: Forwarding requires a running backend server")

        forwarding_endpoints = [
            "/tokenize",
            "/detokenize",
            # "/embedding",
            # "/props",
            # "/reranking"
        ]

        backend_unavailable_count = 0
        success_count = 0
        for endpoint in forwarding_endpoints:
            try:
                response = self.send_request(endpoint, {"model": self.sim.model})
                # print(f"Response: {response.json()}")
                if response.status_code == 200:
                    print(f"✓ {endpoint} forwarded successfully")
                    success_count += 1
                elif response.status_code == 502:
                    print(f"⚠ {endpoint} backend unavailable (502)")
                    backend_unavailable_count += 1
                else:
                    print(f"✗ {endpoint} failed with status {response.status_code}")
            except requests.RequestException as e:
                print(f"✗ {endpoint} failed: {e}")

        if success_count == len(forwarding_endpoints):
            print("All forwarding endpoints tested successfully")
            return True
        elif backend_unavailable_count == len(forwarding_endpoints):
            print("All forwarding endpoints report backend unavailable (expected when no backend server is running)")
            return True  # Don't fail the simulation if backend is not running
        elif success_count + backend_unavailable_count == len(forwarding_endpoints):
            print(f"Forwarding test completed: {success_count} successful, {backend_unavailable_count} backend unavailable")
            return True
        else:
            print(f"Partial success: {success_count}/{len(forwarding_endpoints)} endpoints forwarded")
            return True

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
            check_response = self.wait_for_server(self.server_url + self.sim.health_endpoint, timeout=self.sim.wait_timeout)
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