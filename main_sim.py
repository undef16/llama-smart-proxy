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


def wait_for_server(url: str, timeout: int = 30):
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


def main():
    """Main simulation function."""
    print("Starting Llama Smart Proxy E2E Simulation...")

    # Start the proxy server
    try:
        print("Starting proxy server...")
        server_process = subprocess.Popen(
            [sys.executable, 'main.py'],
            cwd=os.getcwd()
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        return 1

    try:
        # Wait for server to be ready
        print("Waiting for server to start...")
        check_response = wait_for_server("http://localhost:8000/health", timeout=30)
        if not check_response:
            print("Server failed to start within timeout")
            return 1

        print(f"Server is ready. Sending chat completion request... {check_response}")

        # Prepare the request data
        request_data = {
            "model": "unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke about a python programmer.",
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            ],
        }

        # Send the request
        response = requests.post(
            "http://localhost:8000/chat/completions",
            json=request_data,
            timeout=300 
        )

        if response.status_code == 200:
            result = response.json()
            print("Request successful!")
            print("Response:")
            print(json.dumps(result, indent=2))

            # Extract and display key information
            if "choices" in result and len(result["choices"]) > 0:
                assistant_message = result["choices"][0]["message"]["content"]
                print(f"\nAssistant response: {assistant_message}")

            if "usage" in result:
                usage = result["usage"]
                print(f"\nToken usage: {usage['total_tokens']} total "
                      f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)")
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
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()

        # Server output is printed directly

    print("E2E simulation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())