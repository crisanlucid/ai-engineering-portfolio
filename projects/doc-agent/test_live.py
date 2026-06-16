#!/usr/bin/env python3
import json
import urllib.request
import urllib.parse
import sys

BASE_URL = "http://localhost:9010"

def run_test():
    print("=== Testing GET /health ===")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health") as response:
            status = response.getcode()
            body = json.loads(response.read().decode())
            print(f"Status: {status}")
            print(f"Body: {json.dumps(body, indent=2)}\n")
    except Exception as e:
        print(f"Error testing /health: {e}\n")

    print("=== Testing GET /ask ===")
    try:
        params = urllib.parse.urlencode({"question": "What does tesseract.js do?", "k": 2})
        with urllib.request.urlopen(f"{BASE_URL}/ask?{params}") as response:
            status = response.getcode()
            body = json.loads(response.read().decode())
            print(f"Status: {status}")
            print(f"Body: {json.dumps(body, indent=2)}\n")
    except Exception as e:
        print(f"Error testing GET /ask: {e}\n")

    print("=== Testing POST /ask ===")
    try:
        data = json.dumps({"question": "What does tesseract.js do?", "k": 2}).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/ask",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            body = json.loads(response.read().decode())
            print(f"Status: {status}")
            print(f"Body: {json.dumps(body, indent=2)}\n")
    except Exception as e:
        print(f"Error testing POST /ask: {e}\n")

if __name__ == "__main__":
    run_test()
