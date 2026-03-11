"""
One-time setup: register agent with agenthub and create channels.

Usage: uv run scripts/setup_hub.py --server http://localhost:8080 --admin-key YOUR_KEY
"""

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Set up agenthub integration")
    parser.add_argument("--server", required=True, help="Agenthub server URL")
    parser.add_argument("--admin-key", required=True, help="Admin API key")
    parser.add_argument("--agent-name", default="embed-researcher", help="Agent name")
    args = parser.parse_args()

    # Register agent
    print(f"Registering agent '{args.agent_name}' with {args.server}...")
    code, out = run(["ah", "join", "--server", args.server, "--name", args.agent_name, "--admin-key", args.admin_key])
    if code != 0:
        print(f"Warning: join returned {code}: {out}")
    else:
        print(f"Registered: {out}")

    # Create channels
    channels = [
        ("embed-results", "Experiment outcomes (keep/discard/crash)"),
        ("embed-datasets", "Dataset discovery and curation log"),
        ("embed-ideas", "Research hypotheses and reasoning"),
        ("embed-leaderboard", "Current best MTEB scores"),
    ]

    for name, desc in channels:
        print(f"Creating channel: #{name}")
        code, out = run(["ah", "board", "create", name, "--description", desc])
        if code != 0:
            print(f"  Warning: {out} (may already exist)")
        else:
            print(f"  Created: {out}")

    # Post initial message
    code, out = run(["ah", "post", "embed-results", "Agent embed-researcher online. Starting autonomous embedding research."])
    print("\nSetup complete!")


if __name__ == "__main__":
    main()
