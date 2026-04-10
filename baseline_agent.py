"""
Baseline inference agent for Study Planner OpenEnv.
Implements a simple heuristic policy for reproducible baseline scores.

Usage:
    python baseline_agent.py --difficulty easy   # Score: ~0.8 easy, ~0.4 medium, ~0.1 hard
    python baseline_agent.py --difficulty medium  # Score: ~0.9 easy, ~0.6 medium, ~0.2 hard
    python baseline_agent.py --difficulty hard    # Score: ~0.7 easy, ~0.3 medium, ~0.1 hard
    python baseline_agent.py --all               # Run all difficulties, print table
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional

import requests

BASE_URL = "http://localhost:7860"


def reset_env(difficulty: str = "medium", seed: int = 42) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty, "seed": seed})
    resp.raise_for_status()
    return resp.json()


def step_env(action_type: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    resp = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"action_type": action_type, "payload": payload or {}}},
    )
    resp.raise_for_status()
    return resp.json()


def heuristic_policy(obs: Dict[str, Any]) -> tuple[str, Dict]:
    """
    Heuristic baseline policy:
    1. If energy < 25%, rest.
    2. If there are high-priority pending tasks, complete the first one.
    3. Otherwise, study the most urgent subject (closest deadline, least progress).
    """
    energy = obs["energy_level"]
    pending_tasks = obs["pending_tasks"]
    subjects = obs["subjects"]

    if energy < 0.25:
        return "rest", {"hours": 6.0}

    high_priority = [t for t in pending_tasks if t["priority"] == "high"]
    if high_priority:
        return "complete_task", {"task_id": high_priority[0]["id"]}

    incomplete_subjects = [
        s for s in subjects if s["progress"] < 1.0
    ]
    if not incomplete_subjects:
        return "review_schedule", {}

    incomplete_subjects.sort(key=lambda s: (s["days_until_deadline"], -s["progress"]))
    target = incomplete_subjects[0]

    hours = 2.0 if energy > 0.5 else 1.0
    return "study", {"subject_id": target["id"], "hours": hours}


def run_episode(difficulty: str, seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    print(f"\n=== Running episode: difficulty={difficulty}, seed={seed} ===")

    result = reset_env(difficulty, seed)
    obs = result["observation"]
    total_reward = 0.0
    step_count = 0

    if verbose:
        print(f"  Initial state: {obs['subjects'].__len__()} subjects, {len(obs['pending_tasks'])} tasks")

    while True:
        action_type, payload = heuristic_policy(obs)

        if verbose:
            print(f"  Day {obs['day']}/{obs['max_days']} | Energy: {obs['energy_level']:.2f} | "
                  f"Progress: {obs['overall_progress']:.2%} | Action: {action_type}")

        step_result = step_env(action_type, payload)
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]
        info = step_result.get("info", {})

        total_reward += reward
        step_count += 1

        if done:
            print(f"\n  Episode complete after {step_count} steps")
            print(f"  Total reward: {total_reward:.4f}")
            if "final_scores" in info:
                fs = info["final_scores"]
                print(f"\n  GRADER SCORES:")
                print(f"    Easy   (threshold 0.10): {fs['easy']:.4f} → {'PASS' if fs['easy'] >= 0.1 else 'FAIL'}")
                print(f"    Medium (threshold 0.50): {fs['medium']:.4f} → {'PASS' if fs['medium'] >= 0.5 else 'FAIL'}")
                print(f"    Hard   (threshold 0.80): {fs['hard']:.4f} → {'PASS' if fs['hard'] >= 0.8 else 'FAIL'}")
                print(f"\n  DETAILS:")
                print(f"    Overall progress: {fs['overall_progress']:.2%}")
                print(f"    Tasks completed: {fs['tasks_completed_ratio']:.2%}")
                print(f"    On-time ratio: {fs['on_time_ratio']:.2%}")
                print(f"    Days used: {fs['days_used']}")
                print(f"    Total hours studied: {fs['total_hours_studied']:.1f}h")
            return info.get("final_scores", {})


def run_all():
    results = {}
    for diff in ["easy", "medium", "hard"]:
        scores = run_episode(diff, seed=42)
        results[diff] = scores

    print("\n" + "=" * 60)
    print("BASELINE AGENT SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Difficulty':<12} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Progress':>10}")
    print("-" * 60)
    for diff, scores in results.items():
        if scores:
            print(
                f"{diff:<12} "
                f"{scores.get('easy', 0):.4f}  "
                f"{scores.get('medium', 0):.4f}  "
                f"{scores.get('hard', 0):.4f}  "
                f"{scores.get('overall_progress', 0):.2%}"
            )
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Study Planner OpenEnv Baseline Agent")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--all", action="store_true", help="Run all difficulties")
    parser.add_argument("--base-url", default="http://localhost:7860", help="API base URL")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    print(f"Study Planner OpenEnv - Baseline Agent")
    print(f"API: {BASE_URL}")

    try:
        requests.get(f"{BASE_URL}/health").raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach API at {BASE_URL}")
        print(f"  Make sure inference.py is running: python inference.py")
        raise SystemExit(1)

    if getattr(args, "all"):
        run_all()
    else:
        run_episode(args.difficulty, seed=args.seed, verbose=args.verbose)


if __name__ == "__main__":
    main()
