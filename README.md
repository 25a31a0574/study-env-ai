# AI Study Planner Environment

## Problem Statement
Students often struggle to manage time effectively across multiple subjects. This project simulates a study planning environment where an AI agent learns to optimize study schedules.

## Solution
I created a reinforcement learning-style environment where:
- The agent selects actions (study subject or take break)
- The system updates state (time, remaining tasks, energy)
- Rewards are assigned based on productivity

## Features
- 3 difficulty levels:
  - Easy: 3 subjects
  - Medium: 5 subjects
  - Hard: Includes energy management
- Reward system:
  - +1 for completing a task
  - -1 for invalid action
  - -0.5 for breaks

## How to Run

```bash
python inference.py
