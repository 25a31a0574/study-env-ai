 ---
title: Study Planner OpenEnv
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: AI Study Planner environment for real-world agent training


---

# Study Planner OpenEnv

A complete, real-world **Study Planner** environment built to the [OpenEnv](https://openenv.ai) specification. An AI agent learns to manage a student's study schedule across multiple subjects with deadlines, energy constraints, and task prioritization.

---

## What It Simulates

A student must prepare for exams across 3–5 subjects within 14 days. Each subject has:
- A **total study hours needed** (varies by difficulty)
- A **deadline** (day by which they must be prepared)
- A **difficulty** level (easy / medium / hard)

The agent must:
- Allocate study time to subjects
- Manage its energy level (rest when tired)
- Create and complete tasks
- Prioritize by urgency and deadline

---

## OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| Real-world task (not a game) | ✅ Study planning |
| `POST /reset` | ✅ Initializes episode |
| `POST /step` | ✅ Executes action |
| `GET /state` | ✅ Returns full state |
| `openenv.yaml` | ✅ At repo root |
| Typed Pydantic models | ✅ Full request/response types |
| Minimum 3 tasks with agent graders | ✅ Easy / Medium / Hard |
| Scores/rewards 0.0–1.0 | ✅ Grader scores in this range |
| Meaningful reward with partial signals | ✅ Progress + completion + deadline penalties |
| Baseline inference script | ✅ `baseline_agent.py` |
| Reproducible scores with seeds | ✅ `seed` parameter in reset |
| Deploy to Hugging Face Spaces | ✅ Dockerfile at root |
| `inference.py` at repo root | ✅ |

---

## Action Space

| Action | Payload | Description |
|---|---|---|
| `study` | `subject_id`, `hours` (0.1–4.0) | Study a subject; progress proportional to energy |
| `create_task` | `subject_id`, `description`, `estimated_hours`, `priority` | Create a new task |
| `complete_task` | `task_id` | Mark a task as done (+credit to subject progress) |
| `rest` | `hours` (1.0–8.0) | Recover energy |
| `review_schedule` | _(none)_ | Reorganize schedule (costs 1 day) |

---

## Observation Space

```json
{
  "day": 3,
  "max_days": 14,
  "energy_level": 0.72,
  "subjects": [
    {
      "id": "...",
      "name": "Mathematics",
      "total_hours_needed": 20.0,
      "hours_studied": 6.0,
      "progress": 0.3,
      "difficulty": "hard",
      "deadline_day": 10,
      "days_until_deadline": 7
    }
  ],
  "pending_tasks": [...],
  "completed_tasks_count": 2,
  "total_hours_studied": 8.5,
  "overall_progress": 0.18,
  "days_remaining": 11,
  "urgent_subjects": []
}
```

---

## Reward Function

Reward is bounded in **[-1.0, +1.0]** with partial progress signals:

| Signal | Value |
|---|---|
| Progress made | `delta_progress × 5.0` |
| High-priority task completed | `+0.5` |
| Medium-priority task completed | `+0.2` |
| Low-priority task completed | `+0.1` |
| Resting when energy < 30% | `+0.3` |
| Resting when energy > 50% | `-0.1` |
| Efficient study (energy > 50%, hours ≥ 1) | `+0.1` |
| Missed deadline (per subject) | `−deficit × 1.0` |

---

## Grader Tasks

| Task | Threshold | Description |
|---|---|---|
| Easy | ≥ 0.10 | Any meaningful study activity |
| Medium | ≥ 0.50 | More than half the material covered |
| Hard | ≥ 0.80 | Near-complete coverage, minimal missed deadlines |

---

## API Usage

### Reset Environment

```bash
curl -X POST https://srisiri2074-study-env-ai.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium", "seed": 42}'
```

### Take a Step

```bash
curl -X POST https://srisiri2074-study-env-ai.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "study",
      "payload": {
        "subject_id": "<subject-id-from-reset>",
        "hours": 2.0
      }
    }
  }'
```

### Get Current State

```bash
curl https://srisiri2074-study-env-ai.hf.space/state
```

### Interactive API Docs

Visit: `https://srisiri2074-study-env-ai.hf.space/docs`

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python inference.py

# In another terminal, run the baseline agent
python baseline_agent.py --difficulty medium --seed 42
```

---

## Running with Docker

```bash
docker build -t study-planner-openenv .
docker run -p 7860:7860 study-planner-openenv
```

---

## Baseline Agent Results

The heuristic baseline policy (`baseline_agent.py`) achieves approximately:

| Difficulty | Easy Score | Medium Score | Hard Score |
|---|---|---|---|
| easy | ~1.00 | ~0.80 | ~0.30 |
| medium | ~1.00 | ~0.65 | ~0.20 |
| hard | ~1.00 | ~0.45 | ~0.15 |

Scores are reproducible with `--seed 42`.

---

## Setup on Hugging Face Spaces

1. Fork this repository to your GitHub
2. Create a new Hugging Face Space with **Docker** SDK
3. Link the Space to your GitHub repository via Settings → Repository
4. The `Dockerfile` at root builds and serves the API on port 7860 automatically

---

## Project Structure

```
.
├── inference.py          # FastAPI server (OpenEnv API: /reset, /step, /state)
├── baseline_agent.py     # Heuristic baseline agent + grader runner
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Container definition for HF Spaces
├── requirements.txt      # Python dependencies
└── README.md             # This file (also serves as HF Space card)
```

---

## License

MIT
