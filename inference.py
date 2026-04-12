"""
Study Planner OpenEnv - inference.py
FastAPI server implementing the OpenEnv spec: reset(), step(), state() API.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# ─── Domain Models ────────────────────────────────────────────────────────────

class Subject(BaseModel):
    id: str
    name: str
    total_hours_needed: float
    hours_studied: float = 0.0
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    deadline_day: int = 7  # days from episode start


class Task(BaseModel):
    id: str
    subject_id: str
    description: str
    estimated_hours: float
    completed: bool = False
    priority: Literal["low", "medium", "high"] = "medium"


class StudySession(BaseModel):
    task_id: str
    hours: float
    quality: Literal["poor", "average", "good", "excellent"] = "average"


# ─── Environment State ─────────────────────────────────────────────────────────

class StudyPlannerState(BaseModel):
    episode_id: str
    day: int = 1
    max_days: int = 14
    subjects: List[Subject] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    completed_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    total_hours_studied: float = 0.0
    energy_level: float = 1.0  # 0.0 to 1.0
    score: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── Action Models ────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: Literal["study", "create_task", "complete_task", "rest", "review_schedule"]
    payload: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    difficulty: Optional[Literal["easy", "medium", "hard"]] = "medium"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Action


# ─── Response Models ──────────────────────────────────────────────────────────

class Observation(BaseModel):
    day: int
    max_days: int
    energy_level: float
    subjects: List[Dict[str, Any]]
    pending_tasks: List[Dict[str, Any]]
    completed_tasks_count: int
    total_hours_studied: float
    overall_progress: float  # 0.0 to 1.0
    days_remaining: int
    urgent_subjects: List[str]  # subjects with deadlines approaching


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any]


# ─── Grade Configurations ─────────────────────────────────────────────────────

DIFFICULTY_CONFIGS = {
    "easy": {
        "subjects": [
            {"name": "Mathematics", "total_hours": 10.0, "difficulty": "easy", "deadline_day": 14},
            {"name": "English Literature", "total_hours": 8.0, "difficulty": "easy", "deadline_day": 14},
            {"name": "History", "total_hours": 6.0, "difficulty": "easy", "deadline_day": 12},
        ],
        "max_days": 14,
        "initial_tasks_per_subject": 2,
    },
    "medium": {
        "subjects": [
            {"name": "Mathematics", "total_hours": 20.0, "difficulty": "hard", "deadline_day": 10},
            {"name": "Physics", "total_hours": 15.0, "difficulty": "medium", "deadline_day": 12},
            {"name": "Chemistry", "total_hours": 12.0, "difficulty": "medium", "deadline_day": 14},
            {"name": "English", "total_hours": 8.0, "difficulty": "easy", "deadline_day": 14},
        ],
        "max_days": 14,
        "initial_tasks_per_subject": 3,
    },
    "hard": {
        "subjects": [
            {"name": "Advanced Mathematics", "total_hours": 30.0, "difficulty": "hard", "deadline_day": 8},
            {"name": "Quantum Physics", "total_hours": 25.0, "difficulty": "hard", "deadline_day": 10},
            {"name": "Organic Chemistry", "total_hours": 20.0, "difficulty": "hard", "deadline_day": 9},
            {"name": "Computer Science", "total_hours": 18.0, "difficulty": "medium", "deadline_day": 12},
            {"name": "Statistics", "total_hours": 15.0, "difficulty": "medium", "deadline_day": 11},
        ],
        "max_days": 14,
        "initial_tasks_per_subject": 4,
    },
}

TASK_TEMPLATES = {
    "Mathematics": ["Solve practice problems", "Review theorems", "Complete exercises", "Study formulas", "Work on proofs"],
    "Physics": ["Study concepts", "Solve numerical problems", "Review lab notes", "Practice derivations", "Read textbook chapter"],
    "Chemistry": ["Learn reactions", "Practice equations", "Review periodic table", "Study mechanisms", "Complete worksheets"],
    "English Literature": ["Read assigned texts", "Write essay outline", "Analyze themes", "Review grammar rules", "Prepare notes"],
    "History": ["Review timeline", "Study key events", "Analyze primary sources", "Make flashcards", "Write summaries"],
    "Computer Science": ["Code practice problems", "Review algorithms", "Study data structures", "Debug programs", "Read documentation"],
    "Statistics": ["Solve probability problems", "Review distributions", "Practice hypothesis testing", "Study formulas", "Work on datasets"],
    "English": ["Grammar exercises", "Vocabulary review", "Writing practice", "Reading comprehension", "Essay planning"],
    "Advanced Mathematics": ["Prove theorems", "Solve complex problems", "Review analysis concepts", "Study topology", "Work on problem sets"],
    "Quantum Physics": ["Study wave functions", "Review Schrodinger equation", "Practice bra-ket notation", "Solve quantum problems", "Review postulates"],
    "Organic Chemistry": ["Study reaction mechanisms", "Practice synthesis routes", "Review functional groups", "Solve retrosynthesis", "Study stereochemistry"],
}


# ─── Environment Core ─────────────────────────────────────────────────────────

_state: Optional[StudyPlannerState] = None


def _make_observation(state: StudyPlannerState) -> Observation:
    pending_tasks = [
        {
            "id": t.id,
            "subject_id": t.subject_id,
            "description": t.description,
            "estimated_hours": t.estimated_hours,
            "priority": t.priority,
        }
        for t in state.tasks if not t.completed
    ]
    completed_count = sum(1 for t in state.tasks if t.completed)

    total_needed = sum(s.total_hours_needed for s in state.subjects)
    total_studied = sum(s.hours_studied for s in state.subjects)
    overall_progress = (total_studied / total_needed) if total_needed > 0 else 0.0
    overall_progress = min(overall_progress, 1.0)

    urgent_subjects = [
        s.name for s in state.subjects
        if (s.deadline_day - state.day) <= 2 and (s.hours_studied / s.total_hours_needed) < 0.8
    ]

    return Observation(
        day=state.day,
        max_days=state.max_days,
        energy_level=state.energy_level,
        subjects=[
            {
                "id": s.id,
                "name": s.name,
                "total_hours_needed": s.total_hours_needed,
                "hours_studied": round(s.hours_studied, 2),
                "progress": round(s.hours_studied / s.total_hours_needed, 3),
                "difficulty": s.difficulty,
                "deadline_day": s.deadline_day,
                "days_until_deadline": max(0, s.deadline_day - state.day),
            }
            for s in state.subjects
        ],
        pending_tasks=pending_tasks,
        completed_tasks_count=completed_count,
        total_hours_studied=round(state.total_hours_studied, 2),
        overall_progress=round(overall_progress, 3),
        days_remaining=max(0, state.max_days - state.day),
        urgent_subjects=urgent_subjects,
    )


def _compute_reward(
    state: StudyPlannerState, action: Action, prev_progress: float, new_progress: float
) -> float:
    """
    Reward function with partial progress signals.
    """
    reward = 0.0

    progress_delta = new_progress - prev_progress
    reward += progress_delta * 5.0

    if action.action_type == "complete_task":
        task_id = (action.payload or {}).get("task_id")
        task = next((t for t in state.tasks if t.id == task_id), None)
        if task and task.completed:
            priority_bonus = {"low": 0.1, "medium": 0.2, "high": 0.5}.get(task.priority, 0.1)
            reward += priority_bonus

    if action.action_type == "rest":
        if state.energy_level < 0.3:
            reward += 0.3
        else:
            reward -= 0.1

    if action.action_type == "study":
        hours = (action.payload or {}).get("hours", 1.0)
        if state.energy_level > 0.5 and hours >= 1.0:
            reward += 0.1

    for subject in state.subjects:
        days_left = subject.deadline_day - state.day
        progress = subject.hours_studied / subject.total_hours_needed
        if days_left <= 0 and progress < 1.0:
            deficit = 1.0 - progress
            reward -= deficit * 1.0

    reward = float(max(-1.0, min(1.0, reward)))
    return round(reward, 4)


def _initialize_state(difficulty: str, seed: Optional[int]) -> StudyPlannerState:
    if seed is not None:
        random.seed(seed)

    config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["medium"])
    episode_id = str(uuid.uuid4())

    subjects = []
    tasks = []

    for subj_conf in config["subjects"]:
        subj_id = str(uuid.uuid4())
        subject = Subject(
            id=subj_id,
            name=subj_conf["name"],
            total_hours_needed=subj_conf["total_hours"],
            difficulty=subj_conf["difficulty"],
            deadline_day=subj_conf["deadline_day"],
        )
        subjects.append(subject)

        templates = TASK_TEMPLATES.get(subj_conf["name"], ["Study material", "Review notes", "Practice problems"])
        n_tasks = config["initial_tasks_per_subject"]
        selected = templates[:n_tasks] if len(templates) >= n_tasks else templates

        for i, desc in enumerate(selected):
            priority_map = ["high", "medium", "low"]
            task = Task(
                id=str(uuid.uuid4()),
                subject_id=subj_id,
                description=desc,
                estimated_hours=round(subj_conf["total_hours"] / n_tasks, 1),
                priority=priority_map[i % 3],
            )
            tasks.append(task)

    return StudyPlannerState(
        episode_id=episode_id,
        day=1,
        max_days=config["max_days"],
        subjects=subjects,
        tasks=tasks,
        energy_level=1.0,
        info={"difficulty": difficulty, "seed": seed},
    )


def _apply_action(state: StudyPlannerState, action: Action) -> str:
    """Apply action to state in-place and return a message string."""
    payload = action.payload or {}
    message = ""

    if action.action_type == "study":
        subject_id = payload.get("subject_id")
        hours = float(payload.get("hours", 1.0))
        hours = max(0.1, min(hours, 4.0))

        subject = next((s for s in state.subjects if s.id == subject_id), None)
        if subject is None:
            raise HTTPException(status_code=400, detail=f"Subject '{subject_id}' not found")

        if state.energy_level < 0.1:
            raise HTTPException(status_code=400, detail="Energy too low to study. Rest first.")

        quality_multiplier = 0.7 + (state.energy_level * 0.6)
        effective_hours = hours * quality_multiplier
        subject.hours_studied = min(subject.hours_studied + effective_hours, subject.total_hours_needed)
        state.total_hours_studied += effective_hours

        state.energy_level = max(0.0, state.energy_level - (hours * 0.15))
        state.day += 1
        message = f"Studied {subject.name} for {hours}h (effective: {round(effective_hours, 2)}h)"

    elif action.action_type == "create_task":
        subject_id = payload.get("subject_id")
        description = payload.get("description", "New task")
        estimated_hours = float(payload.get("estimated_hours", 1.0))
        priority = payload.get("priority", "medium")

        subject = next((s for s in state.subjects if s.id == subject_id), None)
        if subject is None:
            raise HTTPException(status_code=400, detail=f"Subject '{subject_id}' not found")

        new_task = Task(
            id=str(uuid.uuid4()),
            subject_id=subject_id,
            description=description,
            estimated_hours=estimated_hours,
            priority=priority,
        )
        state.tasks.append(new_task)
        message = f"Created task: {description} for {subject.name}"

    elif action.action_type == "complete_task":
        task_id = payload.get("task_id")
        task = next((t for t in state.tasks if t.id == task_id), None)
        if task is None:
            raise HTTPException(status_code=400, detail=f"Task '{task_id}' not found")
        if task.completed:
            message = f"Task already completed: {task.description}"
        else:
            task.completed = True
            subject = next((s for s in state.subjects if s.id == task.subject_id), None)
            subj_name = subject.name if subject else "Unknown"
            message = f"Completed task: {task.description} ({subj_name})"

    elif action.action_type == "rest":
        rest_recovery = 0.3
        state.energy_level = min(1.0, state.energy_level + rest_recovery)
        state.day += 1
        message = f"Rested. Energy restored to {round(state.energy_level, 2)}"

    elif action.action_type == "review_schedule":
        pending_count = sum(1 for t in state.tasks if not t.completed)
        message = (
            f"Schedule review: Day {state.day}/{state.max_days}, "
            f"{pending_count} pending tasks, energy={round(state.energy_level, 2)}"
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action type: {action.action_type}")

    return message


def _check_done(state: StudyPlannerState) -> bool:
    if state.day > state.max_days:
        return True
    total_needed = sum(s.total_hours_needed for s in state.subjects)
    total_studied = sum(s.hours_studied for s in state.subjects)
    if total_needed > 0 and (total_studied / total_needed) >= 1.0:
        return True
    return False


def _compute_final_score(state: StudyPlannerState) -> Dict[str, float]:
    total_needed = sum(s.total_hours_needed for s in state.subjects)
    total_studied = sum(s.hours_studied for s in state.subjects)
    completion_ratio = (total_studied / total_needed) if total_needed > 0 else 0.0
    completion_ratio = min(completion_ratio, 1.0)

    tasks_completed = sum(1 for t in state.tasks if t.completed)
    tasks_total = len(state.tasks)
    task_ratio = (tasks_completed / tasks_total) if tasks_total > 0 else 0.0

    deadline_score = 0.0
    for subject in state.subjects:
        progress = subject.hours_studied / subject.total_hours_needed
        if state.day <= subject.deadline_day:
            deadline_score += progress
        else:
            deadline_score += progress * 0.5
    deadline_score = deadline_score / len(state.subjects) if state.subjects else 0.0

    easy_score = min(1.0, completion_ratio)
    medium_score = min(1.0, (completion_ratio * 0.6) + (task_ratio * 0.2) + (deadline_score * 0.2))
    hard_score = min(1.0, (completion_ratio * 0.5) + (task_ratio * 0.2) + (deadline_score * 0.3))

    return {
        "easy": round(easy_score, 4),
        "medium": round(medium_score, 4),
        "hard": round(hard_score, 4),
        "completion_ratio": round(completion_ratio, 4),
        "task_completion_ratio": round(task_ratio, 4),
        "deadline_score": round(deadline_score, 4),
    }


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Study Planner OpenEnv",
    description="An OpenEnv-compatible study planner environment for AI agent training.",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """Initialize or reset the environment for a new episode."""
    global _state
    difficulty = req.difficulty or "medium"
    if difficulty not in DIFFICULTY_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown difficulty: {difficulty}")
    try:
        _state = _initialize_state(difficulty, req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize state: {str(e)}")

    obs = _make_observation(_state)
    return ResetResponse(
        observation=obs,
        info={
            "episode_id": _state.episode_id,
            "difficulty": req.difficulty,
            "seed": req.seed,
            "message": "Episode initialized successfully",
        },
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Execute one action in the environment."""
    global _state
    if _state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if _state.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new episode.")

    total_needed = sum(s.total_hours_needed for s in _state.subjects)
    total_studied_before = sum(s.hours_studied for s in _state.subjects)
    prev_progress = total_studied_before / total_needed if total_needed > 0 else 0.0

    try:
        message = _apply_action(_state, req.action)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action failed: {str(e)}")

    total_studied_after = sum(s.hours_studied for s in _state.subjects)
    new_progress = total_studied_after / total_needed if total_needed > 0 else 0.0

    reward = _compute_reward(_state, req.action, prev_progress, new_progress)
    _state.score += reward
    _state.done = _check_done(_state)

    obs = _make_observation(_state)

    info: Dict[str, Any] = {
        "message": message,
        "cumulative_score": round(_state.score, 4),
        "episode_id": _state.episode_id,
    }

    if _state.done:
        final_scores = _compute_final_score(_state)
        info["final_scores"] = final_scores
        info["grader"] = {
            "easy": {"score": final_scores["easy"], "passed": final_scores["easy"] >= 0.1},
            "medium": {"score": final_scores["medium"], "passed": final_scores["medium"] >= 0.5},
            "hard": {"score": final_scores["hard"], "passed": final_scores["hard"] >= 0.8},
        }

    return StepResponse(observation=obs, reward=reward, done=_state.done, info=info)


@app.get("/state")
def state():
    """Get the current raw environment state."""
    global _state
    if _state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _state.model_dump()


@app.get("/openenv.yaml", include_in_schema=False)
def serve_openenv_yaml():
    """Serve the openenv.yaml spec file."""
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            content = f.read()
        return Response(content=content, media_type="text/yaml")
    raise HTTPException(status_code=404, detail="openenv.yaml not found")


