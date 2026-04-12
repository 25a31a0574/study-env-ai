fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
python-multipart==0.0.20
openai>=1.0.0
requests
2. inference.py (Complete File with LLM Integration)
"""
Study Planner OpenEnv - inference.py
FastAPI server implementing the OpenEnv spec: reset(), step(), state() API.
"""

from __future__ import annotations

import json
import random
import time
import uuid
import os
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import OpenAI client for LLM integration
from openai import OpenAI

# Initialize OpenAI client with Scalar Hackathon's LiteLLM proxy
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)

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


# ─── Global State ─────────────────────────────────────────────────────────────

_state: Optional[StudyPlannerState] = None


# ─── LLM Integration ──────────────────────────────────────────────────────────

def _generate_llm_feedback(state: StudyPlannerState) -> str:
    """
    Use LLM to generate personalized study feedback.
    This makes the required API call to Scalar's LiteLLM proxy.
    """
    try:
        total_needed = sum(s.total_hours_needed for s in state.subjects)
        total_studied = sum(s.hours_studied for s in state.subjects)
        progress = total_studied / total_needed if total_needed > 0 else 0.0
        
        prompt = f"""You are a study advisor. A student is on day {state.day} of {state.max_days}.
They have {len(state.subjects)} subjects and overall progress is {progress:.1%}.
Energy level: {state.energy_level:.1%}.

Provide brief motivational feedback (max 2 sentences)."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Keep up the good work! (LLM unavailable: {str(e)})"


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _initialize_subjects(config: Dict) -> List[Subject]:
    subjects = []
    for s_config in config["subjects"]:
        subject = Subject(
            id=str(uuid.uuid4()),
            name=s_config["name"],
            total_hours_needed=s_config["total_hours"],
            difficulty=s_config["difficulty"],
            deadline_day=s_config["deadline_day"],
        )
        subjects.append(subject)
    return subjects


def _initialize_tasks(subjects: List[Subject], tasks_per_subject: int) -> List[Task]:
    tasks = []
    for subject in subjects:
        base_templates = TASK_TEMPLATES.get(subject.name, ["Study this subject", "Review material", "Practice problems"])
        templates = base_templates * 10  # ensure enough
        for i in range(tasks_per_subject):
            task = Task(
                id=str(uuid.uuid4()),
                subject_id=subject.id,
                description=templates[i % len(templates)],
                estimated_hours=round(random.uniform(0.5, 2.5), 1),
                priority=random.choice(["low", "medium", "high"]),
            )
            tasks.append(task)
    return tasks


def _make_observation(state: StudyPlannerState) -> Observation:
    total_needed = sum(s.total_hours_needed for s in state.subjects)
    total_studied = sum(s.hours_studied for s in state.subjects)
    overall_progress = total_studied / total_needed if total_needed > 0 else 0.0

    subjects_data = []
    for s in state.subjects:
        progress = s.hours_studied / s.total_hours_needed if s.total_hours_needed > 0 else 0.0
        subjects_data.append({
            "id": s.id,
            "name": s.name,
            "total_hours_needed": s.total_hours_needed,
            "hours_studied": round(s.hours_studied, 2),
            "progress": round(progress, 4),
            "difficulty": s.difficulty,
            "deadline_day": s.deadline_day,
            "days_until_deadline": max(0, s.deadline_day - state.day),
        })

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

    urgent_subjects = [
        s.name for s in state.subjects
        if (s.deadline_day - state.day <= 3) and (s.hours_studied / s.total_hours_needed < 0.8)
    ]

    return Observation(
        day=state.day,
        max_days=state.max_days,
        energy_level=round(state.energy_level, 4),
        subjects=subjects_data,
        pending_tasks=pending_tasks,
        completed_tasks_count=completed_count,
        total_hours_studied=round(state.total_hours_studied, 2),
        overall_progress=round(overall_progress, 4),
        days_remaining=state.max_days - state.day,
        urgent_subjects=urgent_subjects,
    )


def _apply_action(state: StudyPlannerState, action: Action) -> str:
    if action.action_type == "study":
        payload = action.payload or {}
        subject_id = payload.get("subject_id")
        hours = payload.get("hours", 1.0)
        
        subject = next((s for s in state.subjects if s.id == subject_id), None)
        if not subject:
            return "Invalid subject_id"
        
        hours = max(0.1, min(hours, 4.0))
        effective_hours = hours * state.energy_level
        subject.hours_studied += effective_hours
        state.total_hours_studied += effective_hours
        state.energy_level = max(0.0, state.energy_level - hours * 0.15)
        state.day += max(1, int(hours / 2))
        
        return f"Studied {subject.name} for {hours:.1f}h (effective: {effective_hours:.1f}h)"

    elif action.action_type == "create_task":
        payload = action.payload or {}
        subject_id = payload.get("subject_id")
        description = payload.get("description", "New task")
        estimated_hours = payload.get("estimated_hours", 1.0)
        priority = payload.get("priority", "medium")
        
        task = Task(
            id=str(uuid.uuid4()),
            subject_id=subject_id,
            description=description,
            estimated_hours=estimated_hours,
            priority=priority,
        )
        state.tasks.append(task)
        return f"Created task: {description}"

    elif action.action_type == "complete_task":
        payload = action.payload or {}
        task_id = payload.get("task_id")
        
        task = next((t for t in state.tasks if t.id == task_id), None)
        if not task or task.completed:
            return "Invalid or already completed task"
        
        task.completed = True
        subject = next((s for s in state.subjects if s.id == task.subject_id), None)
        if subject:
            bonus_hours = task.estimated_hours * 0.8
            subject.hours_studied += bonus_hours
            state.total_hours_studied += bonus_hours
        
        state.day += 1
        return f"Completed task: {task.description}"

    elif action.action_type == "rest":
        payload = action.payload or {}
        hours = payload.get("hours", 4.0)
        hours = max(1.0, min(hours, 8.0))
        
        recovery = hours * 0.12
        state.energy_level = min(1.0, state.energy_level + recovery)
        state.day += max(1, int(hours / 4))
        
        return f"Rested for {hours:.1f}h, energy now {state.energy_level:.2f}"

    elif action.action_type == "review_schedule":
        state.day += 1
        return "Reviewed and reorganized study schedule"

    return "Unknown action"


def _compute_reward(state: StudyPlannerState, action: Action, prev_progress: float, new_progress: float) -> float:
    reward = 0.0
    delta_progress = new_progress - prev_progress
    
    if delta_progress > 0:
        reward += delta_progress * 5.0
    
    if action.action_type == "complete_task":
        task_id = action.payload.get("task_id") if action.payload else None
        task = next((t for t in state.tasks if t.id == task_id), None)
        if task:
            if task.priority == "high":
                reward += 0.5
            elif task.priority == "medium":
                reward += 0.2
            else:
                reward += 0.1
    
    if action.action_type == "rest":
        if state.energy_level < 0.3:
            reward += 0.3
        elif state.energy_level > 0.5:
            reward -= 0.1
    
    if action.action_type == "study":
        if state.energy_level > 0.5 and action.payload.get("hours", 0) >= 1.0:
            reward += 0.1
    
    for subject in state.subjects:
        if state.day > subject.deadline_day:
            deficit = subject.total_hours_needed - subject.hours_studied
            if deficit > 0:
                reward -= deficit * 0.1
    
    return max(-1.0, min(1.0, reward))


def _check_done(state: StudyPlannerState) -> bool:
    if state.day >= state.max_days:
        return True
    
    all_complete = all(
        s.hours_studied >= s.total_hours_needed
        for s in state.subjects
    )
    return all_complete


def _compute_final_score(state: StudyPlannerState) -> Dict[str, Any]:
    total_needed = sum(s.total_hours_needed for s in state.subjects)
    total_studied = sum(s.hours_studied for s in state.subjects)
    overall_progress = total_studied / total_needed if total_needed > 0 else 0.0
    
    completed_count = sum(1 for t in state.tasks if t.completed)
    total_tasks = len(state.tasks)
    tasks_ratio = completed_count / total_tasks if total_tasks > 0 else 0.0
    
    on_time_count = sum(
        1 for s in state.subjects
        if s.hours_studied >= s.total_hours_needed and state.day <= s.deadline_day
    )
    on_time_ratio = on_time_count / len(state.subjects) if state.subjects else 0.0
    
    easy_score = min(1.0, overall_progress * 2.0)
    medium_score = min(1.0, (overall_progress * 0.7 + tasks_ratio * 0.3))
    hard_score = min(1.0, (overall_progress * 0.5 + on_time_ratio * 0.5))
    
    return {
        "easy": round(easy_score, 4),
        "medium": round(medium_score, 4),
        "hard": round(hard_score, 4),
        "overall_progress": round(overall_progress, 4),
        "tasks_completed_ratio": round(tasks_ratio, 4),
        "on_time_ratio": round(on_time_ratio, 4),
        "days_used": state.day,
        "total_hours_studied": round(state.total_hours_studied, 2),
    }


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Study Planner OpenEnv",
    description="AI-powered study planning environment for OpenEnv hackathon",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Study Planner OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """Initialize a new episode with the specified difficulty and seed."""
    global _state
    
    difficulty = req.difficulty or "medium"
    seed = req.seed
    
    if seed is not None:
        random.seed(seed)
    
    config = DIFFICULTY_CONFIGS[difficulty]
    
    subjects = _initialize_subjects(config)
    tasks = _initialize_tasks(subjects, config["initial_tasks_per_subject"])
    
    _state = StudyPlannerState(
        episode_id=str(uuid.uuid4()),
        day=1,
        max_days=config["max_days"],
        subjects=subjects,
        tasks=tasks,
        energy_level=1.0,
    )
    
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
    
    message = _apply_action(_state, req.action)
    
    total_studied_after = sum(s.hours_studied for s in _state.subjects)
    new_progress = total_studied_after / total_needed if total_needed > 0 else 0.0
    
    reward = _compute_reward(_state, req.action, prev_progress, new_progress)
    _state.score += reward
    _state.done = _check_done(_state)
    
    obs = _make_observation(_state)
    
    # Generate LLM feedback (makes required API call to Scalar's proxy)
    llm_feedback = _generate_llm_feedback(_state)
    
    info: Dict[str, Any] = {
        "message": message,
        "llm_feedback": llm_feedback,
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
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            content = f.read()
        from fastapi.responses import Response
        return Response(content=content, media_type="text/yaml")
    raise HTTPException(status_code=404, detail="openenv.yaml not found")


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
