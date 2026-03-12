"""Swahili ASD Game API (Golden Baseline + Safe Incremental Patches)

This file is intentionally self-contained to reduce "missing import" and "duplicate model" failures.

Key goals:
- App boots reliably on Windows with: uvicorn main:app --reload
- DB schema matches the provided SQL dump (topics.key, lexicon_items.topic TEXT, etc.)
- CORS / OPTIONS preflight consistently returns 200
- Stable Auth (teacher/caregiver) + role guards
- CRUD for teacher/caregiver
- Gameplay endpoints kept stable (sessions/tasks/reports/settings)
- AI generation endpoint never crashes (falls back to template if model unavailable)
- CSV import for lexicon (create topic if missing, upsert items, return summary)
"""

# =========================
# Standard library imports
# =========================
import csv
import io
import os
import re
import json
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Literal

# =========================
# Third-party imports
# =========================
from dotenv import load_dotenv
import bcrypt
import jwt

# =========================
# FastAPI imports
# =========================
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Body,
    Request,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel, Field

# =========================
# SQLAlchemy imports
# =========================
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    SmallInteger,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    JSON as SAJSON,
    and_,
    or_,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

# =========================
# Optional GenAI imports
# =========================
_TRANSFORMERS_OK = True
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    _TRANSFORMERS_OK = False

# =========================
# ENV + APP
# =========================
load_dotenv()

app = FastAPI(title="Swahili ASD Game API", version="1.0.0")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "google/mt5-small").strip()
AI_DEBUG = os.getenv("AI_DEBUG", "0") == "1"

ASSETS_DIR = os.getenv("ASSETS_DIR", "assets").strip()
AUTO_APPROVE_AI = os.getenv("AUTO_APPROVE_AI", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")

AI_DISABLED = os.getenv("AI_DISABLED", "0") == "1"

_tokenizer = None
_model = None
_AI_LOAD_ERROR = None

def _looks_like_spm_corruption(e: Exception) -> bool:
    msg = str(e)
    return ("spiece.model" in msg and "Error parsing line" in msg) or ("SentencePiece" in msg and "Error parsing" in msg)

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ISSUER = os.getenv("JWT_ISSUER", "swahili-asd-game")
JWT_TTL_MINUTES = int(os.getenv("JWT_TTL_MINUTES", "43200"))  # 30 days default

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Put it in .env")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR_ABS = os.path.join(BASE_DIR, ASSETS_DIR)

# =========================
# CORS (development-friendly, production-safe)
# =========================
# NOTE: allow_origin_regex is used to support Expo/Metro dev origins that can vary.
# Keep allow_origins for stable local URLs.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?:\\/\\/(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Ensure assets mount does not crash if folder missing
if not os.path.isdir(ASSETS_DIR_ABS):
    os.makedirs(ASSETS_DIR_ABS, exist_ok=True)

app.mount("/assets", StaticFiles(directory=ASSETS_DIR_ABS), name="assets")

# Some clients still send OPTIONS that route-matches to app routes; CORSMiddleware handles most,
# but we keep a hard "catch-all" OPTIONS to guarantee 200.
@app.options("/{path:path}")
async def options_ok(path: str, request: Request):
    return JSONResponse(status_code=200, content={"ok": True})

# =========================
# DB setup
# =========================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================
# ORM models (match SQL dump)
# =========================

class Topic(Base):
    __tablename__ = "topics"

    id = Column(BigInteger, primary_key=True)
    key = Column(Text, nullable=False)
    label_sw = Column(Text, nullable=False)
    label_en = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LexiconItem(Base):
    __tablename__ = "lexicon_items"

    id = Column(BigInteger, primary_key=True)
    topic = Column(Text, nullable=False)  # stores Topic.key
    pos = Column(Text, nullable=False)
    difficulty = Column(SmallInteger, nullable=False)
    en_word = Column(Text, nullable=False)
    sw_word = Column(Text, nullable=False)
    example_sw = Column(Text, nullable=True)
    example_en = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)
    image_asset_id = Column(Text, nullable=True)
    audio_asset_id = Column(Text, nullable=True)


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True)
    role = Column(Text, nullable=False)  # 'teacher' | 'caregiver' | 'child' (admin optional via migration)
    full_name = Column(Text, nullable=False)
    email = Column(Text, nullable=True)
    password_hash = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)





class Caregiver(Base):
    __tablename__ = "caregivers"

    # NOTE: In the provided DB dump, caregivers.id is a separate PK referenced by children.caregiver_id.
    # To keep the system simple and avoid schema changes, we mirror users.id into caregivers.id for caregiver accounts.
    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    full_name = Column(Text, nullable=False)
    phone = Column(Text, nullable=True)
    email = Column(Text, nullable=True)


class Child(Base):
    __tablename__ = "children"

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    caregiver_id = Column(BigInteger, nullable=True)  # we treat this as users.id for caregiver
    display_name = Column(Text, nullable=False)
    age_years = Column(SmallInteger, nullable=True)
    min_level = Column(SmallInteger, nullable=False)
    max_level = Column(SmallInteger, nullable=False)
    current_level = Column(SmallInteger, nullable=False)
    notes = Column(Text, nullable=True)


class CaregiverSettings(Base):
    __tablename__ = "caregiver_settings"

    child_id = Column(BigInteger, primary_key=True)
    session_minutes = Column(Integer, nullable=False, default=10)
    sound_on = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(BigInteger, primary_key=True)
    child_id = Column(BigInteger, nullable=False)
    lesson_focus = Column(Text, nullable=False)  # Topic.key
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    current_modality = Column(Text, nullable=False, server_default="mixed")
    low_stim_enabled = Column(Boolean, nullable=False, server_default="false")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(BigInteger, primary_key=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    task_type = Column(Text, nullable=False)
    target_skill = Column(Text, nullable=False)
    difficulty = Column(SmallInteger, nullable=False)
    modality = Column(Text, nullable=False)  # 'text' | 'image' | 'audio' | 'mixed'
    payload_json = Column(SAJSON, nullable=False)
    generated_by = Column(Text, nullable=False)  # 'template' | 'ai'
    approved = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TaskAttempt(Base):
    __tablename__ = "task_attempts"

    id = Column(BigInteger, primary_key=True)
    task_id = Column(BigInteger, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    child_id = Column(BigInteger, nullable=False)
    answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    retries = Column(SmallInteger, nullable=False, default=0)
    skipped = Column(Boolean, nullable=False, default=False)
    hint_used = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    taps = Column(Integer, nullable=False, default=0)
    timeouts = Column(Integer, nullable=False, default=0)
    pauses = Column(Integer, nullable=False, default=0)
    audio_muted = Column(Boolean, nullable=False, default=False)
    abandon_mid_task = Column(Boolean, nullable=False, default=False)


class Mastery(Base):
    __tablename__ = "mastery"

    child_id = Column(BigInteger, primary_key=True)
    lexicon_item_id = Column(BigInteger, primary_key=True)
    mastery_score = Column(Numeric(4, 3), nullable=False, default=0.0)
    correct_count = Column(Integer, nullable=False, default=0)
    wrong_count = Column(Integer, nullable=False, default=0)
    last_seen = Column(DateTime(timezone=True), nullable=True)


class AIGenerationLog(Base):
    __tablename__ = "ai_generation_logs"

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    topic = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)
    input_prompt = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    validation_ok = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)


# =========================
# Pydantic schemas
# =========================
Role = Literal["teacher", "caregiver", "child", "admin"]


class HealthResponse(BaseModel):
    ok: bool
    time: str


class LoginRequest(BaseModel):
    email: str
    password: str
    role: Optional[Literal["teacher", "caregiver"]] = None


class LoginResponse(BaseModel):
    token: str
    user: Dict[str, Any]


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    role: Literal["teacher", "caregiver"]


class TopicIn(BaseModel):
    key: str = Field(..., min_length=1)
    label_sw: str = Field(..., min_length=1)
    label_en: Optional[str] = None
    is_active: bool = True


class TopicOut(TopicIn):
    id: int
    created_at: datetime


class LexiconIn(BaseModel):
    topic: str
    pos: str = "noun"
    difficulty: int = Field(1, ge=1, le=5)
    en_word: str
    sw_word: str
    example_sw: Optional[str] = None
    example_en: Optional[str] = None
    tags: Optional[str] = None
    image_asset_id: Optional[str] = None
    audio_asset_id: Optional[str] = None


class LexiconOut(LexiconIn):
    id: int


class ChildIn(BaseModel):
    display_name: str
    age_years: Optional[int] = None
    min_level: int = 1
    max_level: int = 5
    current_level: int = 1
    notes: Optional[str] = None


class ChildOut(ChildIn):
    id: int
    caregiver_id: Optional[int]
    created_at: datetime


class PublicChildOut(BaseModel):
    id: int
    display_name: str
    age_years: Optional[int] = None
    current_level: Optional[int] = None


class StartSessionRequest(BaseModel):
    child_id: int
    lesson_focus: str


class StartSessionResponse(BaseModel):
    session_id: int
    task_id: int
    task: Dict[str, Any]


class NextTaskRequest(BaseModel):
    session_id: int
    child_id: int


class NextTaskResponse(BaseModel):
    task_id: int
    task: Dict[str, Any]


class SubmitTaskRequest(BaseModel):
    session_id: int
    task_id: int
    child_id: int
    answer: Optional[str] = None
    response_time_ms: int = 0
    skipped: bool = False
    hint_used: bool = False
    retries: int = 0
    taps: int = 0
    timeouts: int = 0
    pauses: int = 0
    audio_muted: bool = False
    abandon_mid_task: bool = False


class SubmitTaskResponse(BaseModel):
    is_correct: bool
    feedback_sw: str
    reward: Dict[str, Any]


class ChildReportResponse(BaseModel):
    child_id: int
    total_attempts: int
    correct: int
    accuracy: float
    avg_response_time_ms: int = 0
    current_streak: int = 0
    points_last_7_days: List[Dict[str, Any]] = []
    modality_breakdown: List[Dict[str, Any]] = []


class SettingsResponse(BaseModel):
    child_id: int
    session_minutes: int
    sound_on: bool


class SettingsUpdateRequest(BaseModel):
    child_id: int
    session_minutes: int = Field(..., ge=1, le=180)
    sound_on: bool


class AIGenerateTaskRequest(BaseModel):
    topic: str
    target_lexicon_id: int
    task_type: Optional[str] = "match_word"
    max_words: Optional[int] = 4
    modality: Optional[Literal["text", "image", "audio", "mixed"]] = "mixed"


class CSVImportTextRequest(BaseModel):
    csv_text: str


class CSVImportSummary(BaseModel):
    created_topics: int
    created_items: int
    updated_items: int
    skipped_duplicates: int
    errors: List[str] = []


# =========================
# Auth helpers
# =========================
security = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def _create_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "name": user.full_name,
        "email": user.email,
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_TTL_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db=Depends(get_db),
) -> Optional[User]:
    if not creds or not creds.credentials:
        return None
    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], issuer=JWT_ISSUER)
        user_id = int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_role(*roles: str):
    def _guard(user: Optional[User] = Depends(get_current_user)) -> User:
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return _guard


# =========================
# GenAI (lazy + safe)
# =========================
_tokenizer = None
_model = None


def _get_mt5():
    global _tokenizer, _model, _AI_LOAD_ERROR
    if AI_DISABLED:
        _AI_LOAD_ERROR = "AI_DISABLED=1"
        return None, None
    if not _TRANSFORMERS_OK:
        _AI_LOAD_ERROR = "transformers not installed"
        return None, None

    if _tokenizer is None or _model is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            _AI_LOAD_ERROR = None
        except Exception as e:
            _AI_LOAD_ERROR = f"{type(e).__name__}: {e}"
            _tokenizer = None
            _model = None
            if _looks_like_spm_corruption(e):
                print("[AI] SentencePiece cache appears corrupted (spiece.model).")
                print(r"[AI] Fix: delete C:\Users\<YOU>\.cache\huggingface\hub\models--google--mt5-small and restart.")
            else:
                print("[AI] Failed to load model:", _AI_LOAD_ERROR)

    return _tokenizer, _model



def mt5_generate(prompt: str, max_new_tokens: int = 80) -> Optional[str]:
    tok, mdl = _get_mt5()
    if tok is None or mdl is None:
        return None
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    outputs = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        early_stopping=True,
    )
    return tok.decode(outputs[0], skip_special_tokens=True).strip()


def fallback_task(topic: str = "general", task_type: str = "match_word", target=None, notes: str = None):
    """Always returns a valid task payload so gameplay never crashes."""
    target = target or {"lexicon_id": None, "label_sw": "mbwa", "label_en": "dog"}
    return {
        "topic": topic,
        "task_type": task_type,
        "prompt_sw": "Chagua neno sahihi.",
        "prompt_en": "Choose the correct word.",
        "target": target,
        "options": [
            {"label_sw": "mbwa", "label_en": "dog", "lexicon_id": target.get("lexicon_id")},
            {"label_sw": "paka", "label_en": "cat", "lexicon_id": None},
            {"label_sw": "ng'ombe", "label_en": "cow", "lexicon_id": None},
        ],
        "answer": "mbwa",
        "meta": {"source": "fallback", "ai_error": _AI_LOAD_ERROR, "notes": notes},
    }


# =========================
# Utility: task builder
# =========================

def _asset_url(asset_id: Optional[str], kind: Literal["images", "audio"]):
    if not asset_id:
        return None
    ext = ".png" if kind == "images" else ".mp3"
    return f"/assets/{kind}/{asset_id}{ext}"


def _build_match_word_task(topic_key: str, target: LexiconItem, options: List[LexiconItem]) -> Dict[str, Any]:
    tiles = []
    for li in options:
        tiles.append(
            {
                "lexicon_id": int(li.id),
                "label_sw": li.sw_word,
                "label_en": li.en_word,
                "image_url": _asset_url(li.image_asset_id, "images"),
                "audio_url": _asset_url(li.audio_asset_id, "audio"),
            }
        )

    random.shuffle(tiles)

    return {
        "topic": topic_key,
        "task_type": "match_word",
        "prompt_sw": f"Chagua neno sahihi: {target.sw_word}",
        "prompt_en": f"Choose the correct word: {target.en_word}",
        "prompt_image_url": _asset_url(target.image_asset_id, "images"),
        "prompt_audio_url": _asset_url(target.audio_asset_id, "audio"),
        "target": {
            "lexicon_id": int(target.id),
            "label_sw": target.sw_word,
            "label_en": target.en_word,
            "image_url": _asset_url(target.image_asset_id, "images"),
            "audio_url": _asset_url(target.audio_asset_id, "audio"),
        },
        "options": tiles,
        "answer": str(target.id),
        "feedback": {
            "correct_sw": "Vizuri sana!",
            "wrong_sw": "Jaribu tena.",
        },
    }


def _pick_lexicon_options(db, topic_key: str, difficulty: int, n: int, target_id: int) -> List[LexiconItem]:
    q = db.query(LexiconItem).filter(
        LexiconItem.topic == topic_key,
        LexiconItem.difficulty == difficulty,
    )
    items = q.all()
    if not items:
        # fallback to any difficulty within topic
        items = db.query(LexiconItem).filter(LexiconItem.topic == topic_key).all()

    if not items:
        raise HTTPException(status_code=400, detail=f"No lexicon items found for topic '{topic_key}'")

    # Ensure target is included
    by_id = {int(x.id): x for x in items}
    target = by_id.get(target_id) or db.query(LexiconItem).filter(LexiconItem.id == target_id).first()
    if not target:
        target = random.choice(items)

    # sample options (ensure target in options)
    pool = [x for x in items if int(x.id) != int(target.id)]
    options = random.sample(pool, k=min(max(0, n - 1), len(pool)))
    options.append(target)
    return target, options


# =========================
# Routes
# =========================

@app.get("/health", response_model=HealthResponse)
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


def _ensure_caregiver_row(db, user: User) -> None:
    """Ensure a caregivers row exists for a caregiver user.

    The DB dump uses: children.caregiver_id -> caregivers.id (FK).
    To avoid changing the schema, we mirror users.id into caregivers.id for caregiver accounts.
    """
    if user.role != "caregiver":
        return

    existing = db.get(Caregiver, int(user.id))
    if existing:
        # keep basic profile fields in sync (safe, non-destructive)
        if user.full_name and existing.full_name != user.full_name:
            existing.full_name = user.full_name
        if user.email and existing.email != user.email:
            existing.email = user.email
        return

    caregiver = Caregiver(
        id=int(user.id),
        full_name=user.full_name or "Caregiver",
        email=user.email,
    )
    db.add(caregiver)
    # commit handled by caller

    # If a sequence exists, bump it forward so later inserts without explicit IDs don't collide.
    try:
        db.execute(text("""SELECT setval('public.caregivers_id_seq', GREATEST((SELECT COALESCE(MAX(id), 1) FROM public.caregivers), 1));"""))
    except Exception:
        # Some dumps may not use a sequence name like this; safe to ignore.
        pass


# -------- Auth --------

@app.post("/auth/register", response_model=LoginResponse)
def register(req: RegisterRequest, db=Depends(get_db)):
    email_norm = req.email.strip().lower()
    if db.query(User).filter(User.email == email_norm).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        role=req.role,
        full_name=req.full_name.strip(),
        email=email_norm,
        password_hash=_hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Ensure caregiver profile row exists for FK integrity
    _ensure_caregiver_row(db, user)
    db.commit()

    token = _create_token(user)
    return {"token": token, "user": {"id": int(user.id), "role": user.role, "full_name": user.full_name, "email": user.email}}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, db=Depends(get_db)):
    email_norm = req.email.strip().lower()
    # Role is optional; if provided we validate it after fetching by email.
    user = db.query(User).filter(User.email == email_norm).first()
    if user and req.role and user.role != req.role:
        user = None
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not _verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Ensure caregiver profile row exists for FK integrity
    _ensure_caregiver_row(db, user)
    db.commit()

    token = _create_token(user)
    return {"token": token, "user": {"id": int(user.id), "role": user.role, "full_name": user.full_name, "email": user.email}}


# -------- Teacher: Topics CRUD --------

@app.get("/teacher/topics", response_model=List[TopicOut])
def teacher_list_topics(db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    rows = db.query(Topic).order_by(Topic.key.asc()).all()
    return [
        {
            "id": int(t.id),
            "key": t.key,
            "label_sw": t.label_sw,
            "label_en": t.label_en,
            "is_active": bool(t.is_active),
            "created_at": t.created_at,
        }
        for t in rows
    ]


@app.post("/teacher/topics", response_model=TopicOut)
def teacher_create_topic(payload: TopicIn, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    key = payload.key.strip()
    if db.query(Topic).filter(Topic.key == key).first():
        raise HTTPException(status_code=409, detail="Topic key already exists")

    row = Topic(
        key=key,
        label_sw=payload.label_sw.strip(),
        label_en=(payload.label_en.strip() if payload.label_en else None),
        is_active=payload.is_active,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": int(row.id),
        "key": row.key,
        "label_sw": row.label_sw,
        "label_en": row.label_en,
        "is_active": bool(row.is_active),
        "created_at": row.created_at,
    }


@app.put("/teacher/topics/{topic_key}", response_model=TopicOut)
def teacher_update_topic(topic_key: str, payload: TopicIn, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    row = db.query(Topic).filter(Topic.key == topic_key).first()
    if not row:
        raise HTTPException(status_code=404, detail="Topic not found")

    # For safety: do not allow key change through this endpoint
    row.label_sw = payload.label_sw.strip()
    row.label_en = payload.label_en.strip() if payload.label_en else None
    row.is_active = payload.is_active

    db.commit()
    db.refresh(row)

    return {
        "id": int(row.id),
        "key": row.key,
        "label_sw": row.label_sw,
        "label_en": row.label_en,
        "is_active": bool(row.is_active),
        "created_at": row.created_at,
    }


@app.delete("/teacher/topics/{topic_key}")
def teacher_delete_topic(topic_key: str, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    row = db.query(Topic).filter(Topic.key == topic_key).first()
    if not row:
        raise HTTPException(status_code=404, detail="Topic not found")

    # Safety: do not cascade delete lexicon items automatically.
    # Only delete if no lexicon items reference this topic key.
    in_use = db.query(LexiconItem).filter(LexiconItem.topic == topic_key).first()
    if in_use:
        raise HTTPException(status_code=400, detail="Topic has lexicon items; delete/move them first")

    db.delete(row)
    db.commit()
    return {"ok": True}


# -------- Teacher: Lexicon CRUD --------

@app.get("/teacher/lexicon", response_model=List[LexiconOut])
def teacher_list_lexicon(topic: Optional[str] = None, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    q = db.query(LexiconItem)
    if topic:
        q = q.filter(LexiconItem.topic == topic)
    rows = q.order_by(LexiconItem.topic.asc(), LexiconItem.difficulty.asc(), LexiconItem.en_word.asc()).all()
    return [
        {
            "id": int(x.id),
            "topic": x.topic,
            "pos": x.pos,
            "difficulty": int(x.difficulty),
            "en_word": x.en_word,
            "sw_word": x.sw_word,
            "example_sw": x.example_sw,
            "example_en": x.example_en,
            "tags": x.tags,
            "image_asset_id": x.image_asset_id,
            "audio_asset_id": x.audio_asset_id,
        }
        for x in rows
    ]


@app.post("/teacher/lexicon", response_model=LexiconOut)
def teacher_create_lexicon(payload: LexiconIn, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    topic_key = payload.topic.strip()
    # Ensure topic exists
    if not db.query(Topic).filter(Topic.key == topic_key).first():
        # auto-create topic as inactive with best-effort labels
        t = Topic(key=topic_key, label_sw=topic_key, label_en=topic_key, is_active=True)
        db.add(t)
        db.commit()

    # Avoid duplicates: (topic, en_word, sw_word) is treated as a natural key
    exists = db.query(LexiconItem).filter(
        LexiconItem.topic == topic_key,
        func.lower(LexiconItem.en_word) == payload.en_word.strip().lower(),
        func.lower(LexiconItem.sw_word) == payload.sw_word.strip().lower(),
    ).first()
    if exists:
        raise HTTPException(status_code=409, detail="Lexicon item already exists")

    row = LexiconItem(
        topic=topic_key,
        pos=payload.pos.strip(),
        difficulty=payload.difficulty,
        en_word=payload.en_word.strip(),
        sw_word=payload.sw_word.strip(),
        example_sw=payload.example_sw,
        example_en=payload.example_en,
        tags=payload.tags,
        image_asset_id=payload.image_asset_id,
        audio_asset_id=payload.audio_asset_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "id": int(row.id),
        **payload.model_dump(),
    }


@app.put("/teacher/lexicon/{item_id}", response_model=LexiconOut)
def teacher_update_lexicon(item_id: int, payload: LexiconIn, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    row = db.query(LexiconItem).filter(LexiconItem.id == item_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Lexicon item not found")

    topic_key = payload.topic.strip()
    if not db.query(Topic).filter(Topic.key == topic_key).first():
        t = Topic(key=topic_key, label_sw=topic_key, label_en=topic_key, is_active=True)
        db.add(t)
        db.commit()

    row.topic = topic_key
    row.pos = payload.pos.strip()
    row.difficulty = payload.difficulty
    row.en_word = payload.en_word.strip()
    row.sw_word = payload.sw_word.strip()
    row.example_sw = payload.example_sw
    row.example_en = payload.example_en
    row.tags = payload.tags
    row.image_asset_id = payload.image_asset_id
    row.audio_asset_id = payload.audio_asset_id

    db.commit()
    db.refresh(row)

    return {"id": int(row.id), **payload.model_dump()}


@app.delete("/teacher/lexicon/{item_id}")
def teacher_delete_lexicon(item_id: int, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    row = db.query(LexiconItem).filter(LexiconItem.id == item_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Lexicon item not found")
    db.delete(row)
    db.commit()
    return {"ok": True}


# -------- Teacher: CSV Import --------


def _normalize_topic_key(topic_raw: str) -> str:
    t = (topic_raw or "").strip()
    # convert to a key-like string (spaces -> underscores)
    t = re.sub(r"\s+", "_", t)
    return t


def _import_csv_rows(db, rows: List[Dict[str, str]]) -> CSVImportSummary:
    created_topics = 0
    created_items = 0
    updated_items = 0
    skipped_duplicates = 0
    errors: List[str] = []

    # cache topics in memory for speed
    existing_topics = {t.key: t for t in db.query(Topic).all()}

    for idx, r in enumerate(rows, start=1):
        try:
            en = (r.get("english") or r.get("en") or "").strip()
            sw = (r.get("swahili") or r.get("sw") or "").strip()
            topic_raw = (r.get("topic") or "").strip()
            image_url = (r.get("image_url") or "").strip() or None

            if not en or not sw or not topic_raw:
                errors.append(f"Row {idx}: missing required columns (english, swahili, topic)")
                continue

            topic_key = _normalize_topic_key(topic_raw)

            if topic_key not in existing_topics:
                t = Topic(key=topic_key, label_sw=topic_raw, label_en=topic_raw, is_active=True)
                db.add(t)
                db.commit()
                db.refresh(t)
                existing_topics[topic_key] = t
                created_topics += 1

            # upsert by (topic, en_word, sw_word)
            existing = db.query(LexiconItem).filter(
                LexiconItem.topic == topic_key,
                func.lower(LexiconItem.en_word) == en.lower(),
                func.lower(LexiconItem.sw_word) == sw.lower(),
            ).first()

            if existing:
                # If image_url provided, we store it in tags for now (non-breaking), unless you add a column later.
                # This keeps migrations additive.
                if image_url:
                    existing.tags = (existing.tags or "")
                    if "image_url=" not in (existing.tags or ""):
                        existing.tags = (existing.tags + (";" if existing.tags else "") + f"image_url={image_url}")
                db.commit()
                updated_items += 1
                continue

            li = LexiconItem(
                topic=topic_key,
                pos="noun",
                difficulty=1,
                en_word=en,
                sw_word=sw,
                tags=(f"image_url={image_url}" if image_url else None),
            )
            db.add(li)
            db.commit()
            created_items += 1

        except Exception as e:
            db.rollback()
            errors.append(f"Row {idx}: {type(e).__name__}: {str(e)}")

    return CSVImportSummary(
        created_topics=created_topics,
        created_items=created_items,
        updated_items=updated_items,
        skipped_duplicates=skipped_duplicates,
        errors=errors,
    )


@app.post("/teacher/lexicon/import-csv", response_model=CSVImportSummary)
def teacher_import_csv_text(
    payload: CSVImportTextRequest = Body(...),
    db=Depends(get_db),
    user: User = Depends(require_role("teacher")),
):
    csv_text = payload.csv_text
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty or invalid")
    return _import_csv_rows(db, rows)


@app.post("/teacher/lexicon/import-csv-file", response_model=CSVImportSummary)
def teacher_import_csv_file(
    file: UploadFile = File(...),
    db=Depends(get_db),
    user: User = Depends(require_role("teacher")),
):
    raw = file.file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("utf-8", errors="ignore")

    f = io.StringIO(text)
    reader = csv.DictReader(f)
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty or invalid")
    return _import_csv_rows(db, rows)


# -------- Caregiver: Children CRUD --------

@app.get("/caregiver/children", response_model=List[ChildOut])
def caregiver_list_children(db=Depends(get_db), user: User = Depends(require_role("caregiver"))):
    rows = db.query(Child).filter(Child.caregiver_id == user.id).order_by(Child.created_at.desc()).all()
    return [
        {
            "id": int(c.id),
            "caregiver_id": int(c.caregiver_id) if c.caregiver_id is not None else None,
            "display_name": c.display_name,
            "age_years": int(c.age_years) if c.age_years is not None else None,
            "min_level": int(c.min_level),
            "max_level": int(c.max_level),
            "current_level": int(c.current_level),
            "notes": c.notes,
            "created_at": c.created_at,
        }
        for c in rows
    ]


@app.post("/caregiver/children", response_model=ChildOut)
def caregiver_create_child(payload: ChildIn, db=Depends(get_db), user: User = Depends(require_role("caregiver"))):
    row = Child(
        caregiver_id=user.id,
        display_name=payload.display_name.strip(),
        age_years=payload.age_years,
        min_level=payload.min_level,
        max_level=payload.max_level,
        current_level=payload.current_level,
        notes=payload.notes,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": int(row.id),
        "caregiver_id": int(row.caregiver_id) if row.caregiver_id is not None else None,
        "display_name": row.display_name,
        "age_years": int(row.age_years) if row.age_years is not None else None,
        "min_level": int(row.min_level),
        "max_level": int(row.max_level),
        "current_level": int(row.current_level),
        "notes": row.notes,
        "created_at": row.created_at,
    }


@app.put("/caregiver/children/{child_id}", response_model=ChildOut)
def caregiver_update_child(child_id: int, payload: ChildIn, db=Depends(get_db), user: User = Depends(require_role("caregiver"))):
    row = db.query(Child).filter(Child.id == child_id, Child.caregiver_id == user.id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Child not found")

    row.display_name = payload.display_name.strip()
    row.age_years = payload.age_years
    row.min_level = payload.min_level
    row.max_level = payload.max_level
    row.current_level = payload.current_level
    row.notes = payload.notes

    db.commit()
    db.refresh(row)
    return {
        "id": int(row.id),
        "caregiver_id": int(row.caregiver_id) if row.caregiver_id is not None else None,
        "display_name": row.display_name,
        "age_years": int(row.age_years) if row.age_years is not None else None,
        "min_level": int(row.min_level),
        "max_level": int(row.max_level),
        "current_level": int(row.current_level),
        "notes": row.notes,
        "created_at": row.created_at,
    }


@app.delete("/caregiver/children/{child_id}")
def caregiver_delete_child(child_id: int, db=Depends(get_db), user: User = Depends(require_role("caregiver"))):
    row = db.query(Child).filter(Child.id == child_id, Child.caregiver_id == user.id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Child not found")
    db.delete(row)
    db.commit()
    return {"ok": True}


# -------- Caregiver Settings (used by RN) --------


@app.get("/children/public", response_model=List[PublicChildOut])
def children_public(limit: int = 50, db: Session = Depends(get_db)):
    # MVP: no auth, return minimal non-sensitive fields only
    rows = (
        db.query(Child)
        .order_by(Child.created_at.desc() if hasattr(Child, "created_at") else Child.id.desc())
        .limit(limit)
        .all()
    )
    return [
        PublicChildOut(
            id=r.id,
            display_name=r.display_name,
            age_years=getattr(r, "age_years", None),
            current_level=getattr(r, "current_level", None),
        )
        for r in rows
    ]


@app.get("/caregiver/settings/{child_id}", response_model=SettingsResponse)
def get_settings(child_id: int, db=Depends(get_db)):
    row = db.query(CaregiverSettings).filter(CaregiverSettings.child_id == child_id).first()
    if not row:
        # Create default settings on first read (non-breaking)
        row = CaregiverSettings(child_id=child_id, session_minutes=10, sound_on=True)
        db.add(row)
        db.commit()
        db.refresh(row)
    return {"child_id": int(row.child_id), "session_minutes": int(row.session_minutes), "sound_on": bool(row.sound_on)}


@app.post("/caregiver/settings")
def update_settings(payload: SettingsUpdateRequest, db=Depends(get_db)):
    row = db.query(CaregiverSettings).filter(CaregiverSettings.child_id == payload.child_id).first()
    if not row:
        row = CaregiverSettings(child_id=payload.child_id, session_minutes=payload.session_minutes, sound_on=payload.sound_on)
        db.add(row)
        db.commit()
        return {"ok": True}

    row.session_minutes = payload.session_minutes
    row.sound_on = payload.sound_on
    db.commit()
    return {"ok": True}


# -------- Gameplay: Sessions & Tasks (kept stable for existing app) --------

@app.post("/sessions/start", response_model=StartSessionResponse)
def start_session(payload: StartSessionRequest, db=Depends(get_db)):
    # Validate child exists
    child = db.query(Child).filter(Child.id == payload.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Validate lesson focus topic exists OR still allow if lexicon exists
    topic_key = payload.lesson_focus.strip()

    session = Session(child_id=payload.child_id, lesson_focus=topic_key)
    db.add(session)
    db.commit()
    db.refresh(session)

    # Create first task
    task_id, task_payload = _create_next_task(db, session.id, payload.child_id)
    return {"session_id": int(session.id), "task_id": int(task_id), "task": task_payload}


def _create_next_task(db, session_id: int, child_id: int) -> (int, Dict[str, Any]):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # adaptive difficulty: clamp to child current_level
    child = db.query(Child).filter(Child.id == child_id).first()
    difficulty = int(child.current_level) if child else 1
    difficulty = max(1, min(5, difficulty))

    # pick a target lexicon item (prefer unseen / low mastery)
    topic_key = session.lesson_focus

    # Fetch candidates
    candidates = db.query(LexiconItem).filter(LexiconItem.topic == topic_key).all()
    if not candidates:
        # fallback: any lexicon items
        candidates = db.query(LexiconItem).all()
        if not candidates:
            raise HTTPException(status_code=400, detail="No lexicon items available")

    target = random.choice(candidates)

    # Build options
    target, options = _pick_lexicon_options(db, target.topic, difficulty, 4, int(target.id))
    task_payload = _build_match_word_task(target.topic, target, options)

    task = Task(
        session_id=session_id,
        task_type=task_payload["task_type"],
        target_skill="vocabulary",
        difficulty=difficulty,
        modality="mixed",
        payload_json=task_payload,
        generated_by="template",
        approved=True,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    return int(task.id), task_payload


@app.post("/tasks/next", response_model=NextTaskResponse)
def next_task(payload: NextTaskRequest, db=Depends(get_db)):
    task_id, task_payload = _create_next_task(db, payload.session_id, payload.child_id)
    return {"task_id": int(task_id), "task": task_payload}


@app.post("/tasks/submit", response_model=SubmitTaskResponse)
def submit_task(payload: SubmitTaskRequest, db=Depends(get_db)):
    task = db.query(Task).filter(Task.id == payload.task_id, Task.session_id == payload.session_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    correct_answer = None
    try:
        correct_answer = str((task.payload_json or {}).get("answer"))
    except Exception:
        correct_answer = None

    is_correct = (not payload.skipped) and (correct_answer is not None) and (str(payload.answer) == correct_answer)

    attempt = TaskAttempt(
        task_id=payload.task_id,
        child_id=payload.child_id,
        answer=str(payload.answer) if payload.answer is not None else None,
        is_correct=bool(is_correct),
        response_time_ms=max(0, int(payload.response_time_ms or 0)),
        retries=max(0, int(payload.retries or 0)),
        skipped=bool(payload.skipped),
        hint_used=bool(payload.hint_used),
        taps=max(0, int(payload.taps or 0)),
        timeouts=max(0, int(payload.timeouts or 0)),
        pauses=max(0, int(payload.pauses or 0)),
        audio_muted=bool(payload.audio_muted),
        abandon_mid_task=bool(payload.abandon_mid_task),
    )
    db.add(attempt)

    # Update mastery best-effort
    try:
        target_id = int(correct_answer) if correct_answer and correct_answer.isdigit() else None
        if target_id:
            m = db.query(Mastery).filter(Mastery.child_id == payload.child_id, Mastery.lexicon_item_id == target_id).first()
            if not m:
                m = Mastery(child_id=payload.child_id, lexicon_item_id=target_id, mastery_score=0.0, correct_count=0, wrong_count=0)
                db.add(m)

            if is_correct:
                m.correct_count += 1
                m.mastery_score = min(1.0, float(m.mastery_score) + 0.05)
            else:
                m.wrong_count += 1
                m.mastery_score = max(0.0, float(m.mastery_score) - 0.02)
            m.last_seen = datetime.now(timezone.utc)

            # Light level progression
            child = db.query(Child).filter(Child.id == payload.child_id).first()
            if child and is_correct:
                child.current_level = min(child.max_level, int(child.current_level) + 0)
    except Exception:
        pass

    db.commit()

    feedback_sw = "Vizuri sana!" if is_correct else "Jaribu tena."
    reward = {"coins": 1 if is_correct else 0, "streak": 0}

    return {"is_correct": bool(is_correct), "feedback_sw": feedback_sw, "reward": reward}


# -------- Reports --------

@app.get("/reports/child/{child_id}", response_model=ChildReportResponse)
def child_report(child_id: int, db=Depends(get_db)):
    attempts = (
        db.query(TaskAttempt, Task)
        .join(Task, Task.id == TaskAttempt.task_id)
        .filter(TaskAttempt.child_id == child_id)
        .order_by(TaskAttempt.created_at.asc(), TaskAttempt.id.asc())
        .all()
    )

    total = len(attempts)
    correct = sum(1 for attempt, _task in attempts if bool(attempt.is_correct))
    accuracy = (correct / total) if total else 0.0

    avg_response_time_ms = 0
    if total:
        avg_response_time_ms = int(round(sum(max(0, int(attempt.response_time_ms or 0)) for attempt, _task in attempts) / total))

    current_streak = 0
    for attempt, _task in reversed(attempts):
        if bool(attempt.is_correct):
            current_streak += 1
        else:
            break

    today = datetime.now(timezone.utc).date()
    points_last_7_days = []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        pts = 0
        for attempt, _task in attempts:
            created = attempt.created_at
            if not created:
                continue
            created_day = created.date() if hasattr(created, "date") else created
            if created_day == day and bool(attempt.is_correct):
                pts += 1
        points_last_7_days.append({
            "date": day.isoformat(),
            "label": day.strftime("%a"),
            "points": int(pts),
        })

    modality_rollup: Dict[str, Dict[str, Any]] = {}
    for attempt, task in attempts:
        modality = (getattr(task, "modality", None) or "mixed").strip() or "mixed"
        bucket = modality_rollup.setdefault(modality, {"modality": modality, "attempts": 0, "correct": 0})
        bucket["attempts"] += 1
        if bool(attempt.is_correct):
            bucket["correct"] += 1

    modality_breakdown = []
    for item in modality_rollup.values():
        item["accuracy"] = round((item["correct"] / item["attempts"]) if item["attempts"] else 0.0, 4)
        modality_breakdown.append(item)
    modality_breakdown.sort(key=lambda x: x["modality"])

    return {
        "child_id": int(child_id),
        "total_attempts": int(total),
        "correct": int(correct),
        "accuracy": float(round(accuracy, 4)),
        "avg_response_time_ms": int(avg_response_time_ms),
        "current_streak": int(current_streak),
        "points_last_7_days": points_last_7_days,
        "modality_breakdown": modality_breakdown,
    }


# -------- AI Task Generation (stable, never crashes) --------

@app.post("/ai/generate-task")
def ai_generate_task(
    req: AIGenerateTaskRequest,
    db=Depends(get_db),
    user: User = Depends(require_role("teacher")),
):
    """Generate a task using mT5 if available; otherwise return a deterministic fallback.

    This endpoint must NEVER crash gameplay flows.
    It returns a payload compatible with the RN client.
    """
    # Resolve topic first (must follow the user's selection)
    topic_key = (req.topic or "").strip() or "general"

    # Resolve target lexicon:
    # - If target_lexicon_id <= 0, pick a random item from the selected topic.
    # - If target_lexicon_id is provided but does not belong to the selected topic, re-pick from the selected topic.
    target = None
    if int(req.target_lexicon_id or 0) > 0:
        target = db.query(LexiconItem).filter(LexiconItem.id == req.target_lexicon_id).first()
        if not target:
            raise HTTPException(status_code=404, detail="target_lexicon_id not found")

    if target is None or (getattr(target, "topic", None) and topic_key and str(target.topic) != str(topic_key)):
        # Pick a target from the selected topic
        candidates = db.query(LexiconItem).filter(LexiconItem.topic == topic_key).all()
        if not candidates:
            # fallback to any lexicon if the topic has none
            candidates = db.query(LexiconItem).all()
        if not candidates:
            raise HTTPException(status_code=400, detail="No lexicon items available to generate a task")
        target = random.choice(candidates)

    # Ensure topic exists (additive, safe)
    if not db.query(Topic).filter(Topic.key == topic_key).first():
        t = Topic(key=topic_key, label_sw=topic_key, label_en=topic_key, is_active=True)
        db.add(t)
        db.commit()

    # Build prompt (simple + explicit JSON expectation)
    prompt = (
        "Create a simple ASD-friendly vocabulary matching task. "
        "Return ONLY valid JSON (no markdown). "
        "JSON fields: prompt_sw, prompt_en, options (array of {label_sw,label_en}), answer (string). "
        f"Topic: {topic_key}. Target: {target.en_word} / {target.sw_word}."
    )

    ai_text = None
    notes = None

    # 1) Generate with AI safely
    try:
        ai_text = mt5_generate(prompt)
        # mT5 sometimes returns placeholders like "<extra_id_0>" when the prompt/output is unsuitable.
        # Treat that as unusable and fall back.
        if ai_text and ai_text.strip().startswith("<extra_id_"):
            ai_text = None
    except Exception as e:
        notes = (f"AI error: {type(e).__name__}: {str(e)}" if AI_DEBUG else None)
        ai_text = None

    # 2) Parse AI JSON safely
    task_payload = None
    if ai_text:
        try:
            # Extract JSON object if the model adds leading/trailing text
            m = re.search(r"\{[\s\S]*\}", ai_text)
            maybe_json = m.group(0) if m else ai_text
            parsed = json.loads(maybe_json)

            if isinstance(parsed, dict) and isinstance(parsed.get("options"), list) and parsed.get("answer") is not None:
                # Normalize options
                norm_opts = []
                for o in parsed.get("options", []):
                    if isinstance(o, dict):
                        ls = o.get("label_sw") or o.get("sw") or o.get("sw_word") or ""
                        le = o.get("label_en") or o.get("en") or o.get("en_word") or ""
                        if ls or le:
                            norm_opts.append({"label_sw": ls, "label_en": le})
                    elif isinstance(o, str):
                        norm_opts.append({"label_sw": o, "label_en": ""})

                if norm_opts:
                    task_payload = {
                        "topic": topic_key,
                        "task_type": req.task_type or "match_word",
                        "prompt_sw": parsed.get("prompt_sw") or f"Chagua neno sahihi: {target.sw_word}",
                        "prompt_en": parsed.get("prompt_en") or f"Choose the correct word: {target.en_word}",
                        "target": {
                            "lexicon_id": int(target.id),
                            "label_sw": target.sw_word,
                            "label_en": target.en_word,
                        },
                        "options": norm_opts,
                        "answer": str(parsed.get("answer")),
                        "source": "mt5",
                    }
        except Exception as e:
            notes = (f"AI parse error: {type(e).__name__}: {str(e)}" if AI_DEBUG else None)
            task_payload = None

    # 3) Fallback if AI missing / parse failed
    if not task_payload:
        tgt, opts = _pick_lexicon_options(
            db,
            topic_key,
            difficulty=1,
            n=int(req.max_words or 4),
            target_id=int(target.id),
        )
        task_payload = _build_match_word_task(topic_key, tgt, opts)
        task_payload["source"] = "fallback"

    # Add ASD-friendly visual cue prompts (safe additive fields).
    # Frontend can ignore these if not needed.
    try:
        tkey = (topic_key or "").strip().lower()
        if tkey in ("songs", "song", "nyimbo"):
            # A simple "song sequence" with visual cues for each stage.
            task_payload["task_type"] = "song_sequence"
            task_payload["song_title_sw"] = "Wimbo wa Vitendo (Actions Song)"
            task_payload["song_steps"] = [
                {"step": 1, "text_sw": "Mtoto anakula", "image_prompt": "Picha ya mtoto akila chakula mezani, mtindo wa cartoon rahisi, rangi tulivu."},
                {"step": 2, "text_sw": "Mtoto anakunywa maji", "image_prompt": "Picha ya mtoto anakunywa maji kwenye kikombe, cartoon rahisi."},
                {"step": 3, "text_sw": "Mtoto anapiga makofi", "image_prompt": "Picha ya mtoto akipiga makofi kwa furaha, cartoon rahisi."},
                {"step": 4, "text_sw": "Mtoto anaruka", "image_prompt": "Picha ya mtoto akiruka juu, cartoon rahisi."},
            ]
        elif tkey in ("food", "chakula"):
            # Add per-option visual cue prompts to reduce instruction difficulty.
            cues = []
            for opt in task_payload.get("options", []):
                en = opt.get("label_en") or ""
                sw = opt.get("label_sw") or ""
                if en or sw:
                    cues.append({
                        "lexicon_id": opt.get("lexicon_id"),
                        "image_prompt": f"Picha rahisi ya {sw or en} kwenye mazingira ya kawaida, mtindo wa cartoon, rangi tulivu."
                    })
            if cues:
                task_payload["visual_cues"] = cues
    except Exception:
        pass

    log_id = None

    # 4) Log generation (best-effort; never fail request)
    try:
        log = AIGenerationLog(
            topic=topic_key,
            model_name=MODEL_NAME,
            raw_output=ai_text,
            notes=notes,
            is_valid=True,
        )
        db.add(log)
        db.commit()
        try:
            db.refresh(log)
        except Exception:
            pass
        log_id = getattr(log, "id", None)
    except Exception:
        db.rollback()

    # Return in the "review wrapper" shape expected by the teacher UI.
    # Keep the original task payload unchanged inside "task".
    return {
        "ok": True,
        "validation_ok": True,
        "task_id": log_id,
        "approval": "auto",
        "approved": True,
        "generated_text": (ai_text.strip() if ai_text and ai_text.strip() else f"Kazi imetengenezwa kwa mada: {topic_key}."),
        "task": task_payload,
        "notes": notes,
    }

@app.api_route("/topics", methods=["GET","OPTIONS","HEAD"])
@app.api_route("/topics/", methods=["GET","OPTIONS","HEAD"])
def public_topics(db=Depends(get_db)):
    """Public topics list for gameplay / lightweight clients.

    Supports both /topics and /topics/ to avoid 405/redirect edge-cases in clients.
    """
    rows = db.query(Topic).filter(Topic.is_active == True).order_by(Topic.key.asc()).all()  # noqa: E712
    return [{"key": r.key, "label_en": r.label_en, "label_sw": r.label_sw} for r in rows]





@app.api_route("/teacher/ai/pending", methods=["GET","OPTIONS","HEAD"])
def teacher_ai_pending(limit: int = 30, db=Depends(get_db), user: User = Depends(require_role("teacher"))):
    """List recent AI generations that need review.
    Pending := validation_ok is false OR notes exist (notes shown only with AI_DEBUG).
    """
    limit = int(limit or 30)
    q = db.query(AIGenerationLog).order_by(AIGenerationLog.id.desc()).limit(limit).all()
    items = []
    for row in q:
        pending = (not bool(row.validation_ok)) or (row.notes is not None and str(row.notes).strip() != "")
        if pending:
            items.append({
                "id": int(row.id),
                "topic": row.topic,
                "model_name": row.model_name,
                "created_at": row.created_at.isoformat() if getattr(row, "created_at", None) else None,
                "validation_ok": bool(row.validation_ok),
                "notes": row.notes,
                "output_text": row.output_text,
            })
    return {"ok": True, "items": items}
