import os
import json
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, Text, Boolean, SmallInteger,
    DateTime, ForeignKey, Numeric, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func

# -------------------------
# Optional GenAI imports
# -------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# ENV + APP
# =========================
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "google/mt5-small")
ASSETS_DIR = os.getenv("ASSETS_DIR", "assets")
AUTO_APPROVE_AI = os.getenv("AUTO_APPROVE_AI", "false").lower() == "true"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Put it in .env")

app = FastAPI(title="Swahili ASD Game Backend", version="0.1.0")

# Serve static assets (images/audio)
# Access images like: GET /assets/images/img_dog.png
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


# =========================
# DB SETUP (SQLAlchemy)
# =========================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# =========================
# DB MODELS (minimal subset)
# =========================
class LexiconItem(Base):
    __tablename__ = "lexicon_items"

    id = Column(BigInteger, primary_key=True)
    topic = Column(Text, nullable=False)
    pos = Column(Text, nullable=False)
    difficulty = Column(SmallInteger, nullable=False)
    en_word = Column(Text, nullable=False)
    sw_word = Column(Text, nullable=False)
    example_sw = Column(Text)
    example_en = Column(Text)
    tags = Column(Text)
    image_asset_id = Column(Text)   # asset_key
    audio_asset_id = Column(Text)   # asset_key


class Session(Base):
    __tablename__ = "sessions"

    id = Column(BigInteger, primary_key=True)
    child_id = Column(BigInteger, nullable=False)
    lesson_focus = Column(Text, nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(BigInteger, primary_key=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    task_type = Column(Text, nullable=False)
    target_skill = Column(Text, nullable=False)
    difficulty = Column(SmallInteger, nullable=False)
    modality = Column(Text, nullable=False)  # text/image/audio
    payload_json = Column(JSON, nullable=False)
    generated_by = Column(Text, nullable=False)  # template/ai
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


class Mastery(Base):
    __tablename__ = "mastery"

    child_id = Column(BigInteger, primary_key=True)
    lexicon_item_id = Column(BigInteger, primary_key=True)
    mastery_score = Column(Numeric(4, 3), nullable=False, default=0.0)  # 0..1
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
# GenAI: load once
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def mt5_generate(text: str, max_new_tokens: int = 60) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,      # predictable
        num_beams=1           # deterministic
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# =========================
# Simple validators (thesis-friendly)
# =========================
BANNED_MARKERS = ["idiom", "metaphor", "slang"]  # placeholder idea

def validate_ai_sentence(sentence: str, allowed_sw_words: set, max_words: int = 8) -> (bool, str):
    s = (sentence or "").strip()
    if not s:
        return False, "Empty output"

    # Basic length
    words = [w.strip(".,!?;:()[]\"'").lower() for w in s.split() if w.strip()]
    if len(words) > max_words:
        return False, f"Too long ({len(words)} words)"

    # Block strange markers (keeps it literal / predictable)
    lowered = s.lower()
    for b in BANNED_MARKERS:
        if b in lowered:
            return False, "Contains banned marker"

    # Vocabulary constraint: allow only known Swahili words + very common function words
    allowed_function = {"mimi", "yeye", "sisi", "wewe", "na", "ni", "hii", "hapa", "kiko", "yuko", "an", "ana", "nina", "ninakula", "ninakunywa", "ninasoma"}
    for w in words:
        if w in allowed_function:
            continue
        if w not in allowed_sw_words:
            return False, f"Out-of-lexicon word: {w}"

    return True, "OK"

def build_match_word_task(topic: str, target: LexiconItem, options: List[LexiconItem]) -> Dict[str, Any]:
    def img_url(li: LexiconItem) -> str:
        if not li.image_asset_id:
            return ""
        return f"/assets/images/{li.image_asset_id}.png"

    payload = {
        "task_type": "match_word",  # ✅ new type
        "prompt_sw": "Chagua neno sahihi.",
        "target_lexicon_id": int(target.id),
        "prompt_image_url": img_url(target),
        "options": [
            {"lexicon_id": int(o.id), "label_sw": o.sw_word}
            for o in options
        ],
        "correct_lexicon_id": int(target.id),
        "feedback": {
            "correct_sw": f"Sawa! Hiyo ni {target.sw_word}.",
            "wrong_sw": "Jaribu tena. Chagua neno sahihi."
        }
    }
    return payload
def build_sentence_builder_task(topic: str, target: LexiconItem) -> Dict[str, Any]:
    # very literal, predictable, short
    subject_tiles = ["Mimi", "Yeye"]
    verb_tiles = ["Ninakula", "Ninapenda", "Ninaona"]
    obj_tiles = [target.sw_word, "ndizi", "maji"]

    correct = ["Mimi", "Ninaona", target.sw_word]

    tiles = (
        [{"text": t, "slot": "SUBJECT"} for t in subject_tiles] +
        [{"text": t, "slot": "VERB"} for t in verb_tiles] +
        [{"text": t, "slot": "OBJECT"} for t in obj_tiles]
    )
    random.shuffle(tiles)

    return {
        "task_type": "sentence_builder",
        "prompt_sw": "Tengeneza sentensi.",
        "slots": ["SUBJECT", "VERB", "OBJECT"],
        "tiles": tiles,
        "correct": correct,
        "feedback": {"correct_sw": "Vizuri!", "wrong_sw": "Jaribu tena."}
    }

# =========================
# Adaptive logic (simple + explainable)
# =========================
def update_mastery(db, child_id: int, lexicon_item_id: int, is_correct: bool):
    m = db.query(Mastery).filter(
        Mastery.child_id == child_id,
        Mastery.lexicon_item_id == lexicon_item_id
    ).first()

    now = datetime.utcnow()
    if not m:
        m = Mastery(child_id=child_id, lexicon_item_id=lexicon_item_id, mastery_score=0.0,
                   correct_count=0, wrong_count=0, last_seen=None)
        db.add(m)

    if is_correct:
        m.correct_count += 1
        # simple mastery bump
        m.mastery_score = min(1.0, float(m.mastery_score) + 0.10)
    else:
        m.wrong_count += 1
        m.mastery_score = max(0.0, float(m.mastery_score) - 0.08)

    m.last_seen = now


def pick_next_difficulty(recent_attempts: List[TaskAttempt], current: int) -> int:
    # last 5 attempts rule
    if not recent_attempts:
        return current

    last5 = recent_attempts[-5:]
    acc = sum(1 for a in last5 if a.is_correct) / len(last5)

    if acc < 0.6:
        return max(1, current - 1)
    if acc > 0.8:
        return min(5, current + 1)
    return current


# =========================
# Task builders
# =========================
def build_match_image_task(topic: str, target: LexiconItem, options: List[LexiconItem]) -> Dict[str, Any]:
    # Build image_url from asset_key
    def img_url(li: LexiconItem) -> str:
        if not li.image_asset_id:
            return ""
        # served by StaticFiles mount
        return f"/assets/images/{li.image_asset_id}.png"

    payload = {
        "task_type": "match_image",
        "prompt_sw": f"Chagua picha ya: {target.sw_word}",
        "target_lexicon_id": int(target.id),
        "options": [
            {
                "lexicon_id": int(o.id),
                "label_sw": o.sw_word,
                "image_url": img_url(o)
            } for o in options
        ],
        "correct_lexicon_id": int(target.id),
        "feedback": {
            "correct_sw": f"Sawa! Hiyo ni {target.sw_word}.",
            "wrong_sw": f"Jaribu tena. Tafuta {target.sw_word}."
        }
    }
    return payload


def build_fill_blank_task(target: LexiconItem) -> Dict[str, Any]:
    # Very simple, literal template examples
    # Pick a safe verb phrase based on topic
    if target.topic == "food" and target.sw_word in ("maji", "maziwa"):
        prompt = "Mimi ___ " + target.sw_word + "."
        options = ["ninakunywa", "ninakula", "ninasoma"]
        correct = "ninakunywa"
    elif target.topic == "food":
        prompt = "Mimi ___ " + target.sw_word + "."
        options = ["ninakula", "ninakunywa", "ninasoma"]
        correct = "ninakula"
    else:
        prompt = "Hii ni ___ ."
        options = [target.sw_word, "mwalimu", "kitabu"]
        correct = target.sw_word

    return {
        "task_type": "fill_blank",
        "prompt_sw": prompt,
        "options": options,
        "correct": correct,
        "hint_sw": "Chagua jibu sahihi.",
        "feedback": {
            "correct_sw": "Vizuri!",
            "wrong_sw": "Jaribu tena."
        }
    }


def store_task(db, session_id: int, task_type: str, target_skill: str, difficulty: int,
               modality: str, payload: Dict[str, Any], generated_by: str, approved: bool) -> Task:
    t = Task(
        session_id=session_id,
        task_type=task_type,
        target_skill=target_skill,
        difficulty=difficulty,
        modality=modality,
        payload_json=payload,
        generated_by=generated_by,
        approved=approved
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


# =========================
# Pydantic Schemas
# =========================
class StartSessionRequest(BaseModel):
    child_id: int
    lesson_focus: str = Field(default="animals")


class StartSessionResponse(BaseModel):
    session_id: int
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


class SubmitTaskResponse(BaseModel):
    is_correct: bool
    feedback_sw: str
    reward: Dict[str, Any]


class NextTaskRequest(BaseModel):
    session_id: int
    child_id: int


class NextTaskResponse(BaseModel):
    task_id: int
    task: Dict[str, Any]


class AIGenerateTaskRequest(BaseModel):
    topic: str
    target_lexicon_id: int
    task_type: str = Field(default="match_image")  # match_image or fill_blank
    max_words: int = 8


class AIGenerateTaskResponse(BaseModel):
    generated_text: str
    validation_ok: bool
    notes: str
    task_id: Optional[int] = None
    approved: Optional[bool] = None


# =========================
# Helpers
# =========================
def get_allowed_lexicon_for_topic(db, topic: str) -> List[LexiconItem]:
    return db.query(LexiconItem).filter(LexiconItem.topic == topic).all()


def pick_target_and_options(db, topic: str) -> (LexiconItem, List[LexiconItem]):
    items = get_allowed_lexicon_for_topic(db, topic)
    items = [i for i in items if i.image_asset_id]  # for image match tasks
    if len(items) < 4:
        raise HTTPException(status_code=400, detail=f"Not enough items with images for topic '{topic}' (need 4).")

    target = random.choice(items)
    pool = [i for i in items if i.id != target.id]
    options = random.sample(pool, k=3) + [target]
    random.shuffle(options)
    return target, options


# =========================
# ENDPOINTS
# =========================
@app.post("/sessions/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    db = SessionLocal()
    try:
        s = Session(child_id=req.child_id, lesson_focus=req.lesson_focus)
        db.add(s)
        db.commit()
        db.refresh(s)

        # Prefer an approved AI task if available, else template task
        task = get_next_task_internal(db, session_id=int(s.id), child_id=req.child_id)

        return StartSessionResponse(session_id=int(s.id), task_id=int(task.id), task=task.payload_json)
    finally:
        db.close()


@app.post("/tasks/submit", response_model=SubmitTaskResponse)
def submit_task(req: SubmitTaskRequest):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == req.task_id, Task.session_id == req.session_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found for this session.")

        payload = task.payload_json
        is_correct = False

        # Determine correctness based on task_type
        ttype = payload.get("task_type")
        if ttype == "match_image":
            # answer can be lexicon_id or sw_word; prefer lexicon_id
            correct_id = payload.get("correct_lexicon_id")
            if req.answer is None:
                is_correct = False
            else:
                try:
                    ans_id = int(req.answer)
                    is_correct = (ans_id == int(correct_id))
                except ValueError:
                    # fall back compare sw_word
                    correct_word = None
                    for opt in payload.get("options", []):
                        if int(opt.get("lexicon_id")) == int(correct_id):
                            correct_word = opt.get("label_sw")
                            break
                    is_correct = (req.answer.strip().lower() == (correct_word or "").lower())

            # Update mastery using target_lexicon_id
            target_lex_id = payload.get("target_lexicon_id") or correct_id
            if target_lex_id:
                update_mastery(db, req.child_id, int(target_lex_id), is_correct)

        elif ttype == "fill_blank":
            correct = payload.get("correct")
            is_correct = (str(req.answer).strip().lower() == str(correct).strip().lower())
            # no lexicon id always available, so skip mastery update unless embedded
        elif ttype == "match_word":
            correct_id = payload.get("correct_lexicon_id")
            is_correct = False
            if req.answer:
                try:
                    is_correct = (int(req.answer) == int(correct_id))
                except ValueError:
                    is_correct = False
            target_lex_id = payload.get("target_lexicon_id") or correct_id
            if target_lex_id:
                update_mastery(db, req.child_id, int(target_lex_id), is_correct)

        elif ttype == "sentence_builder":
            # answer is JSON list string like '["Mimi","Ninaona","mbwa"]'
            is_correct = False
            try:
                ans = json.loads(req.answer or "[]")
                is_correct = (ans == payload.get("correct"))
            except Exception:
                is_correct = False
    
        else:
            # default safe behavior
            is_correct = False

        attempt = TaskAttempt(
            task_id=req.task_id,
            child_id=req.child_id,
            answer=req.answer,
            is_correct=is_correct,
            response_time_ms=req.response_time_ms,
            retries=req.retries,
            skipped=req.skipped,
            hint_used=req.hint_used
        )
        db.add(attempt)
        db.commit()

        # Feedback + rewards
        feedback = payload.get("feedback", {})
        if is_correct:
            fb = feedback.get("correct_sw", "Vizuri!")
            reward = {"stars": 1}
        else:
            fb = feedback.get("wrong_sw", "Jaribu tena.")
            reward = {"stars": 0}

        return SubmitTaskResponse(is_correct=is_correct, feedback_sw=fb, reward=reward)
    finally:
        db.close()


@app.post("/tasks/next", response_model=NextTaskResponse)
def next_task(req: NextTaskRequest):
    db = SessionLocal()
    try:
        task = get_next_task_internal(db, session_id=req.session_id, child_id=req.child_id)
        return NextTaskResponse(task_id=int(task.id), task=task.payload_json)
    finally:
        db.close()


def get_next_task_internal(db, session_id: int, child_id: int) -> Task:
    session_row = db.query(Session).filter(Session.id == session_id).first()
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found.")

    topic = session_row.lesson_focus

    # recent attempts in this session
    attempts = (
        db.query(TaskAttempt)
        .join(Task, TaskAttempt.task_id == Task.id)
        .filter(Task.session_id == session_id, TaskAttempt.child_id == child_id)
        .order_by(TaskAttempt.created_at.asc())
        .all()
    )

    # difficulty selection (default 1)
    current_diff = 1
    if attempts:
        # use last task difficulty if possible
        last_task = db.query(Task).filter(Task.id == attempts[-1].task_id).first()
        if last_task:
            current_diff = int(last_task.difficulty)

    next_diff = pick_next_difficulty(attempts, current_diff)

    # 1) Try: approved AI tasks for this topic (and close difficulty)
    ai_task = (
        db.query(Task)
        .filter(
            Task.session_id == session_id,
            Task.generated_by == "ai",
            Task.approved == True,
            Task.difficulty.between(max(1, next_diff-1), min(5, next_diff+1))
        )
        .order_by(Task.created_at.desc())
        .first()
    )
    if ai_task:
        return ai_task
    # 1b) Try pool AI tasks (teacher approved), stored under child_id=0 topic session
    pool = db.query(Session).filter(Session.child_id == 0, Session.lesson_focus == topic).first()
    if pool:
        pool_ai = (
            db.query(Task)
            .filter(
                Task.session_id == int(pool.id),
                Task.generated_by == "ai",
                Task.approved == True,
                Task.difficulty.between(max(1, next_diff-1), min(5, next_diff+1))
            )
            .order_by(Task.created_at.desc())
            .first()
        )
        if pool_ai:
            return pool_ai

    # 2) Otherwise: generate a template task and store it
    # rotate task types for MVP
    cycle = ["match_image", "match_word", "fill_blank", "sentence_builder"]
    idx = len(attempts) % len(cycle)
    next_type = cycle[idx]

    target, options = pick_target_and_options(db, topic)

    if next_type == "match_image":
        payload = build_match_image_task(topic, target, options)
        task_type = "match_image"
        modality = "image"

    elif next_type == "match_word":
        payload = build_match_word_task(topic, target, options)
        task_type = "match_word"
        modality = "mixed"

    elif next_type == "fill_blank":
        payload = build_fill_blank_task(target)
        task_type = "fill_blank"
        modality = "text"

    else:  # sentence_builder
        payload = build_sentence_builder_task(topic, target)
        task_type = "sentence_builder"
        modality = "mixed"

    return store_task(
        db=db,
        session_id=session_id,
        task_type=task_type,
        target_skill="vocabulary",
        difficulty=next_diff,
        modality=modality,
        payload=payload,
        generated_by="template",
        approved=True
    )


    return store_task(
        db=db,
        session_id=session_id,
        task_type="match_image",
        target_skill="vocabulary",
        difficulty=next_diff,
        modality="image",
        payload=payload,
        generated_by="template",
        approved=True
    )


@app.post("/ai/generate-task", response_model=AIGenerateTaskResponse)
def ai_generate_task(req: AIGenerateTaskRequest):
    """
    Teacher-controlled generation endpoint.
    Generates candidate task text using mT5, validates, then stores as Task.
    By default, stored as approved=false (teacher review), unless AUTO_APPROVE_AI=true.
    """
    db = SessionLocal()
    try:
        target = db.query(LexiconItem).filter(LexiconItem.id == req.target_lexicon_id).first()
        if not target:
            raise HTTPException(status_code=404, detail="Target lexicon item not found.")

        # Allowed vocab for topic
        items = get_allowed_lexicon_for_topic(db, req.topic)
        allowed_sw = {i.sw_word.strip().lower() for i in items if i.sw_word}

        # Build a constrained generation prompt (literal + short)
        # Note: mT5 expects text-to-text; we keep the instruction tight.
        if req.task_type == "fill_blank":
            gen_prompt = (
                f"Generate a very short Kiswahili sentence (max {req.max_words} words) "
                f"using the word '{target.sw_word}'. Make it literal and simple for a child. "
                f"Do not use idioms. Output ONLY the sentence."
            )
        else:
            # For match_image: generate a short instruction line
            gen_prompt = (
                f"Write a very short Kiswahili instruction (max {req.max_words} words) "
                f"that tells a child to choose the picture of '{target.sw_word}'. "
                f"Make it literal. Output ONLY the instruction."
            )

        generated = mt5_generate(gen_prompt, max_new_tokens=60)

        # Validate generated output
        ok, reason = validate_ai_sentence(generated, allowed_sw_words=allowed_sw, max_words=req.max_words)

        # Log generation
        log = AIGenerationLog(
            topic=req.topic,
            model_name=MODEL_NAME,
            input_prompt=gen_prompt,
            output_text=generated,
            validation_ok=ok,
            notes=reason
        )
        db.add(log)
        db.commit()

        if not ok:
            return AIGenerateTaskResponse(
                generated_text=generated,
                validation_ok=False,
                notes=reason,
                task_id=None,
                approved=None
            )

        # Store as an AI task (not tied to session yet); we’ll attach it later or keep a session_id=0 pattern.
        # To keep schema consistent, we store AI tasks under a "pool session" (session_id must exist).
        # Quick prototype solution: create or reuse a "pool" session per topic for this child_id=0.
        # Better: create a dedicated ai_tasks table. For now: pool session.
        pool_session = db.query(Session).filter(Session.child_id == 0, Session.lesson_focus == req.topic).first()
        if not pool_session:
            pool_session = Session(child_id=0, lesson_focus=req.topic)
            db.add(pool_session)
            db.commit()
            db.refresh(pool_session)

        # Build payload depending on task type
        if req.task_type == "fill_blank":
            payload = {
                "task_type": "fill_blank",
                "prompt_sw": generated if generated.endswith(".") else generated + ".",
                "options": ["ninakula", "ninakunywa", "ninasoma", target.sw_word],
                "correct": target.sw_word if "___" not in generated else None,
                "hint_sw": "Chagua jibu sahihi.",
                "feedback": {"correct_sw": "Vizuri!", "wrong_sw": "Jaribu tena."}
            }
            # If model didn’t include blank marker, convert to a blank template safely
            if "___" not in payload["prompt_sw"]:
                payload["prompt_sw"] = f"Hii ni ___ ."
                payload["options"] = [target.sw_word, "mwalimu", "kitabu"]
                payload["correct"] = target.sw_word

            task_type = "fill_blank"
            modality = "text"
            target_skill = "vocabulary"

        else:
            # match_image uses the AI-generated instruction as prompt_sw
            # options are built from lexicon items with images
            target2, options2 = pick_target_and_options(db, req.topic)
            # Force the target to be the selected lexicon_id if it has an image
            if target.image_asset_id:
                target2 = target
                # ensure target included among options
                candidates = [i for i in items if i.image_asset_id and i.id != target.id]
                if len(candidates) >= 3:
                    options2 = random.sample(candidates, 3) + [target]
                    random.shuffle(options2)

            payload = build_match_image_task(req.topic, target2, options2)
            payload["prompt_sw"] = generated  # override with AI instruction

            task_type = "match_image"
            modality = "image"
            target_skill = "vocabulary"

        stored = store_task(
            db=db,
            session_id=int(pool_session.id),
            task_type=task_type,
            target_skill=target_skill,
            difficulty=max(1, min(5, int(target.difficulty))),
            modality=modality,
            payload=payload,
            generated_by="ai",
            approved=AUTO_APPROVE_AI
        )

        return AIGenerateTaskResponse(
            generated_text=generated,
            validation_ok=True,
            notes=reason,
            task_id=int(stored.id),
            approved=bool(stored.approved)
        )
    finally:
        db.close()


@app.get("/reports/child/{child_id}")
def report_child(child_id: int):
    db = SessionLocal()
    try:
        # Simple report: accuracy per topic based on attempts
        rows = (
            db.query(TaskAttempt.is_correct, Task.session_id)
            .join(Task, TaskAttempt.task_id == Task.id)
            .filter(TaskAttempt.child_id == child_id)
            .all()
        )

        # also compute totals
        total = len(rows)
        correct = sum(1 for r in rows if r[0] is True)
        acc = (correct / total) if total else 0.0
        
        return {
            "child_id": child_id,
            "total_attempts": total,
            "correct": correct,
            "accuracy": round(acc, 3)
        }
    finally:
        db.close()
@app.get("/topics")
def list_topics():
    db = SessionLocal()
    try:
        rows = db.query(Topic).filter(Topic.is_active == True).order_by(Topic.id.asc()).all()
        return [{"key": r.key, "label_sw": r.label_sw, "label_en": r.label_en} for r in rows]
    finally:
        db.close()
@app.get("/teacher/ai/pending")
def teacher_pending_ai(topic: Optional[str] = None, limit: int = 30):
    db = SessionLocal()
    try:
        q = db.query(Task).filter(Task.generated_by == "ai", Task.approved == False)
        if topic:
            # topic is stored in the pool session lesson_focus
            q = q.join(Session, Task.session_id == Session.id).filter(Session.lesson_focus == topic)
        rows = q.order_by(Task.created_at.desc()).limit(limit).all()
        out = []
        for t in rows:
            out.append({
                "task_id": int(t.id),
                "task_type": t.task_type,
                "difficulty": int(t.difficulty),
                "payload": t.payload_json,
                "created_at": t.created_at.isoformat() if t.created_at else None
            })
        return out
    finally:
        db.close()
class ApproveTaskRequest(BaseModel):
    task_id: int
    approved: bool = True

@app.post("/teacher/ai/approve")
def teacher_approve(req: ApproveTaskRequest):
    db = SessionLocal()
    try:
        t = db.query(Task).filter(Task.id == req.task_id, Task.generated_by == "ai").first()
        if not t:
            raise HTTPException(status_code=404, detail="AI task not found")
        t.approved = bool(req.approved)
        db.commit()
        return {"ok": True, "task_id": int(t.id), "approved": bool(t.approved)}
    finally:
        db.close()
@app.get("/caregiver/settings/{child_id}")
def get_caregiver_settings(child_id: int):
    db = SessionLocal()
    try:
        s = db.query(CaregiverSettings).filter(CaregiverSettings.child_id == child_id).first()
        if not s:
            return {"child_id": child_id, "session_minutes": 10, "sound_on": True}
        return {"child_id": child_id, "session_minutes": s.session_minutes, "sound_on": s.sound_on}
    finally:
        db.close()


class UpdateCaregiverSettingsRequest(BaseModel):
    child_id: int
    session_minutes: int = 10
    sound_on: bool = True

@app.post("/caregiver/settings")
def update_caregiver_settings(req: UpdateCaregiverSettingsRequest):
    db = SessionLocal()
    try:
        s = db.query(CaregiverSettings).filter(CaregiverSettings.child_id == req.child_id).first()
        if not s:
            s = CaregiverSettings(child_id=req.child_id, session_minutes=req.session_minutes, sound_on=req.sound_on)
            db.add(s)
        else:
            s.session_minutes = req.session_minutes
            s.sound_on = req.sound_on
        db.commit()
        return {"ok": True}
    finally:
        db.close()
