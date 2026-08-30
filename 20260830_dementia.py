"""
20260830_dementia.py
====================
Futuristic AI-Enabled Dementia Care Full-Stack System
======================================================
Blueprint & Runnable Skeleton

Architecture
------------
  FastAPI (REST + WebSocket)  ←→  PostgreSQL + pgvector
        ↕                               ↕
  Celery + Redis (async tasks)    Minio / S3 (media storage)
        ↕                               ↕
  AI Layer (OpenAI / local LLM)   Elasticsearch (search)

Sections
--------
  1.  Models            – SQLAlchemy ORM
  2.  Schemas           – Pydantic request/response
  3.  Storage           – S3-compatible media upload
  4.  AI Services       – transcription, captioning, embeddings
  5.  Search            – Elasticsearch ingestion & query
  6.  Scale Engine      – standardised dementia scales (MMSE, CDR, GDS, NPI …)
  7.  Progress Tracker  – longitudinal self-tracking
  8.  Authentication    – JWT-based, role=patient|carer|clinician
  9.  API Routes        – /auth /media /scales /progress /search /ai /ws
  10. WebSocket         – real-time session recording
  11. Celery Tasks      – background AI processing
  12. App Factory       – startup / lifespan
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. DEPENDENCIES  (pip install -r requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────
# fastapi uvicorn[standard] sqlalchemy[asyncio] asyncpg alembic
# pydantic[email] python-jose[cryptography] passlib[bcrypt]
# boto3 python-multipart celery[redis] elasticsearch
# openai python-magic pillow ffmpeg-python pgvector
# python-docx httpx

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import os
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ── FastAPI / Starlette ───────────────────────────────────────────────────────
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# ── Pydantic ──────────────────────────────────────────────────────────────────
from pydantic import BaseModel, EmailStr, Field

# ── SQLAlchemy async ──────────────────────────────────────────────────────────
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, selectinload

# ── Auth ──────────────────────────────────────────────────────────────────────
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── AWS / Minio ───────────────────────────────────────────────────────────────
import boto3
from botocore.exceptions import ClientError

# ── Celery ────────────────────────────────────────────────────────────────────
from celery import Celery

# ── Elasticsearch ─────────────────────────────────────────────────────────────
from elasticsearch import AsyncElasticsearch

# ── OpenAI ────────────────────────────────────────────────────────────────────
import openai


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

class Settings(BaseModel):
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "******localhost/dementia_db"
    )
    # Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE-ME-IN-PRODUCTION")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8
    # Storage
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "dementia-media")
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY", "minioadmin")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY", "minioadmin")
    # Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Elasticsearch
    ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATABASE MODELS
# ═════════════════════════════════════════════════════════════════════════════

class Base(DeclarativeBase):
    pass


class UserRole(str, Enum):
    patient = "patient"
    carer = "carer"
    clinician = "clinician"
    admin = "admin"


class User(Base):
    """Represents a patient, carer, or clinician."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default=UserRole.patient)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

    media_items = relationship("MediaItem", back_populates="owner", lazy="select")
    scale_entries = relationship("ScaleEntry", back_populates="user", lazy="select")
    progress_logs = relationship("ProgressLog", back_populates="user", lazy="select")


class MediaType(str, Enum):
    image = "image"
    audio = "audio"
    video = "video"
    document = "document"


class MediaItem(Base):
    """Stores metadata for uploaded images, audio, video, documents."""
    __tablename__ = "media_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    media_type = Column(String, nullable=False)
    original_filename = Column(String)
    s3_key = Column(String, nullable=False, unique=True)
    mime_type = Column(String)
    size_bytes = Column(Integer)
    duration_seconds = Column(Float, nullable=True)   # audio/video
    checksum_sha256 = Column(String)
    tags = Column(ARRAY(String), default=[])
    transcript = Column(Text, nullable=True)          # ASR output
    caption = Column(Text, nullable=True)             # image/video caption
    embedding_vector = Column(JSON, nullable=True)    # pgvector placeholder
    session_id = Column(UUID(as_uuid=True), nullable=True)  # groups a recording session
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ai_processed = Column(Boolean, default=False)

    owner = relationship("User", back_populates="media_items")


class ScaleName(str, Enum):
    MMSE = "MMSE"          # Mini-Mental State Examination
    CDR = "CDR"            # Clinical Dementia Rating
    GDS = "GDS"            # Geriatric Depression Scale
    NPI = "NPI"            # Neuropsychiatric Inventory
    ADAS_COG = "ADAS-Cog"  # Alzheimer's Disease Assessment Scale – Cognitive
    MoCA = "MoCA"          # Montreal Cognitive Assessment
    RUDAS = "RUDAS"        # Rowland Universal Dementia Assessment Scale
    FAST = "FAST"          # Functional Assessment Staging Test
    ADL = "ADL"            # Activities of Daily Living
    IADL = "IADL"          # Instrumental ADL
    ZBI = "ZBI"            # Zarit Burden Interview (carer)
    QOL_AD = "QOL-AD"      # Quality of Life – Alzheimer's Disease


class ScaleEntry(Base):
    """A single administration of a standardised scale."""
    __tablename__ = "scale_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    administered_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    scale_name = Column(String, nullable=False)
    raw_responses = Column(JSON, nullable=False)   # {item_key: value, …}
    total_score = Column(Float, nullable=True)
    subscores = Column(JSON, nullable=True)        # domain breakdowns
    severity_label = Column(String, nullable=True) # e.g. "Mild", "Moderate"
    notes = Column(Text, nullable=True)
    administered_at = Column(DateTime(timezone=True), server_default=func.now())
    sequence_number = Column(Integer, nullable=True)  # sequential ordering

    user = relationship("User", foreign_keys=[user_id], back_populates="scale_entries")


class ProgressLog(Base):
    """Daily/weekly self-tracking entry (mood, sleep, activity, pain, etc.)."""
    __tablename__ = "progress_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    log_date = Column(DateTime(timezone=True), nullable=False)
    mood_score = Column(Integer, nullable=True)          # 1–10
    sleep_hours = Column(Float, nullable=True)
    activity_minutes = Column(Integer, nullable=True)
    pain_score = Column(Integer, nullable=True)          # 0–10
    appetite_score = Column(Integer, nullable=True)      # 1–5
    orientation_score = Column(Integer, nullable=True)   # 0–4 (time/place/person/event)
    custom_fields = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="progress_logs")


# ═════════════════════════════════════════════════════════════════════════════
# 3. PYDANTIC SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.patient


class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str]
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class MediaItemOut(BaseModel):
    id: str
    media_type: str
    original_filename: Optional[str]
    s3_key: str
    mime_type: Optional[str]
    size_bytes: Optional[int]
    transcript: Optional[str]
    caption: Optional[str]
    tags: Optional[List[str]]
    created_at: datetime
    ai_processed: bool

    class Config:
        from_attributes = True


class ScaleEntryCreate(BaseModel):
    scale_name: ScaleName
    raw_responses: Dict[str, Any]
    notes: Optional[str] = None


class ScaleEntryOut(BaseModel):
    id: str
    scale_name: str
    total_score: Optional[float]
    severity_label: Optional[str]
    subscores: Optional[Dict[str, Any]]
    administered_at: datetime
    sequence_number: Optional[int]

    class Config:
        from_attributes = True


class ProgressLogCreate(BaseModel):
    log_date: datetime
    mood_score: Optional[int] = Field(None, ge=1, le=10)
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    activity_minutes: Optional[int] = Field(None, ge=0)
    pain_score: Optional[int] = Field(None, ge=0, le=10)
    appetite_score: Optional[int] = Field(None, ge=1, le=5)
    orientation_score: Optional[int] = Field(None, ge=0, le=4)
    custom_fields: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ProgressLogOut(ProgressLogCreate):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    query: str
    media_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    tags: Optional[List[str]] = None
    page: int = 1
    page_size: int = 20


class SearchResult(BaseModel):
    id: str
    media_type: str
    original_filename: Optional[str]
    highlight: Optional[str]
    score: float
    created_at: datetime


# ═════════════════════════════════════════════════════════════════════════════
# 4. AUTHENTICATION
# ═════════════════════════════════════════════════════════════════════════════

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(lambda: None),  # replaced by lifespan dependency
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    # NOTE: fetch user from DB here (omitted for blueprint clarity)
    return user_id  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# 5. STORAGE SERVICE  (S3 / Minio)
# ═════════════════════════════════════════════════════════════════════════════

class StorageService:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.bucket = settings.S3_BUCKET

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def upload_fileobj(
        self,
        fileobj: io.BytesIO,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        self.client.upload_fileobj(
            fileobj,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return key

    def generate_presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def delete_object(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


storage = StorageService()


# ═════════════════════════════════════════════════════════════════════════════
# 6. AI SERVICES
# ═════════════════════════════════════════════════════════════════════════════

class AIService:
    """
    Wraps OpenAI APIs for:
      - Audio transcription  (Whisper)
      - Image captioning     (Vision / GPT-4o)
      - Text embeddings      (text-embedding-3-large)
      - Clinical summarisation
    """

    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY

    # ── Transcription ─────────────────────────────────────────────────────────
    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Whisper ASR – converts voice/video audio to text."""
        file_tuple = (filename, audio_bytes, "audio/wav")
        result = await asyncio.to_thread(
            openai.audio.transcriptions.create,
            model="whisper-1",
            file=file_tuple,
            language="en",
        )
        return result.text

    # ── Image Captioning ──────────────────────────────────────────────────────
    async def caption_image(self, image_url: str) -> str:
        """GPT-4o vision – generates a clinical description of an image."""
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a clinical assistant. Describe this image in the context "
                                "of dementia care. Be concise and factual."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=256,
        )
        return response.choices[0].message.content

    # ── Text Embedding ────────────────────────────────────────────────────────
    async def embed_text(self, text: str) -> List[float]:
        """Generates a 3072-dim embedding for semantic search & AI training."""
        response = await asyncio.to_thread(
            openai.embeddings.create,
            model="text-embedding-3-large",
            input=text[:8191],
        )
        return response.data[0].embedding

    # ── Clinical Summary ──────────────────────────────────────────────────────
    async def summarise_patient_timeline(
        self, scale_entries: List[dict], progress_logs: List[dict]
    ) -> str:
        """GPT-4o generates a narrative clinical summary from longitudinal data."""
        payload = json.dumps(
            {"scales": scale_entries[-10:], "progress": progress_logs[-30:]},
            default=str,
        )
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a specialist dementia care AI. Summarise the patient's "
                        "cognitive trajectory, highlight concerning trends, and suggest "
                        "care plan adjustments. Output structured markdown."
                    ),
                },
                {"role": "user", "content": payload},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content


ai_service = AIService()


# ═════════════════════════════════════════════════════════════════════════════
# 7. ELASTICSEARCH SERVICE
# ═════════════════════════════════════════════════════════════════════════════

MEDIA_INDEX = "dementia_media"

ES_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "owner_id": {"type": "keyword"},
            "media_type": {"type": "keyword"},
            "original_filename": {"type": "text"},
            "transcript": {"type": "text", "analyzer": "english"},
            "caption": {"type": "text", "analyzer": "english"},
            "tags": {"type": "keyword"},
            "created_at": {"type": "date"},
            "embedding_vector": {
                "type": "dense_vector",
                "dims": 3072,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


class SearchService:
    def __init__(self):
        self.es = AsyncElasticsearch([settings.ES_URL])

    async def ensure_index(self) -> None:
        exists = await self.es.indices.exists(index=MEDIA_INDEX)
        if not exists:
            await self.es.indices.create(index=MEDIA_INDEX, body=ES_INDEX_MAPPING)

    async def index_media(self, item: MediaItem, embedding: Optional[List[float]] = None) -> None:
        doc = {
            "id": str(item.id),
            "owner_id": str(item.owner_id),
            "media_type": item.media_type,
            "original_filename": item.original_filename,
            "transcript": item.transcript,
            "caption": item.caption,
            "tags": item.tags or [],
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        if embedding:
            doc["embedding_vector"] = embedding
        await self.es.index(index=MEDIA_INDEX, id=str(item.id), document=doc)

    async def full_text_search(
        self, req: SearchRequest, owner_id: str
    ) -> List[SearchResult]:
        must_clauses: list = [
            {"term": {"owner_id": owner_id}},
            {
                "multi_match": {
                    "query": req.query,
                    "fields": ["transcript^2", "caption^1.5", "original_filename", "tags"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
        ]
        if req.media_types:
            must_clauses.append({"terms": {"media_type": req.media_types}})
        if req.date_from or req.date_to:
            rng: dict = {}
            if req.date_from:
                rng["gte"] = req.date_from.isoformat()
            if req.date_to:
                rng["lte"] = req.date_to.isoformat()
            must_clauses.append({"range": {"created_at": rng}})

        result = await self.es.search(
            index=MEDIA_INDEX,
            body={
                "query": {"bool": {"must": must_clauses}},
                "highlight": {
                    "fields": {"transcript": {}, "caption": {}},
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                },
                "from": (req.page - 1) * req.page_size,
                "size": req.page_size,
            },
        )
        hits = result["hits"]["hits"]
        return [
            SearchResult(
                id=h["_source"]["id"],
                media_type=h["_source"]["media_type"],
                original_filename=h["_source"].get("original_filename"),
                highlight=" … ".join(
                    h.get("highlight", {}).get("transcript", [])
                    + h.get("highlight", {}).get("caption", [])
                )
                or None,
                score=h["_score"],
                created_at=h["_source"]["created_at"],
            )
            for h in hits
        ]

    async def semantic_search(
        self, query_embedding: List[float], owner_id: str, top_k: int = 10
    ) -> List[dict]:
        result = await self.es.search(
            index=MEDIA_INDEX,
            body={
                "knn": {
                    "field": "embedding_vector",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 5,
                    "filter": {"term": {"owner_id": owner_id}},
                }
            },
        )
        return result["hits"]["hits"]


search_service = SearchService()


# ═════════════════════════════════════════════════════════════════════════════
# 8. SCALE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ScaleEngine:
    """
    Computes scores for standardised dementia scales from raw item responses.
    Each method returns (total_score, subscores_dict, severity_label).
    """

    # ── MMSE (0–30) ───────────────────────────────────────────────────────────
    @staticmethod
    def score_mmse(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        """
        Domains:
          orientation_time (5), orientation_place (5),
          registration (3), attention_calc (5), recall (3),
          language (8), visuospatial (1)
        """
        domains = {
            "orientation_time": 5,
            "orientation_place": 5,
            "registration": 3,
            "attention_calc": 5,
            "recall": 3,
            "language": 8,
            "visuospatial": 1,
        }
        subscores = {k: min(int(responses.get(k, 0)), v) for k, v in domains.items()}
        total = sum(subscores.values())
        if total >= 24:
            label = "Normal"
        elif total >= 19:
            label = "Mild"
        elif total >= 10:
            label = "Moderate"
        else:
            label = "Severe"
        return float(total), subscores, label

    # ── CDR Sum of Boxes (0–18) ───────────────────────────────────────────────
    @staticmethod
    def score_cdr(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        """
        Six boxes: memory, orientation, judgment, community, home, care.
        Each 0/0.5/1/2/3.
        """
        boxes = ["memory", "orientation", "judgment", "community", "home", "care"]
        subscores = {b: float(responses.get(b, 0)) for b in boxes}
        total = sum(subscores.values())
        if total == 0:
            label = "No impairment"
        elif total <= 4.0:
            label = "Mild"
        elif total <= 9.0:
            label = "Moderate"
        else:
            label = "Severe"
        return total, subscores, label

    # ── GDS (0–30) ────────────────────────────────────────────────────────────
    @staticmethod
    def score_gds(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        """30 yes/no items; 1 point each for depressive response."""
        total = sum(int(bool(v)) for v in responses.values())
        total = min(total, 30)
        if total <= 9:
            label = "Normal"
        elif total <= 19:
            label = "Mild Depression"
        else:
            label = "Severe Depression"
        return float(total), {}, label

    # ── MoCA (0–30) ───────────────────────────────────────────────────────────
    @staticmethod
    def score_moca(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        domains = {
            "visuospatial": 5,
            "naming": 3,
            "attention": 6,
            "language": 3,
            "abstraction": 2,
            "delayed_recall": 5,
            "orientation": 6,
        }
        subscores = {k: min(int(responses.get(k, 0)), v) for k, v in domains.items()}
        total = sum(subscores.values())
        # +1 if ≤12 years education
        if responses.get("education_adjustment"):
            total = min(total + 1, 30)
        label = "Normal" if total >= 26 else ("Mild" if total >= 18 else "Moderate/Severe")
        return float(total), subscores, label

    # ── FAST (stage 1–7) ──────────────────────────────────────────────────────
    @staticmethod
    def score_fast(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        stage = int(responses.get("stage", 1))
        labels = {
            1: "Normal adult", 2: "Normal older adult", 3: "Early Alzheimer's",
            4: "Mild Alzheimer's", 5: "Moderate Alzheimer's",
            6: "Moderately severe Alzheimer's", 7: "Severe Alzheimer's",
        }
        return float(stage), {}, labels.get(stage, "Unknown")

    # ── ADL (0–6 Katz) ────────────────────────────────────────────────────────
    @staticmethod
    def score_adl(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        items = ["bathing", "dressing", "toileting", "transferring", "continence", "feeding"]
        subscores = {k: int(bool(responses.get(k, 0))) for k in items}
        total = sum(subscores.values())
        label = "Independent" if total == 6 else ("Moderate dependence" if total >= 3 else "Severe dependence")
        return float(total), subscores, label

    # ── ZBI (0–88) ────────────────────────────────────────────────────────────
    @staticmethod
    def score_zbi(responses: Dict[str, Any]) -> tuple[float, dict, str]:
        """22 items, each 0–4."""
        total = sum(min(int(v), 4) for v in responses.values())
        total = min(total, 88)
        if total <= 20:
            label = "Little or no burden"
        elif total <= 40:
            label = "Mild to moderate burden"
        elif total <= 60:
            label = "Moderate to severe burden"
        else:
            label = "Severe burden"
        return float(total), {}, label

    SCALE_MAP = {
        ScaleName.MMSE: score_mmse.__func__,      # type: ignore
        ScaleName.CDR: score_cdr.__func__,        # type: ignore
        ScaleName.GDS: score_gds.__func__,        # type: ignore
        ScaleName.MoCA: score_moca.__func__,      # type: ignore
        ScaleName.FAST: score_fast.__func__,      # type: ignore
        ScaleName.ADL: score_adl.__func__,        # type: ignore
        ScaleName.ZBI: score_zbi.__func__,        # type: ignore
    }

    def compute(self, scale_name: ScaleName, responses: Dict[str, Any]):
        fn = self.SCALE_MAP.get(scale_name)
        if fn is None:
            return None, None, None
        return fn(responses)


scale_engine = ScaleEngine()


# ═════════════════════════════════════════════════════════════════════════════
# 9. CELERY BACKGROUND TASKS
# ═════════════════════════════════════════════════════════════════════════════

celery_app = Celery("dementia", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_expires = 3600


@celery_app.task(name="process_media_ai")
def process_media_ai_task(media_id: str, s3_key: str, media_type: str) -> dict:
    """
    Background task:
      1. Download media from S3
      2. Transcribe (audio/video) or caption (image)
      3. Generate embedding
      4. Index in Elasticsearch
      5. Update DB record
    Returns status dict.
    """
    import asyncio as _asyncio

    async def _run():
        # Download from S3
        s3_obj = storage.client.get_object(Bucket=storage.bucket, Key=s3_key)
        raw_bytes = s3_obj["Body"].read()

        transcript = None
        caption = None
        embedding_text = ""

        if media_type in ("audio", "video"):
            transcript = await ai_service.transcribe_audio(raw_bytes, Path(s3_key).name)
            embedding_text = transcript or ""

        if media_type == "image":
            # Generate presigned URL for vision API
            img_url = storage.generate_presigned_url(s3_key)
            caption = await ai_service.caption_image(img_url)
            embedding_text = caption or ""

        embedding = None
        if embedding_text:
            embedding = await ai_service.embed_text(embedding_text)

        return {
            "media_id": media_id,
            "transcript": transcript,
            "caption": caption,
            "embedding": embedding,
        }

    return _asyncio.run(_run())


@celery_app.task(name="generate_patient_summary")
def generate_patient_summary_task(user_id: str) -> str:
    """Asynchronously generate and cache a narrative AI summary for a patient."""
    import asyncio as _asyncio

    # In production: fetch from DB, call ai_service.summarise_patient_timeline
    return _asyncio.run(
        ai_service.summarise_patient_timeline(scale_entries=[], progress_logs=[])
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10. WEBSOCKET – REAL-TIME RECORDING SESSION
# ═════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages active WebSocket connections keyed by session_id."""

    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        self.active.setdefault(session_id, []).append(ws)

    def disconnect(self, ws: WebSocket, session_id: str) -> None:
        conns = self.active.get(session_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, session_id: str, message: dict) -> None:
        for ws in self.active.get(session_id, []):
            await ws.send_json(message)


ws_manager = ConnectionManager()


# ═════════════════════════════════════════════════════════════════════════════
# 11. FASTAPI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

engine = create_async_engine(settings.DATABASE_URL, echo=False)


async def get_db():
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as session:
        yield session


app = FastAPI(
    title="DementiaAI Care System",
    description=(
        "Full-stack AI platform for dementia patient/carer management. "
        "Supports media upload, voice/video recording, clinical scale tracking, "
        "self-monitoring, and AI-powered search & summarisation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    storage.ensure_bucket()
    await search_service.ensure_index()


# ─── AUTH ROUTER ──────────────────────────────────────────────────────────────

from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=UserOut, status_code=201)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select as sa_select

    existing = (await db.execute(sa_select(User).where(User.email == user_in.email))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        full_name=user_in.full_name,
        role=user_in.role.value,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@auth_router.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select as sa_select

    user = (await db.execute(sa_select(User).where(User.email == form.username))).scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return Token(access_token=token, token_type="bearer")


# ─── MEDIA ROUTER ─────────────────────────────────────────────────────────────

media_router = APIRouter(prefix="/media", tags=["Media"])


@media_router.post("/upload", response_model=MediaItemOut, status_code=201)
async def upload_media(
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    session_id: Optional[str] = Query(None),
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    bg: BackgroundTasks = None,
):
    """
    Upload image, audio, video, or document.
    Automatically triggers AI processing (transcription/captioning/embedding).
    """
    raw = await file.read()
    checksum = hashlib.sha256(raw).hexdigest()
    mime = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"

    # Determine media type
    if mime.startswith("image/"):
        mtype = MediaType.image.value
    elif mime.startswith("audio/"):
        mtype = MediaType.audio.value
    elif mime.startswith("video/"):
        mtype = MediaType.video.value
    else:
        mtype = MediaType.document.value

    s3_key = f"{current_user_id}/{mtype}/{uuid.uuid4()}_{file.filename}"
    storage.upload_fileobj(io.BytesIO(raw), s3_key, mime)

    item = MediaItem(
        owner_id=uuid.UUID(current_user_id),
        media_type=mtype,
        original_filename=file.filename,
        s3_key=s3_key,
        mime_type=mime,
        size_bytes=len(raw),
        checksum_sha256=checksum,
        tags=tags.split(",") if tags else [],
        session_id=uuid.UUID(session_id) if session_id else None,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    # Kick off async AI processing
    process_media_ai_task.delay(str(item.id), s3_key, mtype)

    return item


@media_router.get("/{media_id}/download")
async def download_media(
    media_id: str,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select as sa_select

    item = (await db.execute(sa_select(MediaItem).where(MediaItem.id == uuid.UUID(media_id)))).scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
    url = storage.generate_presigned_url(item.s3_key)
    return {"download_url": url, "expires_in": 3600}


@media_router.get("/", response_model=List[MediaItemOut])
async def list_media(
    media_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select as sa_select

    q = sa_select(MediaItem).where(MediaItem.owner_id == uuid.UUID(current_user_id))
    if media_type:
        q = q.where(MediaItem.media_type == media_type)
    q = q.order_by(MediaItem.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    items = (await db.execute(q)).scalars().all()
    return items


# ─── SCALES ROUTER ────────────────────────────────────────────────────────────

scales_router = APIRouter(prefix="/scales", tags=["Clinical Scales"])


@scales_router.post("/", response_model=ScaleEntryOut, status_code=201)
async def submit_scale(
    entry_in: ScaleEntryCreate,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select as sa_select, func as sa_func

    total, subscores, label = scale_engine.compute(entry_in.scale_name, entry_in.raw_responses)

    # Auto-increment sequence number per user/scale
    max_seq = (
        await db.execute(
            sa_select(sa_func.max(ScaleEntry.sequence_number)).where(
                ScaleEntry.user_id == uuid.UUID(current_user_id),
                ScaleEntry.scale_name == entry_in.scale_name.value,
            )
        )
    ).scalar() or 0

    entry = ScaleEntry(
        user_id=uuid.UUID(current_user_id),
        scale_name=entry_in.scale_name.value,
        raw_responses=entry_in.raw_responses,
        total_score=total,
        subscores=subscores,
        severity_label=label,
        notes=entry_in.notes,
        sequence_number=max_seq + 1,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


@scales_router.get("/", response_model=List[ScaleEntryOut])
async def list_scale_entries(
    scale_name: Optional[str] = None,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select as sa_select

    q = sa_select(ScaleEntry).where(ScaleEntry.user_id == uuid.UUID(current_user_id))
    if scale_name:
        q = q.where(ScaleEntry.scale_name == scale_name)
    q = q.order_by(ScaleEntry.administered_at)
    entries = (await db.execute(q)).scalars().all()
    return entries


@scales_router.get("/trajectory/{scale_name}")
async def scale_trajectory(
    scale_name: str,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns time-series score data suitable for charting cognitive trajectory."""
    from sqlalchemy import select as sa_select

    entries = (
        await db.execute(
            sa_select(ScaleEntry)
            .where(
                ScaleEntry.user_id == uuid.UUID(current_user_id),
                ScaleEntry.scale_name == scale_name,
            )
            .order_by(ScaleEntry.administered_at)
        )
    ).scalars().all()

    return [
        {
            "date": e.administered_at.isoformat(),
            "score": e.total_score,
            "severity": e.severity_label,
            "sequence": e.sequence_number,
        }
        for e in entries
    ]


# ─── PROGRESS ROUTER ──────────────────────────────────────────────────────────

progress_router = APIRouter(prefix="/progress", tags=["Self-Tracking Progress"])


@progress_router.post("/", response_model=ProgressLogOut, status_code=201)
async def log_progress(
    log_in: ProgressLogCreate,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    log = ProgressLog(user_id=uuid.UUID(current_user_id), **log_in.model_dump())
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


@progress_router.get("/", response_model=List[ProgressLogOut])
async def list_progress(
    days: int = 30,
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select as sa_select

    since = datetime.now(timezone.utc) - timedelta(days=days)
    logs = (
        await db.execute(
            sa_select(ProgressLog)
            .where(
                ProgressLog.user_id == uuid.UUID(current_user_id),
                ProgressLog.log_date >= since,
            )
            .order_by(ProgressLog.log_date)
        )
    ).scalars().all()
    return logs


@progress_router.get("/summary")
async def progress_summary(
    current_user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns AI-generated narrative summary of the patient's recent progress."""
    task = generate_patient_summary_task.delay(current_user_id)
    return {"task_id": task.id, "status": "processing"}


# ─── SEARCH ROUTER ────────────────────────────────────────────────────────────

search_router = APIRouter(prefix="/search", tags=["AI Search"])


@search_router.post("/", response_model=List[SearchResult])
async def search(
    req: SearchRequest,
    current_user_id: str = Depends(get_current_user),
):
    """Full-text + semantic search across all uploaded media (transcripts, captions, tags)."""
    return await search_service.full_text_search(req, current_user_id)


@search_router.post("/semantic")
async def semantic_search(
    query: str,
    top_k: int = 10,
    current_user_id: str = Depends(get_current_user),
):
    """Vector-based semantic search using text embeddings."""
    embedding = await ai_service.embed_text(query)
    results = await search_service.semantic_search(embedding, current_user_id, top_k)
    return results


# ─── AI ROUTER ────────────────────────────────────────────────────────────────

ai_router = APIRouter(prefix="/ai", tags=["AI Services"])


@ai_router.post("/transcribe/{media_id}")
async def retranscribe(
    media_id: str,
    current_user_id: str = Depends(get_current_user),
):
    """Re-trigger AI transcription/captioning for a media item."""
    process_media_ai_task.delay(media_id, "", "audio")
    return {"status": "queued", "media_id": media_id}


@ai_router.get("/summary")
async def ai_summary(current_user_id: str = Depends(get_current_user)):
    """Queue generation of an AI clinical narrative summary."""
    task = generate_patient_summary_task.delay(current_user_id)
    return {"task_id": task.id, "status": "processing"}


@ai_router.get("/task/{task_id}")
async def get_task_result(task_id: str, current_user_id: str = Depends(get_current_user)):
    """Poll result of a background Celery task."""
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


# ─── WEBSOCKET ENDPOINT ────────────────────────────────────────────────────────

@app.websocket("/ws/session/{session_id}")
async def websocket_session(
    ws: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    """
    Real-time WebSocket endpoint for sequential recording sessions.
    Clients stream audio/video chunks; server responds with live transcriptions.

    Protocol:
      Client → {"type": "chunk", "data": "<base64-audio>", "seq": 1}
      Server → {"type": "transcript", "text": "...", "seq": 1, "final": false}
      Client → {"type": "end"}
      Server → {"type": "session_complete", "session_id": "...", "media_count": N}
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        await ws.close(code=4001)
        return

    await ws_manager.connect(ws, session_id)
    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "chunk":
                import base64

                audio_bytes = base64.b64decode(msg.get("data", ""))
                seq = msg.get("seq", 0)
                try:
                    text = await ai_service.transcribe_audio(audio_bytes)
                    await ws.send_json(
                        {"type": "transcript", "text": text, "seq": seq, "final": False}
                    )
                except Exception as exc:
                    await ws.send_json({"type": "error", "detail": str(exc), "seq": seq})

            elif msg_type == "end":
                await ws.send_json(
                    {"type": "session_complete", "session_id": session_id}
                )
                break

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws, session_id)


# ─── REGISTER ROUTERS ─────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(media_router)
app.include_router(scales_router)
app.include_router(progress_router)
app.include_router(search_router)
app.include_router(ai_router)


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": app.version, "timestamp": datetime.utcnow().isoformat()}


# ═════════════════════════════════════════════════════════════════════════════
# 12. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "20260830_dementia:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

# ─── END OF FILE ──────────────────────────────────────────────────────────────
# Running:
#   uvicorn 20260830_dementia:app --reload
#
# Worker (Celery):
#   celery -A 20260830_dementia.celery_app worker --loglevel=info
#
# API docs:
#   http://localhost:8000/docs
