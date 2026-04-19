import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import regex
import re
import emoji

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from datetime import datetime, timezone

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Ваш сервис должен считывать эти переменные из окружения (env), так как проверяющая система управляет ими
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def normalize_text(text):
    if not text:
        return ""

    clean_text = re.sub(r"<[^>]+>", "", text)

    clean_text = re.sub(r"[\n\r\t]+", " ", clean_text)

    clean_text = emoji.replace_emoji(clean_text, replace=" ")

    clean_text = re.sub(r"\s{2,}", " ", clean_text)

    return clean_text.strip()


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError(
            "Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set"
        )

    missing_env_vars = [
        name
        for name in REQUIRED_ENV_VARS
        if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


# Модель данных, которую мы предоставляем и рассчитываем получать от вас
class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


# Метадата чанков в Qdrant'e, по которой вы можете фильтровать
class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str  # channel, group, private, thread
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.1.0", lifespan=lifespan)


# Внутри шаблона dense и rerank берутся из внешних HTTP endpoint'ов,
# которые предоставляет проверяющая система.
# Текущий код ниже — минимальный пример search pipeline.
DENSE_PREFETCH_K = 50  # 10
SPRASE_PREFETCH_K = 50  # 60
RETRIEVE_K = 100  # 70
RERANK_LIMIT = 30  # 50


async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    # Dense endpoint ожидает OpenAI-compatible body с input как списком строк.
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": [text],
        },
    )
    response.raise_for_status()

    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")

    return payload.data[0].embedding


async def embed_sparse(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text]))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")

    item = vectors[0]
    return SparseVector(
        indices=[int(index) for index in item.indices.tolist()],
        values=[float(value) for value in item.values.tolist()],
    )


def build_search_filter(date_range, participants):

    conditions_must = []
    conditions_should = []

    if date_range:

        def _parse_datetime(value: Any) -> datetime | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return None
                # Accept RFC3339/ISO-8601, including trailing "Z"
                try:
                    if s.endswith("Z"):
                        return datetime.fromisoformat(s[:-1]).replace(
                            tzinfo=timezone.utc
                        )
                    dt = datetime.fromisoformat(s)
                    return (
                        dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
                    )
                except ValueError:
                    return None
            return None

        def _coerce_number(value: Any) -> float | int | None:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return None
                try:
                    return int(s) if s.isdigit() else float(s)
                except ValueError:
                    return None
            return None

        raw_gte = getattr(date_range, "from_", None)
        raw_lte = getattr(date_range, "to", None)

        gte_dt = _parse_datetime(raw_gte)
        lte_dt = _parse_datetime(raw_lte)
        if gte_dt is not None or lte_dt is not None:
            conditions_must.append(
                models.FieldCondition(
                    key="metadata.start",
                    range=models.DatetimeRange(
                        gte=gte_dt,
                        lte=lte_dt,
                    ),
                )
            )
        else:
            gte_num = _coerce_number(raw_gte)
            lte_num = _coerce_number(raw_lte)
            if gte_num is not None or lte_num is not None:
                conditions_must.append(
                    models.FieldCondition(
                        key="metadata.start",
                        range=models.Range(
                            gte=gte_num,
                            lte=lte_num,
                        ),
                    )
                )

    if participants:
        for p in participants:
            conditions_should.append(
                models.FieldCondition(
                    key="metadata.participants", match=models.MatchValue(value=p)
                )
            )

    if not conditions_must and not conditions_should:
        return None

    if conditions_must and conditions_should:
        return models.Filter(
            must=conditions_must, should=conditions_should, min_should=1
        )
    elif conditions_should:
        return models.Filter(should=conditions_should, min_should=1)

    return models.Filter(must=conditions_must)


async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    search_filter: Any | None = None,
) -> Any | None:
    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPRASE_PREFETCH_K,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
        query_filter=search_filter,
    )

    if not response.points:
        return None

    return response.points


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    message_ids = metadata.get("message_ids") or []

    return [str(message_id) for message_id in message_ids]


async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    # Rerank endpoint возвращает score для пары query -> candidate text.
    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": targets,
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []

    return [float(sample["score"]) for sample in data]


def deduplicate_by_message(points: list[Any]) -> list[Any]:
    best_per_message = {}

    for point in points:
        msg_ids = extract_message_ids(point)
        if not msg_ids:
            continue

        # Берем первое message_id как основное (или агрегируем)
        primary_msg_id = msg_ids[0]

        if primary_msg_id not in best_per_message:
            best_per_message[primary_msg_id] = point
        # Иначе пропускаем дубликат

    return list(best_per_message.values())


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    rerank_candidates = points[:RERANK_LIMIT]
    rerank_targets = [point.payload.get("page_content") for point in rerank_candidates]
    scores = await get_rerank_scores(client, query, rerank_targets)

    reranked_candidates = [
        point
        for _, point in sorted(
            zip(scores, rerank_candidates, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    return reranked_candidates


# Ваш сервис должен имплементировать оба этих метода
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    query = payload.question.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="question.text is required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    dense_query = query + " " + " ".join(payload.question.hyde)
    sparse_query = (
        query
        + " "
        + " ".join(payload.question.keywords)
        + " "
        + " ".join(payload.question.entities.people)
        + " "
        + " ".join(payload.question.entities.emails)
        + " "
        + " ".join(payload.question.entities.documents)
        + " "  #######+
        ######"".join(payload.question.variants)
    )

    participants = None
    if payload.question.entities.people:
        participants = payload.question.entities.people

    search_filter = build_search_filter(
        payload.question.date_range, participants
    )
    dense_vector = await embed_dense(client, normalize_text(dense_query))
    sparse_vector = await embed_sparse(normalize_text(sparse_query))
    best_points = await qdrant_search(
        qdrant, dense_vector, sparse_vector, search_filter
    )

    if best_points is None:
        return SearchAPIResponse(results=[])

    best_points = await rerank_points(client, query, list(best_points))
    best_points = deduplicate_by_message(best_points)

    message_ids: list[str] = []
    for point in best_points:
        message_ids += extract_message_ids(point)

    return SearchAPIResponse(results=[SearchAPIItem(message_ids=message_ids)])


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
