import os
import json
import math
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, model_validator
from psycopg import connect
from psycopg.rows import dict_row


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
PERCEPTION_SCALE_UNIT = os.getenv("PERCEPTION_SCALE_UNIT", "m").strip().lower()
FIREBASE_STORAGE_BASE_URI = os.getenv(
    "FIREBASE_STORAGE_BASE_URI",
    "gs://interiorplatform-d58e0.firebasestorage.app",
).rstrip("/")

app = FastAPI(title="Interior Matching Engine", version="0.1.0")


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or matching request.")
    context: str | None = Field(
        default=None,
        description="Optional backend-provided context used when no vector store is configured.",
    )
    obj_id: str | None = Field(default=None, description="Optional object id from perception/reconstruction.")
    room_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured room/object metadata from the backend.",
    )
    model: str | None = Field(default=None, description="Optional OpenAI model override.")
    fit_candidates: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional dimension-filtered product candidates to ground the answer.",
    )


class RagQueryResponse(BaseModel):
    status: str
    answer: str
    model: str
    retrieval: dict[str, Any]


class Dimensions(BaseModel):
    x: float = Field(..., gt=0, description="Width on the x axis, in cm.")
    y: float = Field(..., gt=0, description="Depth on the y axis, in cm.")
    z: float = Field(..., gt=0, description="Height on the z axis, in cm.")


class ProductCandidate(BaseModel):
    id: str | int
    name: str | None = None
    product_name: str | None = None
    dimensions: Dimensions | None = None
    width_x: float | None = None
    depth_y: float | None = None
    height_z: float | None = None
    price: int | None = None
    category: str | None = None
    category_code: str | None = None
    product_url: str | None = None
    storage_path: str | None = Field(
        default=None,
        description="Firebase/GCS object path or full gs:// URI for the product asset.",
    )
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def normalize_schema_fields(self) -> "ProductCandidate":
        if self.name is None:
            self.name = self.product_name

        if self.dimensions is None and None not in (self.width_x, self.depth_y, self.height_z):
            self.dimensions = Dimensions(
                x=float(self.width_x),
                y=float(self.depth_y),
                z=float(self.height_z),
            )

        if self.category is None:
            self.category = self.category_code

        if self.name is None:
            raise ValueError("Either name or product_name is required.")
        if self.dimensions is None:
            raise ValueError("Either dimensions or width_x/depth_y/height_z is required.")
        return self


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Position3D(BaseModel):
    x: float
    y: float
    z: float


class PerceptionObject(BaseModel):
    id: int
    label: str
    confidence: float | None = None
    bbox: BoundingBox | None = None
    position_3d: Position3D | None = None


class ParsedPerceptionObject(BaseModel):
    id: int
    category: str
    asset_filename: str | None = None
    estimated_width: float | None = None
    estimated_height: float | None = None
    confidence: float | None = None
    bbox: BoundingBox | None = None
    position_3d: Position3D | None = None


class FitQueryRequest(BaseModel):
    space: Dimensions
    products: list[ProductCandidate] = Field(..., min_length=1)
    clearance_cm: float = Field(
        default=0,
        ge=0,
        description="Required extra empty space subtracted from each available axis.",
    )
    allow_xy_rotation: bool = Field(
        default=True,
        description="Whether product width/depth can be swapped on the floor plane.",
    )


class FitProduct(BaseModel):
    id: str | int
    name: str
    dimensions: Dimensions
    fits: bool
    orientation: str | None
    remaining_cm: Dimensions | None
    storage_uri: str | None = None
    price: int | None = None
    category: str | None = None
    product_url: str | None = None
    relevance_score: float | None = None
    metadata: dict[str, Any] | None = None


class FitQueryResponse(BaseModel):
    status: str
    space: Dimensions
    clearance_cm: float
    firebase_storage_base_uri: str
    fit_count: int
    products: list[FitProduct]


class FitRagQueryRequest(FitQueryRequest):
    query: str = Field(..., min_length=1, description="Question to answer using only fitting products.")
    context: str | None = None
    obj_id: str | None = None
    room_context: dict[str, Any] | None = None
    model: str | None = None


class PerceptionRagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    perception_objects: list[PerceptionObject] = Field(..., min_length=1)
    products: list[ProductCandidate] | None = Field(
        default=None,
        description="Optional product rows. If omitted, matching-engine loads products from DB.",
    )
    target_object_id: int | None = None
    space: Dimensions | None = Field(
        default=None,
        description="Optional explicit available space. If omitted, the selected perception object's scale is used.",
    )
    clearance_cm: float = Field(default=0, ge=0)
    allow_xy_rotation: bool = True
    limit: int = Field(default=100, ge=1, le=200)
    room_context: dict[str, Any] | None = None
    model: str | None = None


class FitRagQueryResponse(BaseModel):
    status: str
    answer: str
    model: str
    retrieval: dict[str, Any]
    fit: FitQueryResponse
    perception: ParsedPerceptionObject | None = None


def _client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured for matching-engine.",
        )
    return OpenAI()


def _build_input(request: RagQueryRequest) -> str:
    parts = [
        "You are an interior product matching assistant.",
        "Answer in Korean unless the user asks for another language.",
        "Use retrieved or provided context as the source of truth.",
        "",
        f"Query: {request.query}",
    ]

    if request.obj_id:
        parts.append(f"Object ID: {request.obj_id}")
    if request.room_context:
        room_context = json.dumps(request.room_context, ensure_ascii=False)
        parts.append(f"Room context: {room_context}")
    if request.context:
        parts.append(f"Provided context: {request.context}")
    if request.fit_candidates:
        candidates = json.dumps(request.fit_candidates, ensure_ascii=False)
        parts.append(f"Dimension-filtered fit candidates: {candidates}")

    return "\n".join(parts)


def _storage_uri(storage_path: str | None) -> str | None:
    if not storage_path:
        return None
    if storage_path.startswith("gs://"):
        return storage_path
    return f"{FIREBASE_STORAGE_BASE_URI}/{storage_path.lstrip('/')}"


def _parse_perception_label(label: str) -> tuple[str, str | None, float | None, float | None]:
    parts = label.split("|")
    category = parts[0].strip() if parts else label.strip()
    asset_filename = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    estimated_width = _safe_float(parts[2]) if len(parts) > 2 else None
    estimated_height = _safe_float(parts[3]) if len(parts) > 3 else None
    return category, asset_filename, estimated_width, estimated_height


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_perception_object(obj: PerceptionObject) -> ParsedPerceptionObject:
    category, asset_filename, estimated_width, estimated_height = _parse_perception_label(obj.label)
    return ParsedPerceptionObject(
        id=obj.id,
        category=category,
        asset_filename=asset_filename,
        estimated_width=estimated_width,
        estimated_height=estimated_height,
        confidence=obj.confidence,
        bbox=obj.bbox,
        position_3d=obj.position_3d,
    )


def _select_perception_object(
    objects: list[PerceptionObject],
    target_object_id: int | None,
) -> ParsedPerceptionObject:
    if target_object_id is not None:
        for obj in objects:
            if obj.id == target_object_id:
                return _parse_perception_object(obj)
        raise HTTPException(status_code=404, detail=f"Perception object {target_object_id} was not found.")

    sorted_objects = sorted(objects, key=lambda obj: obj.confidence or 0, reverse=True)
    return _parse_perception_object(sorted_objects[0])


def _space_from_perception(obj: ParsedPerceptionObject) -> Dimensions:
    if obj.estimated_width is None or obj.estimated_height is None:
        raise HTTPException(
            status_code=422,
            detail="Perception label must include scale_w and scale_h, or request.space must be provided.",
        )

    scale_multiplier = 100.0 if PERCEPTION_SCALE_UNIT in {"m", "meter", "meters"} else 1.0
    width_cm = obj.estimated_width * scale_multiplier
    height_cm = obj.estimated_height * scale_multiplier

    # ai-perception estimates width/height only. Use width for x/y footprint and height for z.
    return Dimensions(
        x=round(width_cm, 3),
        y=round(width_cm, 3),
        z=round(height_cm, 3),
    )


def _category_matches(perception_category: str, product: ProductCandidate) -> bool:
    product_category = product.category or product.category_code
    if not product_category:
        return True
    return perception_category.lower() in product_category.lower() or product_category.lower() in perception_category.lower()


def _load_products_from_db(category: str | None, limit: int) -> list[ProductCandidate]:
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    db_name = os.getenv("DB_NAME")
    port = int(os.getenv("DB_PORT", "5432"))

    if not all([host, user, password, db_name]):
        raise HTTPException(
            status_code=500,
            detail="Product candidates were not provided and DB_HOST/DB_USER/DB_PASS/DB_NAME are not configured.",
        )

    query = """
        SELECT
            p.id,
            p.product_name,
            p.price,
            p.width_x,
            p.depth_y,
            p.height_z,
            p.product_url,
            c.category_code,
            c.category_name,
            img.image_filename AS storage_path,
            img.image_description
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN LATERAL (
            SELECT image_filename, image_description
            FROM product_images
            WHERE product_id = p.id
            ORDER BY is_main DESC, id ASC
            LIMIT 1
        ) img ON true
        WHERE (
            %(category)s IS NULL
            OR LOWER(c.category_code) LIKE LOWER(%(category_like)s)
            OR LOWER(c.category_name) LIKE LOWER(%(category_like)s)
        )
        ORDER BY p.id ASC
        LIMIT %(limit)s
    """

    params = {
        "category": category,
        "category_like": f"%{category}%" if category else None,
        "limit": limit,
    }

    try:
        with connect(host=host, port=port, dbname=db_name, user=user, password=password, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

                if not rows and category:
                    cur.execute(query, {"category": None, "category_like": None, "limit": limit})
                    rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Product DB query failed: {exc}") from exc

    return [
        ProductCandidate(
            id=row["id"],
            product_name=row["product_name"],
            price=row["price"],
            width_x=row["width_x"],
            depth_y=row["depth_y"],
            height_z=row["height_z"],
            product_url=row["product_url"],
            category=row["category_name"],
            category_code=row["category_code"],
            storage_path=row["storage_path"],
            metadata={"image_description": row["image_description"]},
        )
        for row in rows
    ]


def _remaining(space: Dimensions, product: Dimensions) -> Dimensions:
    return Dimensions(
        x=round(space.x - product.x, 3),
        y=round(space.y - product.y, 3),
        z=round(space.z - product.z, 3),
    )


def _fits(space: Dimensions, product: Dimensions, allow_xy_rotation: bool) -> tuple[bool, str | None, Dimensions | None]:
    if product.x <= space.x and product.y <= space.y and product.z <= space.z:
        return True, "xyz", _remaining(space, product)

    rotated = Dimensions(x=product.y, y=product.x, z=product.z)
    if allow_xy_rotation and rotated.x <= space.x and rotated.y <= space.y and rotated.z <= space.z:
        return True, "yxz", _remaining(space, rotated)

    return False, None, None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "matching-engine",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "vector_store_configured": bool(OPENAI_VECTOR_STORE_ID),
        "firebase_storage_base_uri": FIREBASE_STORAGE_BASE_URI,
        "model": DEFAULT_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "perception_scale_unit": PERCEPTION_SCALE_UNIT,
    }


@app.post("/products/fit", response_model=FitQueryResponse)
def fit_products(request: FitQueryRequest) -> FitQueryResponse:
    return _fit_products(request)


def _fit_products(request: FitQueryRequest) -> FitQueryResponse:
    effective_space = Dimensions(
        x=max(request.space.x - request.clearance_cm, 0.001),
        y=max(request.space.y - request.clearance_cm, 0.001),
        z=max(request.space.z - request.clearance_cm, 0.001),
    )

    products: list[FitProduct] = []
    for product in request.products:
        if product.dimensions is None or product.name is None:
            continue

        fits, orientation, remaining = _fits(
            effective_space,
            product.dimensions,
            request.allow_xy_rotation,
        )
        products.append(
            FitProduct(
                id=product.id,
                name=product.name,
                dimensions=product.dimensions,
                fits=fits,
                orientation=orientation,
                remaining_cm=remaining,
                storage_uri=_storage_uri(product.storage_path),
                price=product.price,
                category=product.category,
                product_url=product.product_url,
                metadata=product.metadata,
            )
        )

    fitting_products = [product for product in products if product.fits]
    fitting_products.sort(
        key=lambda product: (
            product.remaining_cm.x + product.remaining_cm.y + product.remaining_cm.z
            if product.remaining_cm
            else float("inf")
        )
    )

    return FitQueryResponse(
        status="success",
        space=effective_space,
        clearance_cm=request.clearance_cm,
        firebase_storage_base_uri=FIREBASE_STORAGE_BASE_URI,
        fit_count=len(fitting_products),
        products=fitting_products,
    )


def _candidate_text(product: FitProduct) -> str:
    parts = [
        f"name: {product.name}",
        f"category: {product.category or ''}",
        f"dimensions_cm: x={product.dimensions.x}, y={product.dimensions.y}, z={product.dimensions.z}",
    ]
    if product.price is not None:
        parts.append(f"price: {product.price}")
    if product.remaining_cm:
        parts.append(
            "remaining_cm: "
            f"x={product.remaining_cm.x}, y={product.remaining_cm.y}, z={product.remaining_cm.z}"
        )
    if product.metadata:
        parts.append(f"metadata: {json.dumps(product.metadata, ensure_ascii=False)}")
    return "\n".join(parts)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _rank_products_by_embedding(client: OpenAI, query: str, products: list[FitProduct]) -> list[FitProduct]:
    if not products:
        return products

    inputs = [query] + [_candidate_text(product) for product in products]
    embedding_response = client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=inputs,
    )
    vectors = [item.embedding for item in embedding_response.data]
    query_vector = vectors[0]

    for product, vector in zip(products, vectors[1:]):
        product.relevance_score = round(_cosine_similarity(query_vector, vector), 6)

    return sorted(
        products,
        key=lambda product: (
            product.relevance_score if product.relevance_score is not None else -1,
            -(
                product.remaining_cm.x + product.remaining_cm.y + product.remaining_cm.z
                if product.remaining_cm
                else float("inf")
            ),
        ),
        reverse=True,
    )


@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    client = _client()
    model = request.model or DEFAULT_MODEL
    tools = []
    retrieval = {"mode": "provided_context"}

    if OPENAI_VECTOR_STORE_ID:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
            }
        )
        retrieval = {
            "mode": "openai_file_search",
            "vector_store_id": OPENAI_VECTOR_STORE_ID,
        }

    try:
        create_kwargs: dict[str, Any] = {
            "model": model,
            "input": _build_input(request),
        }
        if tools:
            create_kwargs["tools"] = tools

        response = client.responses.create(**create_kwargs)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    return RagQueryResponse(
        status="success",
        answer=response.output_text,
        model=model,
        retrieval=retrieval,
    )


@app.post("/rag/fit-query", response_model=FitRagQueryResponse)
def rag_fit_query(request: FitRagQueryRequest) -> FitRagQueryResponse:
    client = _client()
    fit = _fit_products(request)
    if not fit.products:
        return FitRagQueryResponse(
            status="success",
            answer="입력된 공간 치수에 맞는 상품 후보를 찾지 못했습니다.",
            model=request.model or DEFAULT_MODEL,
            retrieval={"mode": "no_fit_candidates"},
            fit=fit,
        )

    try:
        fit.products = _rank_products_by_embedding(client, request.query, fit.products)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings error: {exc}") from exc

    candidates = [product.model_dump() for product in fit.products]
    rag_response = rag_query(
        RagQueryRequest(
            query=request.query,
            context=request.context,
            obj_id=request.obj_id,
            room_context=request.room_context,
            model=request.model,
            fit_candidates=candidates,
        )
    )

    return FitRagQueryResponse(
        status="success",
        answer=rag_response.answer,
        model=rag_response.model,
        retrieval=rag_response.retrieval,
        fit=fit,
    )


@app.post("/rag/perception-query", response_model=FitRagQueryResponse)
def rag_perception_query(request: PerceptionRagQueryRequest) -> FitRagQueryResponse:
    client = _client()
    perception = _select_perception_object(request.perception_objects, request.target_object_id)
    available_space = request.space or _space_from_perception(perception)
    product_source = request.products or _load_products_from_db(perception.category, request.limit)
    if not product_source:
        raise HTTPException(status_code=404, detail="No product candidates were found for RAG.")

    category_products = [
        product
        for product in product_source
        if _category_matches(perception.category, product)
    ]

    fit = _fit_products(
        FitQueryRequest(
            space=available_space,
            products=category_products or product_source,
            clearance_cm=request.clearance_cm,
            allow_xy_rotation=request.allow_xy_rotation,
        )
    )
    if not fit.products:
        return FitRagQueryResponse(
            status="success",
            answer="감지된 객체의 공간 치수에 들어갈 수 있는 상품 후보를 찾지 못했습니다.",
            model=request.model or DEFAULT_MODEL,
            retrieval={
                "mode": "no_fit_candidates",
                "perception_category": perception.category,
            },
            fit=fit,
            perception=perception,
        )

    try:
        fit.products = _rank_products_by_embedding(client, request.query, fit.products)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings error: {exc}") from exc

    room_context = request.room_context or {}
    rag_response = rag_query(
        RagQueryRequest(
            query=request.query,
            obj_id=str(perception.id),
            room_context={
                **room_context,
                "perception": perception.model_dump(),
                "available_space_cm": available_space.model_dump(),
                "product_schema": {
                    "name": "product_name",
                    "x": "width_x",
                    "y": "depth_y",
                    "z": "height_z",
                    "unit": "cm",
                },
            },
            model=request.model,
            fit_candidates=[product.model_dump() for product in fit.products],
        )
    )

    return FitRagQueryResponse(
        status="success",
        answer=rag_response.answer,
        model=rag_response.model,
        retrieval={
            **rag_response.retrieval,
            "mode": "perception_dimension_embedding_rag",
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
        },
        fit=fit,
        perception=perception,
    )
