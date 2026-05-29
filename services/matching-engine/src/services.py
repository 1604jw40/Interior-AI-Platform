import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from openai import OpenAI, OpenAIError

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    FIREBASE_STORAGE_BASE_URI,
    FIREBASE_STORAGE_PREFIX,
    MAX_RAG_CANDIDATES,
    OPENAI_VECTOR_STORE_ID,
    PERCEPTION_SCALE_UNIT,
    PRODUCT_RAW_DATA_DIR,
)
from .schemas import (
    Dimensions,
    FitProduct,
    FitQueryRequest,
    FitQueryResponse,
    FitRagQueryRequest,
    FitRagQueryResponse,
    ParsedPerceptionObject,
    PerceptionObject,
    PerceptionRagQueryRequest,
    ProductCandidate,
    RagQueryRequest,
    RagQueryResponse,
)


NO_FIT_ANSWER = "No product candidates fit the requested space."
NO_PERCEPTION_FIT_ANSWER = "No product candidates fit the detected object's available space."


def health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "matching-engine",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "vector_store_configured": bool(OPENAI_VECTOR_STORE_ID),
        "firebase_storage_base_uri": FIREBASE_STORAGE_BASE_URI,
        "firebase_storage_prefix": FIREBASE_STORAGE_PREFIX,
        "model": DEFAULT_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "perception_scale_unit": PERCEPTION_SCALE_UNIT,
        "product_source": "raw_json",
        "product_raw_data_dir": str(PRODUCT_RAW_DATA_DIR),
    }


def openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured for matching-engine.",
        )
    return OpenAI()


def build_input(request: RagQueryRequest) -> str:
    parts = [
        "You are an interior product matching assistant.",
        "Always answer in Korean.",
        "Use only the provided candidates and context as the source of truth.",
        "Never recommend a product that is not listed in fit_candidates.",
        "Never invent dimensions, prices, URLs, images, model paths, or product names.",
        "Prioritize in this order: physical fit, category match, useful remaining space, relevance_score, price.",
        "If every candidate is too tight or unsafe, say that no safe recommendation is available.",
        "Keep the answer concise and include the chosen product id, name, dimensions, remaining space, category, storage_uri, and product_url when available.",
        "Mention trace identifiers only when they help backend debugging; do not expose internal implementation details beyond provided trace fields.",
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
        candidates = request.fit_candidates[:MAX_RAG_CANDIDATES]
        candidates_json = json.dumps(candidates, ensure_ascii=False)
        parts.append(f"Candidate count included: {len(candidates)}")
        parts.append(f"Dimension-filtered fit candidates JSON: {candidates_json}")

    return "\n".join(parts)


def storage_uri(storage_path: str | None) -> str | None:
    if not storage_path:
        return None
    if storage_path.startswith("gs://"):
        return storage_path
    return f"{FIREBASE_STORAGE_BASE_URI}/{storage_path.lstrip('/')}"


def firebase_storage_path(category_label: str | None, image_filename: str | None) -> str | None:
    if not image_filename:
        return None
    if image_filename.startswith("gs://") or image_filename.startswith("http://") or image_filename.startswith("https://"):
        return image_filename
    category = (category_label or "uncategorized").strip().strip("/")
    return f"{FIREBASE_STORAGE_PREFIX}/{category}/{image_filename}"


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_perception_label(label: str) -> tuple[str, str | None, float | None, float | None]:
    parts = label.split("|")
    category = parts[0].strip() if parts else label.strip()
    asset_filename = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    estimated_width = safe_float(parts[2]) if len(parts) > 2 else None
    estimated_height = safe_float(parts[3]) if len(parts) > 3 else None
    return category, asset_filename, estimated_width, estimated_height


def parse_perception_object(obj: PerceptionObject) -> ParsedPerceptionObject:
    category, asset_filename, estimated_width, estimated_height = parse_perception_label(obj.label)
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


def select_perception_object(
    objects: list[PerceptionObject],
    target_object_id: int | None,
) -> ParsedPerceptionObject:
    if target_object_id is not None:
        for obj in objects:
            if obj.id == target_object_id:
                return parse_perception_object(obj)
        raise HTTPException(status_code=404, detail=f"Perception object {target_object_id} was not found.")

    sorted_objects = sorted(objects, key=lambda obj: obj.confidence or 0, reverse=True)
    return parse_perception_object(sorted_objects[0])


def space_from_perception(obj: ParsedPerceptionObject) -> Dimensions:
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


def category_matches(perception_category: str, product: ProductCandidate) -> bool:
    product_category = product.category or product.category_code
    if not product_category:
        return True
    return perception_category.lower() in product_category.lower() or product_category.lower() in perception_category.lower()


def raw_data_dir() -> Path:
    path = PRODUCT_RAW_DATA_DIR
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def product_id(item: dict[str, Any], category_label: str, index: int) -> str:
    source = item.get("product_url") or item.get("image_filename") or item.get("product_name") or f"{category_label}-{index}"
    return hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]


def load_raw_product_items() -> list[tuple[str, int, dict[str, Any]]]:
    data_dir = raw_data_dir()
    if not data_dir.exists():
        raise HTTPException(status_code=500, detail=f"Product raw data directory was not found: {data_dir}")

    items: list[tuple[str, int, dict[str, Any]]] = []
    for json_path in sorted(data_dir.glob("ikea_*.json")):
        category_label = json_path.stem.removeprefix("ikea_")
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Invalid product raw JSON: {json_path}: {exc}") from exc
        if not isinstance(data, list):
            raise HTTPException(status_code=500, detail=f"Product raw JSON root must be an array: {json_path}")
        items.extend((category_label, index, item) for index, item in enumerate(data, start=1) if isinstance(item, dict))
    return items


def load_products_from_raw(category: str | None, limit: int) -> list[ProductCandidate]:
    candidates: list[ProductCandidate] = []
    for category_label, index, item in load_raw_product_items():
        category_code = item.get("top_category_code") or category_label
        category_name = item.get("top_category_name") or category_label
        image_filename = item.get("image_filename")
        storage_path = (
            item.get("firebase_storage_path")
            or item.get("firebase_gs_url")
            or firebase_storage_path(category_label, image_filename)
        )

        try:
            candidate = ProductCandidate(
                id=product_id(item, category_label, index),
                product_name=item.get("product_name"),
                price=item.get("price"),
                width_x=item.get("width_x"),
                depth_y=item.get("depth_y"),
                height_z=item.get("height_z"),
                product_url=item.get("product_url"),
                category=category_name,
                category_code=category_code,
                storage_path=storage_path,
                metadata={
                    "source": "raw_json",
                    "raw_category_label": category_label,
                    "image_filename": image_filename,
                    "image_path": item.get("image_path"),
                    "firebase_image_url": item.get("firebase_image_url"),
                    "color": item.get("color"),
                    "materials": item.get("materials"),
                    "texture_keywords": item.get("texture_keywords"),
                    "category_path": item.get("category_path"),
                },
            )
        except ValueError:
            continue

        if category is None or category_matches(category, candidate):
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates

    if category and not candidates:
        return load_products_from_raw(None, limit)
    return candidates[:limit]


def remaining_space(space: Dimensions, product: Dimensions) -> Dimensions:
    return Dimensions(
        x=round(space.x - product.x, 3),
        y=round(space.y - product.y, 3),
        z=round(space.z - product.z, 3),
    )


def fits(space: Dimensions, product: Dimensions, allow_xy_rotation: bool) -> tuple[bool, str | None, Dimensions | None]:
    if product.x <= space.x and product.y <= space.y and product.z <= space.z:
        return True, "xyz", remaining_space(space, product)

    rotated = Dimensions(x=product.y, y=product.x, z=product.z)
    if allow_xy_rotation and rotated.x <= space.x and rotated.y <= space.y and rotated.z <= space.z:
        return True, "yxz", remaining_space(space, rotated)

    return False, None, None


def fit_products(request: FitQueryRequest) -> FitQueryResponse:
    effective_space = Dimensions(
        x=max(request.space.x - request.clearance_cm, 0.001),
        y=max(request.space.y - request.clearance_cm, 0.001),
        z=max(request.space.z - request.clearance_cm, 0.001),
    )

    products: list[FitProduct] = []
    for product in request.products:
        if product.dimensions is None or product.name is None:
            continue

        does_fit, orientation, remaining = fits(
            effective_space,
            product.dimensions,
            request.allow_xy_rotation,
        )
        products.append(
            FitProduct(
                id=product.id,
                name=product.name,
                dimensions=product.dimensions,
                fits=does_fit,
                orientation=orientation,
                remaining_cm=remaining,
                storage_uri=storage_uri(product.storage_path),
                price=product.price,
                category=product.category,
                product_url=product.product_url,
                trace={
                    "source": "request_or_db",
                    "product_id": product.id,
                    "product_name": product.name,
                    "category": product.category,
                    "category_code": product.category_code,
                    "storage_path": product.storage_path,
                    "storage_uri": storage_uri(product.storage_path),
                    "product_url": product.product_url,
                    "dimensions_source": {
                        "width_x": product.width_x,
                        "depth_y": product.depth_y,
                        "height_z": product.height_z,
                    },
                },
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


def candidate_text(product: FitProduct) -> str:
    parts = [
        f"id: {product.id}",
        f"name: {product.name}",
        f"category: {product.category or ''}",
        f"dimensions_cm: x={product.dimensions.x}, y={product.dimensions.y}, z={product.dimensions.z}",
        f"storage_uri: {product.storage_uri or ''}",
    ]
    if product.price is not None:
        parts.append(f"price: {product.price}")
    if product.relevance_score is not None:
        parts.append(f"relevance_score: {product.relevance_score}")
    if product.remaining_cm:
        parts.append(
            "remaining_cm: "
            f"x={product.remaining_cm.x}, y={product.remaining_cm.y}, z={product.remaining_cm.z}"
        )
    if product.metadata:
        parts.append(f"metadata: {json.dumps(product.metadata, ensure_ascii=False)}")
    return "\n".join(parts)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def rank_products_by_embedding(client: OpenAI, query: str, products: list[FitProduct]) -> list[FitProduct]:
    if not products:
        return products

    inputs = [query] + [candidate_text(product) for product in products]
    embedding_response = client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=inputs,
    )
    vectors = [item.embedding for item in embedding_response.data]
    query_vector = vectors[0]

    for product, vector in zip(products, vectors[1:]):
        product.relevance_score = round(cosine_similarity(query_vector, vector), 6)

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


def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    client = openai_client()
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
            "input": build_input(request),
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


def rag_fit_query(request: FitRagQueryRequest) -> FitRagQueryResponse:
    client = openai_client()
    fit = fit_products(request)
    if not fit.products:
        return FitRagQueryResponse(
            status="success",
            answer=NO_FIT_ANSWER,
            model=request.model or DEFAULT_MODEL,
            retrieval={"mode": "no_fit_candidates"},
            fit=fit,
        )

    try:
        fit.products = rank_products_by_embedding(client, request.query, fit.products)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings error: {exc}") from exc
    fit.products = fit.products[:MAX_RAG_CANDIDATES]

    rag_response = rag_query(
        RagQueryRequest(
            query=request.query,
            context=request.context,
            obj_id=request.obj_id,
            room_context=request.room_context,
            model=request.model,
            fit_candidates=[product.model_dump() for product in fit.products],
        )
    )

    return FitRagQueryResponse(
        status="success",
        answer=rag_response.answer,
        model=rag_response.model,
        retrieval=rag_response.retrieval,
        fit=fit,
    )


def rag_perception_query(request: PerceptionRagQueryRequest) -> FitRagQueryResponse:
    client = openai_client()
    perception = select_perception_object(request.perception_objects, request.target_object_id)
    available_space = request.space or space_from_perception(perception)
    product_source_kind = "request" if request.products else "raw_json"
    product_source = request.products or load_products_from_raw(perception.category, request.limit)
    if not product_source:
        raise HTTPException(status_code=404, detail="No product candidates were found for RAG.")

    category_products = [
        product
        for product in product_source
        if category_matches(perception.category, product)
    ]

    fit = fit_products(
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
            answer=NO_PERCEPTION_FIT_ANSWER,
            model=request.model or DEFAULT_MODEL,
            retrieval={
                "mode": "no_fit_candidates",
                "perception_category": perception.category,
            },
            fit=fit,
            perception=perception,
        )

    try:
        fit.products = rank_products_by_embedding(client, request.query, fit.products)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings error: {exc}") from exc
    fit.products = fit.products[:MAX_RAG_CANDIDATES]

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
            "product_source": product_source_kind,
            "perception_category": perception.category,
            "target_object_id": perception.id,
            "candidate_count": len(product_source),
            "fit_count": fit.fit_count,
            "max_rag_candidates": MAX_RAG_CANDIDATES,
        },
        fit=fit,
        perception=perception,
    )
