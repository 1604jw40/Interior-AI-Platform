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
    EMBEDDING_CACHE_PATH,
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
NO_FIT_ANSWER_KO = "요청한 공간에 들어가는 상품 후보가 없습니다."
NO_PERCEPTION_FIT_ANSWER_KO = "감지된 객체의 사용 가능 공간에 들어가는 상품 후보가 없습니다."

CATEGORY_ALIASES = {
    "bed": {"bed", "beds", "mattress", "mattresses", "침대", "매트리스", "beds_mattresses"},
    "beds_mattresses": {"bed", "beds", "mattress", "mattresses", "침대", "매트리스", "beds_mattresses"},
    "desk": {"desk", "desks", "office", "office chair", "office_chair", "책상", "사무용", "desks_office_chairs"},
    "desks_office_chairs": {"desk", "desks", "office", "chair", "office chair", "책상", "의자", "desks_office_chairs"},
    "sofa": {"sofa", "sofas", "armchair", "armchairs", "소파", "암체어", "sofas_armchairs"},
    "sofas_armchairs": {"sofa", "sofas", "armchair", "armchairs", "소파", "암체어", "sofas_armchairs"},
    "storage": {"storage", "cabinet", "shelf", "shelves", "수납", "수납장", "선반", "storage_accessories"},
    "storage_accessories": {"storage", "cabinet", "shelf", "shelves", "수납", "수납장", "선반", "storage_accessories"},
    "table": {"table", "tables", "chair", "chairs", "테이블", "식탁", "의자", "tables_chairs"},
    "tables_chairs": {"table", "tables", "chair", "chairs", "테이블", "식탁", "의자", "tables_chairs"},
}

_EMBEDDING_CACHE: dict[str, list[float]] | None = None

INTENT_KEYWORDS = {
    "bed": {"bed", "mattress", "day-bed", "daybed", "bedframe", "cot"},
    "beds_mattresses": {"bed", "mattress", "day-bed", "daybed", "bedframe", "cot"},
    "desk": {"desk", "table", "workstation", "laptop", "office chair", "swivel chair", "chair", "stool"},
    "desks_office_chairs": {"desk", "table", "workstation", "laptop", "office chair", "swivel chair", "chair", "stool"},
    "sofa": {"sofa", "couch", "armchair", "chaise", "seat", "seater"},
    "sofas_armchairs": {"sofa", "couch", "armchair", "chaise", "seat", "seater"},
    "storage": {"cabinet", "shelf", "shelving", "wardrobe", "drawer", "chest", "bookcase", "storage"},
    "storage_accessories": {"cabinet", "shelf", "shelving", "wardrobe", "drawer", "chest", "bookcase", "storage"},
    "table": {"table", "dining", "coffee table", "desk", "chair", "stool", "bench"},
    "tables_chairs": {"table", "dining", "coffee table", "desk", "chair", "stool", "bench"},
}

ACCESSORY_KEYWORDS = {
    "accessory",
    "accessories",
    "holder",
    "organizer",
    "organiser",
    "insert",
    "box",
    "basket",
    "cover",
    "cushion",
    "pad",
    "roll",
    "paper",
    "lamp",
    "stand",
    "hook",
    "rail",
    "tray",
}


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
        "embedding_cache_path": str(EMBEDDING_CACHE_PATH),
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
        "Return only one valid JSON object. Do not wrap it in markdown.",
        "The JSON schema must be: {\"summary\": string, \"recommendations\": [{\"product_id\": string, \"product_name\": string, \"rank\": number, \"reason\": string, \"fit_reason\": string, \"category\": string, \"dimensions_cm\": {\"x\": number, \"y\": number, \"z\": number}, \"remaining_cm\": {\"x\": number, \"y\": number, \"z\": number}, \"scores\": {\"fit_score\": number, \"semantic_score\": number, \"clearance_score\": number, \"final_score\": number}, \"storage_uri\": string, \"product_url\": string}], \"warnings\": [string], \"trace\": object}.",
        "Keep Korean text in summary, reason, fit_reason, and warnings.",
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


def category_terms(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.lower().replace("&", " ").replace("-", "_").replace(" ", "_")
    raw_terms = {normalized}
    raw_terms.update(part for part in normalized.split("_") if part)

    expanded = set(raw_terms)
    for key, aliases in CATEGORY_ALIASES.items():
        if key in raw_terms or raw_terms.intersection(aliases):
            expanded.update(aliases)
            expanded.add(key)
    return expanded


def normalize_search_text(value: str | None) -> str:
    if not value:
        return ""
    return value.lower().replace("_", " ").replace("-", " ")


def product_search_text(product: FitProduct) -> str:
    metadata = product.metadata or {}
    trace = product.trace or {}
    parts = [
        product.name,
        product.category,
        trace.get("category_code"),
        metadata.get("raw_category_label"),
        " ".join(metadata.get("category_path") or []),
        " ".join(metadata.get("texture_keywords") or []),
    ]
    return normalize_search_text(" ".join(part for part in parts if part))


def query_intent_keys(query: str) -> set[str]:
    terms = category_terms(query)
    normalized_query = normalize_search_text(query)
    keys = {
        key
        for key, aliases in CATEGORY_ALIASES.items()
        if key in terms or any(normalize_search_text(alias) in normalized_query for alias in aliases)
    }
    for key, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized_query for keyword in keywords):
            keys.add(key)
    return keys


def score_product_intent(query: str, product: FitProduct) -> float:
    intent_keys = query_intent_keys(query)
    if not intent_keys:
        product_terms = category_terms(product.category)
        intent_keys = {
            key
            for key, aliases in CATEGORY_ALIASES.items()
            if key in product_terms or product_terms.intersection(aliases)
        }
    text = product_search_text(product)
    if not text:
        return 0.5

    positive_keywords = set()
    for key in intent_keys:
        positive_keywords.update(INTENT_KEYWORDS.get(key, set()))

    positive_hits = sum(1 for keyword in positive_keywords if keyword in text)
    accessory_hits = sum(1 for keyword in ACCESSORY_KEYWORDS if keyword in text)

    score = 0.5
    if positive_hits:
        score += min(0.35, positive_hits * 0.12)
    if accessory_hits:
        score -= min(0.35, accessory_hits * 0.10)
    if intent_keys and not positive_hits:
        score -= 0.15
    return round(max(0.0, min(score, 1.0)), 6)


def category_matches(perception_category: str, product: ProductCandidate) -> bool:
    product_category = product.category or product.category_code
    if not product_category:
        return True
    perception_terms = category_terms(perception_category)
    product_terms = category_terms(product_category)
    product_terms.update(category_terms(product.category_code))
    if perception_terms.intersection(product_terms):
        return True
    perception_text = perception_category.lower()
    product_text = product_category.lower()
    return perception_text in product_text or product_text in perception_text


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


def deficit(space: Dimensions, product: Dimensions) -> Dimensions:
    return Dimensions(
        x=round(max(product.x - space.x, 0), 3),
        y=round(max(product.y - space.y, 0), 3),
        z=round(max(product.z - space.z, 0), 3),
    )


def fit_reason(product: Dimensions, remaining: Dimensions, orientation: str) -> str:
    return (
        f"x/y/z 기준 통과: orientation={orientation}, "
        f"상품 크기 {product.x}x{product.y}x{product.z}cm, "
        f"남은 공간 {remaining.x}x{remaining.y}x{remaining.z}cm"
    )


def reject_reason(space: Dimensions, product: Dimensions, allow_xy_rotation: bool) -> str:
    options = [("xyz", product)]
    if allow_xy_rotation:
        options.append(("yxz", Dimensions(x=product.y, y=product.x, z=product.z)))

    orientation, rejected_product = min(
        options,
        key=lambda option: deficit(space, option[1]).x + deficit(space, option[1]).y + deficit(space, option[1]).z,
    )
    shortage = deficit(space, rejected_product)
    shortage_parts = [
        f"{axis}축 {amount}cm 초과"
        for axis, amount in (("x", shortage.x), ("y", shortage.y), ("z", shortage.z))
        if amount > 0
    ]
    return f"공간 부족: orientation={orientation}, " + ", ".join(shortage_parts)


def score_fit(space: Dimensions, product: Dimensions, remaining: Dimensions | None) -> tuple[float, float]:
    if remaining is None:
        return 0.0, 0.0
    total_space = max(space.x + space.y + space.z, 0.001)
    total_used = min(product.x, space.x) + min(product.y, space.y) + min(product.z, space.z)
    fit_score = round(total_used / total_space, 6)
    clearance_score = round(
        (
            remaining.x / max(space.x, 0.001)
            + remaining.y / max(space.y, 0.001)
            + remaining.z / max(space.z, 0.001)
        )
        / 3,
        6,
    )
    return fit_score, clearance_score


def score_size(space: Dimensions, product: Dimensions) -> float:
    space_volume = max(space.x * space.y * space.z, 0.001)
    product_volume = max(product.x * product.y * product.z, 0.001)
    ratio = product_volume / space_volume
    if ratio < 0.04:
        return round(max(0.05, ratio / 0.04 * 0.35), 6)
    if ratio < 0.12:
        return round(0.35 + ((ratio - 0.04) / 0.08 * 0.35), 6)
    if ratio <= 0.75:
        return 1.0
    return round(max(0.25, 1.0 - ((ratio - 0.75) / 0.25 * 0.45)), 6)


def apply_product_quality_scores(query: str, product: FitProduct) -> None:
    product.intent_score = score_product_intent(query, product)
    if product.remaining_cm:
        oriented_space = Dimensions(
            x=product.dimensions.x + product.remaining_cm.x,
            y=product.dimensions.y + product.remaining_cm.y,
            z=product.dimensions.z + product.remaining_cm.z,
        )
        product.size_score = score_size(oriented_space, product.dimensions)
    else:
        product.size_score = 0.0
    update_final_score(product)


def update_final_score(product: FitProduct) -> None:
    if not product.fits:
        product.final_score = 0.0
        return
    semantic = product.semantic_score if product.semantic_score is not None else product.relevance_score
    semantic = semantic if semantic is not None else 0.5
    fit_score = product.fit_score if product.fit_score is not None else 0.0
    clearance_score = product.clearance_score if product.clearance_score is not None else 0.0
    intent_score = product.intent_score if product.intent_score is not None else 0.5
    size_score = product.size_score if product.size_score is not None else 0.5
    product.final_score = round(
        (semantic * 0.40)
        + (fit_score * 0.25)
        + (intent_score * 0.18)
        + (size_score * 0.12)
        + (clearance_score * 0.05),
        6,
    )


def fits(space: Dimensions, product: Dimensions, allow_xy_rotation: bool) -> tuple[bool, str | None, Dimensions | None, Dimensions | None]:
    if product.x <= space.x and product.y <= space.y and product.z <= space.z:
        return True, "xyz", remaining_space(space, product), product

    rotated = Dimensions(x=product.y, y=product.x, z=product.z)
    if allow_xy_rotation and rotated.x <= space.x and rotated.y <= space.y and rotated.z <= space.z:
        return True, "yxz", remaining_space(space, rotated), rotated

    return False, None, None, None


def fit_products(request: FitQueryRequest) -> FitQueryResponse:
    effective_space = Dimensions(
        x=max(request.space.x - request.clearance_cm, 0.001),
        y=max(request.space.y - request.clearance_cm, 0.001),
        z=max(request.space.z - request.clearance_cm, 0.001),
    )

    products: list[FitProduct] = []
    rejected_products: list[FitProduct] = []
    for product in request.products:
        if product.dimensions is None or product.name is None:
            continue

        does_fit, orientation, remaining, oriented_dimensions = fits(
            effective_space,
            product.dimensions,
            request.allow_xy_rotation,
        )
        scored_dimensions = oriented_dimensions or product.dimensions
        fit_score, clearance_score = score_fit(effective_space, scored_dimensions, remaining)
        size_score = score_size(effective_space, scored_dimensions) if does_fit else 0.0
        fit_product = FitProduct(
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
            fit_reason=fit_reason(scored_dimensions, remaining, orientation) if does_fit and remaining and orientation else None,
            reject_reason=None if does_fit else reject_reason(effective_space, product.dimensions, request.allow_xy_rotation),
            fit_score=fit_score,
            clearance_score=clearance_score,
            size_score=size_score,
            trace={
                "source": product.metadata.get("source") if product.metadata else "request",
                "product_id": product.id,
                "product_name": product.name,
                "category": product.category,
                "category_code": product.category_code,
                "storage_path": product.storage_path,
                "storage_uri": storage_uri(product.storage_path),
                "product_url": product.product_url,
                "orientation": orientation,
                "allow_xy_rotation": request.allow_xy_rotation,
                "scoring": {
                    "fit_score": fit_score,
                    "clearance_score": clearance_score,
                    "size_score": size_score,
                },
                "dimensions_source": {
                    "width_x": product.width_x,
                    "depth_y": product.depth_y,
                    "height_z": product.height_z,
                },
            },
            metadata=product.metadata,
        )
        update_final_score(fit_product)
        if does_fit:
            products.append(fit_product)
        else:
            rejected_products.append(fit_product)

    products.sort(
        key=lambda product: (
            product.final_score if product.final_score is not None else 0,
            product.fit_score if product.fit_score is not None else 0,
        ),
        reverse=True,
    )

    return FitQueryResponse(
        status="success",
        space=effective_space,
        clearance_cm=request.clearance_cm,
        firebase_storage_base_uri=FIREBASE_STORAGE_BASE_URI,
        fit_count=len(products),
        rejected_count=len(rejected_products),
        products=products,
        rejected_products=rejected_products[:10],
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
    if product.fit_score is not None:
        parts.append(f"fit_score: {product.fit_score}")
    if product.semantic_score is not None:
        parts.append(f"semantic_score: {product.semantic_score}")
    if product.clearance_score is not None:
        parts.append(f"clearance_score: {product.clearance_score}")
    if product.intent_score is not None:
        parts.append(f"intent_score: {product.intent_score}")
    if product.size_score is not None:
        parts.append(f"size_score: {product.size_score}")
    if product.final_score is not None:
        parts.append(f"final_score: {product.final_score}")
    if product.fit_reason:
        parts.append(f"fit_reason: {product.fit_reason}")
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


def embedding_cache_key(text: str) -> str:
    source = f"{DEFAULT_EMBEDDING_MODEL}:{text}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def load_embedding_cache() -> dict[str, list[float]]:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is not None:
        return _EMBEDDING_CACHE
    try:
        if EMBEDDING_CACHE_PATH.exists():
            data = json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _EMBEDDING_CACHE = {
                    key: value
                    for key, value in data.items()
                    if isinstance(value, list)
                }
                return _EMBEDDING_CACHE
    except OSError:
        pass
    except json.JSONDecodeError:
        pass
    _EMBEDDING_CACHE = {}
    return _EMBEDDING_CACHE


def save_embedding_cache(cache: dict[str, list[float]]) -> None:
    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EMBEDDING_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        return


def cached_embeddings(client: OpenAI, inputs: list[str]) -> list[list[float]]:
    cache = load_embedding_cache()
    result: list[list[float] | None] = []
    missing_inputs: list[str] = []
    missing_indexes: list[int] = []

    for index, text in enumerate(inputs):
        key = embedding_cache_key(text)
        cached = cache.get(key)
        if cached:
            result.append(cached)
        else:
            result.append(None)
            missing_inputs.append(text)
            missing_indexes.append(index)

    if missing_inputs:
        embedding_response = client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=missing_inputs,
        )
        for index, item in zip(missing_indexes, embedding_response.data):
            vector = item.embedding
            result[index] = vector
            cache[embedding_cache_key(inputs[index])] = vector
        save_embedding_cache(cache)

    return [vector for vector in result if vector is not None]


def rank_products_by_embedding(client: OpenAI, query: str, products: list[FitProduct]) -> list[FitProduct]:
    if not products:
        return products

    inputs = [query] + [candidate_text(product) for product in products]
    vectors = cached_embeddings(client, inputs)
    query_vector = vectors[0]

    for product, vector in zip(products, vectors[1:]):
        product.semantic_score = round(cosine_similarity(query_vector, vector), 6)
        product.relevance_score = product.semantic_score
        apply_product_quality_scores(query, product)

    return sorted(
        products,
        key=lambda product: (
            product.final_score if product.final_score is not None else -1,
            product.semantic_score if product.semantic_score is not None else -1,
        ),
        reverse=True,
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def product_recommendation(product: dict[str, Any], rank: int, reason: str | None = None) -> dict[str, Any]:
    dimensions = product.get("dimensions") or {}
    remaining = product.get("remaining_cm") or {}
    return {
        "product_id": str(product.get("id")),
        "product_name": product.get("name"),
        "rank": rank,
        "reason": reason or "공간 조건과 상품 관련도 기준으로 선별된 후보입니다.",
        "fit_reason": product.get("fit_reason") or "x/y/z 치수 검사를 통과했습니다.",
        "category": product.get("category"),
        "dimensions_cm": {
            "x": dimensions.get("x"),
            "y": dimensions.get("y"),
            "z": dimensions.get("z"),
        },
        "remaining_cm": {
            "x": remaining.get("x"),
            "y": remaining.get("y"),
            "z": remaining.get("z"),
        },
        "scores": {
            "fit_score": product.get("fit_score"),
            "semantic_score": product.get("semantic_score"),
            "clearance_score": product.get("clearance_score"),
            "intent_score": product.get("intent_score"),
            "size_score": product.get("size_score"),
            "final_score": product.get("final_score"),
        },
        "storage_uri": product.get("storage_uri"),
        "product_url": product.get("product_url"),
    }


def structured_from_candidates(
    fit_candidates: list[dict[str, Any]] | None,
    summary: str,
    warning: str | None = None,
) -> dict[str, Any]:
    candidates = fit_candidates or []
    recommendations = [
        product_recommendation(product, index)
        for index, product in enumerate(candidates[:MAX_RAG_CANDIDATES], start=1)
    ]
    warnings = [warning] if warning else []
    return {
        "summary": summary,
        "recommendations": recommendations,
        "warnings": warnings,
        "trace": {
            "candidate_count": len(candidates),
            "max_rag_candidates": MAX_RAG_CANDIDATES,
            "generated_by": "matching_engine_fallback" if warning else "llm_or_matching_engine",
        },
    }


def fallback_rag_response(
    request: RagQueryRequest,
    model: str,
    retrieval: dict[str, Any],
    reason: str,
) -> RagQueryResponse:
    summary = (
        "LLM 응답 생성에 실패해 후보정 결과를 기준으로 추천 후보를 반환합니다."
        if request.fit_candidates
        else "LLM 응답 생성에 실패했고 제공된 후보가 없어 추천을 생성할 수 없습니다."
    )
    structured = structured_from_candidates(request.fit_candidates, summary, reason)
    return RagQueryResponse(
        status="degraded",
        answer=structured["summary"],
        structured_answer=structured,
        model=model,
        retrieval={
            **retrieval,
            "llm_fallback": True,
            "fallback_reason": reason,
        },
    )


def rag_query(request: RagQueryRequest) -> RagQueryResponse:
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
        client = openai_client()
        create_kwargs: dict[str, Any] = {
            "model": model,
            "input": build_input(request),
        }
        if tools:
            create_kwargs["tools"] = tools

        response = client.responses.create(**create_kwargs)
    except OpenAIError as exc:
        return fallback_rag_response(request, model, retrieval, f"OpenAI API error: {exc}")
    except HTTPException as exc:
        return fallback_rag_response(request, model, retrieval, str(exc.detail))

    structured_answer = parse_json_object(response.output_text)
    answer = (
        structured_answer.get("summary")
        if structured_answer and isinstance(structured_answer.get("summary"), str)
        else response.output_text
    )

    return RagQueryResponse(
        status="success",
        answer=answer,
        structured_answer=structured_answer,
        model=model,
        retrieval={
            **retrieval,
            "structured_response": structured_answer is not None,
        },
    )


def rag_fit_query(request: FitRagQueryRequest) -> FitRagQueryResponse:
    fit = fit_products(request)
    if not fit.products:
        structured = structured_from_candidates([], NO_FIT_ANSWER_KO)
        return FitRagQueryResponse(
            status="success",
            answer=NO_FIT_ANSWER_KO,
            structured_answer=structured,
            model=request.model or DEFAULT_MODEL,
            retrieval={"mode": "no_fit_candidates"},
            fit=fit,
        )

    embedding_fallback_reason = None
    try:
        client = openai_client()
        fit.products = rank_products_by_embedding(client, request.query, fit.products)
    except OpenAIError as exc:
        embedding_fallback_reason = f"OpenAI embeddings error: {exc}"
    except HTTPException as exc:
        embedding_fallback_reason = str(exc.detail)
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
        status=rag_response.status,
        answer=rag_response.answer,
        structured_answer=rag_response.structured_answer,
        model=rag_response.model,
        retrieval={
            **rag_response.retrieval,
            "embedding_cache_path": str(EMBEDDING_CACHE_PATH),
            "embedding_fallback": embedding_fallback_reason is not None,
            "embedding_fallback_reason": embedding_fallback_reason,
        },
        fit=fit,
    )


def rag_perception_query(request: PerceptionRagQueryRequest) -> FitRagQueryResponse:
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
        structured = structured_from_candidates([], NO_PERCEPTION_FIT_ANSWER_KO)
        return FitRagQueryResponse(
            status="success",
            answer=NO_PERCEPTION_FIT_ANSWER_KO,
            structured_answer=structured,
            model=request.model or DEFAULT_MODEL,
            retrieval={
                "mode": "no_fit_candidates",
                "perception_category": perception.category,
            },
            fit=fit,
            perception=perception,
        )

    embedding_fallback_reason = None
    try:
        client = openai_client()
        fit.products = rank_products_by_embedding(client, request.query, fit.products)
    except OpenAIError as exc:
        embedding_fallback_reason = f"OpenAI embeddings error: {exc}"
    except HTTPException as exc:
        embedding_fallback_reason = str(exc.detail)
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
        status=rag_response.status,
        answer=rag_response.answer,
        structured_answer=rag_response.structured_answer,
        model=rag_response.model,
        retrieval={
            **rag_response.retrieval,
            "mode": "perception_dimension_embedding_rag",
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_cache_path": str(EMBEDDING_CACHE_PATH),
            "embedding_fallback": embedding_fallback_reason is not None,
            "embedding_fallback_reason": embedding_fallback_reason,
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
