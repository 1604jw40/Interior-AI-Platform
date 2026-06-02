from typing import Any

from .services import load_products_from_raw, storage_uri


def find_best_candidates(category: str, max_width: float, max_height: float, limit: int = 50) -> list[dict[str, Any]]:
    """Legacy matcher wrapper backed by product_db/data/raw JSON.

    The active FastAPI routes use services.rag_perception_query directly. This
    wrapper remains for older callers that still import db_matcher.py.
    """
    products = load_products_from_raw(category, limit)
    candidates: list[dict[str, Any]] = []

    for product in products:
        if product.dimensions is None:
            continue

        footprint_fits = product.dimensions.x <= max_width and product.dimensions.y <= max_width
        height_fits = product.dimensions.z <= max_height
        if not (footprint_fits and height_fits):
            continue

        candidates.append(
            {
                "product_id": product.id,
                "name": product.name,
                "size": f"{product.dimensions.x}x{product.dimensions.y}x{product.dimensions.z}cm",
                "category": product.category,
                "category_code": product.category_code,
                "price": product.price,
                "product_url": product.product_url,
                "storage_uri": storage_uri(product.storage_path),
                "image_filename": (product.metadata or {}).get("image_filename"),
                "source": "raw_json",
            }
        )

    return candidates
