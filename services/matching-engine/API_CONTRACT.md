# Matching Engine API Contract

백엔드, Unity, ai-perception 서비스가 `matching-engine` LLM 서버와 붙을 때 사용하는 API 계약입니다.

## Base URL

```text
http://matching-engine:8010
http://localhost:8010
```

## Product Data Source

`/rag/perception-query`에서 `products`를 생략하면 서버가 Postgres가 아니라 raw JSON 파일을 읽습니다.

```text
infrastructure/product_db/data/raw/*.json
```

현재 필요한 raw 필드:

```text
product_name
price
width_x
depth_y
height_z
top_category_code
top_category_name
product_url
image_filename
image_path
```

이미지 URI는 아래 규칙으로 만듭니다.

```text
{FIREBASE_STORAGE_BASE_URI}/{FIREBASE_STORAGE_PREFIX}/{raw_category}/{image_filename}
```

기본값:

```text
FIREBASE_STORAGE_BASE_URI=gs://interiorplatform-d58e0.firebasestorage.app
FIREBASE_STORAGE_PREFIX=furniture_images
```

## Recommendation Policy

- LLM 답변은 한국어입니다.
- LLM은 `fit.products`에 들어간 상품만 추천합니다.
- x/y/z 치수 필터를 통과하지 못한 상품은 LLM에 전달하지 않습니다.
- 추천 우선순위는 물리적 fit, 카테고리 유사도, 남는 공간, embedding relevance, 가격입니다.
- 상품명, 치수, 가격, URL, 이미지 경로는 제공된 raw JSON과 trace만 근거로 사용합니다.

## Trace Policy

각 추천 상품에는 백엔드 추적용 `trace`와 `metadata`가 포함됩니다.

```json
{
  "trace": {
    "source": "request_or_db",
    "product_id": "e4d6f1a8c912",
    "product_name": "VATTENKAR Desktop shelf - white 49x15 cm",
    "category": "Storage accessories",
    "category_code": "storage_accessories",
    "storage_path": "furniture_images/Storage_accessories/Storage accessories_00001.jpg",
    "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/Storage_accessories/Storage accessories_00001.jpg",
    "product_url": "https://www.ikea.com/kr/en/p/vattenkar-desktop-shelf-white-00541569/",
    "dimensions_source": {
      "width_x": 49,
      "depth_y": 15,
      "height_z": 36
    }
  },
  "metadata": {
    "source": "raw_json",
    "raw_category_label": "Storage_accessories",
    "image_filename": "Storage accessories_00001.jpg"
  }
}
```

## 1. Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "matching-engine",
  "openai_configured": true,
  "vector_store_configured": false,
  "firebase_storage_base_uri": "gs://interiorplatform-d58e0.firebasestorage.app",
  "firebase_storage_prefix": "furniture_images",
  "model": "gpt-4.1-mini",
  "embedding_model": "text-embedding-3-small",
  "perception_scale_unit": "m",
  "product_source": "raw_json",
  "product_raw_data_dir": "..."
}
```

## 2. Dimension Fit Only

```http
POST /products/fit
```

Request:

```json
{
  "space": { "x": 120, "y": 60, "z": 75 },
  "clearance_cm": 2,
  "allow_xy_rotation": true,
  "products": [
    {
      "id": "item-1",
      "product_name": "VATTENKAR Desktop shelf - white 49x15 cm",
      "category_code": "storage_accessories",
      "price": 19900,
      "width_x": 49,
      "depth_y": 15,
      "height_z": 36,
      "storage_path": "furniture_images/Storage_accessories/Storage accessories_00001.jpg",
      "product_url": "https://www.ikea.com/..."
    }
  ]
}
```

## 3. Fit + LLM RAG

```http
POST /rag/fit-query
```

`products`를 직접 넣어 수동 후보로 LLM 추천을 받을 때 사용합니다.

## 4. Perception + Raw JSON + LLM RAG

메인 연동 API입니다.

```http
POST /rag/perception-query
```

Request:

```json
{
  "query": "감지된 객체 자리에 들어갈 수 있는 상품을 추천해줘",
  "target_object_id": 0,
  "clearance_cm": 2,
  "allow_xy_rotation": true,
  "limit": 100,
  "perception_objects": [
    {
      "id": 0,
      "label": "desk|asset_desk_0.ply|1.20|0.75",
      "confidence": 0.92,
      "bbox": { "x_min": 100, "y_min": 120, "x_max": 420, "y_max": 520 },
      "position_3d": { "x": 0.1, "y": 0.0, "z": 2.1 }
    }
  ]
}
```

Response 주요 구조:

```json
{
  "status": "success",
  "answer": "가장 적합한 상품은 ... 입니다.",
  "model": "gpt-4.1-mini",
  "retrieval": {
    "mode": "perception_dimension_embedding_rag",
    "embedding_model": "text-embedding-3-small",
    "product_source": "raw_json",
    "perception_category": "desk",
    "target_object_id": 0,
    "candidate_count": 100,
    "fit_count": 5,
    "max_rag_candidates": 5
  },
  "fit": {
    "fit_count": 5,
    "products": [
      {
        "id": "e4d6f1a8c912",
        "name": "VATTENKAR Desktop shelf - white 49x15 cm",
        "dimensions": { "x": 49, "y": 15, "z": 36 },
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/Storage_accessories/Storage accessories_00001.jpg",
        "price": 19900,
        "category": "Storage accessories",
        "product_url": "https://www.ikea.com/kr/en/p/vattenkar-desktop-shelf-white-00541569/",
        "trace": {
          "category_code": "storage_accessories",
          "storage_path": "furniture_images/Storage_accessories/Storage accessories_00001.jpg"
        },
        "metadata": {
          "source": "raw_json",
          "raw_category_label": "Storage_accessories",
          "image_filename": "Storage accessories_00001.jpg"
        }
      }
    ]
  },
  "perception": {
    "id": 0,
    "category": "desk",
    "asset_filename": "asset_desk_0.ply",
    "estimated_width": 1.2,
    "estimated_height": 0.75
  }
}
```

## Error Cases

OpenAI API key가 없을 때:

```json
{
  "detail": "OPENAI_API_KEY is not configured for matching-engine."
}
```

raw JSON 경로가 없을 때:

```json
{
  "detail": "Product raw data directory was not found: ..."
}
```
