# Matching Engine API Contract

백엔드에서 LLM/RAG 매칭 서버를 호출하기 위한 API 계약입니다.

## Base URL

```text
Docker 내부: http://matching-engine:8010
로컬 테스트: http://localhost:8010
ngrok 사용: https://{ngrok-domain}
```

## Product Source

`/rag/perception-query`에서 `products`를 생략하면 서버가 아래 raw JSON을 직접 읽습니다.

```text
infrastructure/product_db/data/raw/*.json
```

사용 필드:

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

Firebase Storage URI는 다음 규칙으로 생성합니다.

```text
{FIREBASE_STORAGE_BASE_URI}/{FIREBASE_STORAGE_PREFIX}/{raw_category}/{image_filename}
```

기본값:

```text
FIREBASE_STORAGE_BASE_URI=gs://interiorplatform-d58e0.firebasestorage.app
FIREBASE_STORAGE_PREFIX=furniture_images
```

## RAG 후보정 흐름

1. raw JSON에서 상품 후보 로딩
2. perception category와 상품 카테고리 매핑
3. x/y/z 치수 검사
4. 들어가는 상품은 `fit.products`, 탈락 상품은 `fit.rejected_products`에 사유와 함께 기록
5. `text-embedding-3-small`로 query와 후보 상품 임베딩 비교
6. `fit_score`, `semantic_score`, `clearance_score`, `final_score` 계산
7. 상위 `MAX_RAG_CANDIDATES`개만 LLM에 전달
8. LLM은 `structured_answer` JSON 형식으로 한국어 추천 생성

OpenAI 호출이 실패하면 서버가 후보정 결과를 기반으로 fallback JSON을 반환합니다. 이때 `status`는 `degraded`가 될 수 있습니다.

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
  "model": "gpt-5-mini",
  "embedding_model": "text-embedding-3-small",
  "perception_scale_unit": "m",
  "product_source": "raw_json",
  "product_raw_data_dir": "/app/product_raw",
  "embedding_cache_path": "/tmp/matching_engine_embeddings.json"
}
```

## 2. Dimension Fit Only

```http
POST /products/fit
Content-Type: application/json
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

Response 주요 필드:

```json
{
  "status": "success",
  "space": { "x": 118, "y": 58, "z": 73 },
  "clearance_cm": 2,
  "fit_count": 1,
  "rejected_count": 0,
  "products": [
    {
      "id": "item-1",
      "name": "VATTENKAR Desktop shelf - white 49x15 cm",
      "fits": true,
      "orientation": "xyz",
      "remaining_cm": { "x": 69, "y": 43, "z": 37 },
      "fit_reason": "x/y/z 기준 통과: orientation=xyz, ...",
      "fit_score": 0.56,
      "clearance_score": 0.48,
      "intent_score": 0.75,
      "size_score": 0.82,
      "final_score": 0.51,
      "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/Storage_accessories/Storage accessories_00001.jpg",
      "trace": {
        "product_id": "item-1",
        "category_code": "storage_accessories",
        "storage_path": "furniture_images/Storage_accessories/Storage accessories_00001.jpg"
      }
    }
  ],
  "rejected_products": []
}
```

## 3. Fit + LLM RAG

```http
POST /rag/fit-query
Content-Type: application/json
```

`products`를 직접 넘겨서 치수 검사, 임베딩 재정렬, LLM 추천을 받는 API입니다.

## 4. Perception + Raw JSON + LLM RAG

백엔드 메인 연동 API입니다.

```http
POST /rag/perception-query
Content-Type: application/json
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

`label` 형식:

```text
class|ply_filename|scale_w|scale_h
```

`scale_w`, `scale_h`는 기본적으로 meter로 보고 cm로 변환합니다. 명시적인 공간을 쓰려면 request에 `space`를 넣으면 됩니다.

```json
"space": { "x": 120, "y": 60, "z": 75 }
```

Response 주요 구조:

```json
{
  "status": "success",
  "answer": "가장 적합한 상품은 ...",
  "structured_answer": {
    "summary": "가장 적합한 상품은 ...",
    "recommendations": [
      {
        "product_id": "4026381ff131",
        "product_name": "PERJOHAN Stool with storage - pine",
        "rank": 1,
        "reason": "책상 주변에 둘 수 있고 공간 여유가 충분합니다.",
        "fit_reason": "x/y/z 기준 통과: orientation=xyz, ...",
        "category": "Desks & office chairs",
        "dimensions_cm": { "x": 49, "y": 28, "z": 45 },
        "remaining_cm": { "x": 69, "y": 90, "z": 33 },
        "scores": {
          "fit_score": 0.388535,
          "semantic_score": 0.8123,
          "clearance_score": 0.590178,
          "intent_score": 0.85,
          "size_score": 0.440741,
          "final_score": 0.662
        },
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/Desks_office_chairs/Desks office chairs_00002.jpg",
        "product_url": "https://www.ikea.com/kr/en/p/perjohan-stool-with-storage-pine-40501321/"
      }
    ],
    "warnings": [],
    "trace": {
      "candidate_count": 5,
      "max_rag_candidates": 5
    }
  },
  "model": "gpt-5-mini",
  "retrieval": {
    "mode": "perception_dimension_embedding_rag",
    "structured_response": true,
    "embedding_model": "text-embedding-3-small",
    "embedding_cache_path": "/tmp/matching_engine_embeddings.json",
    "embedding_fallback": false,
    "product_source": "raw_json",
    "perception_category": "desk",
    "target_object_id": 0,
    "candidate_count": 100,
    "fit_count": 5,
    "max_rag_candidates": 5
  },
  "fit": {
    "fit_count": 5,
    "rejected_count": 95,
    "products": [],
    "rejected_products": []
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

## 추적 필드

백엔드에서 상품을 추적할 때는 아래 필드를 우선 사용하면 됩니다.

```text
fit.products[].id
fit.products[].name
fit.products[].category
fit.products[].product_url
fit.products[].storage_uri
fit.products[].trace.storage_path
fit.products[].trace.category_code
fit.products[].metadata.raw_category_label
fit.products[].metadata.image_filename
structured_answer.recommendations[].product_id
```

## Fallback

OpenAI LLM 호출이 실패하면:

```json
{
  "status": "degraded",
  "answer": "LLM 응답 생성에 실패해 후보정 결과를 기준으로 추천 후보를 반환합니다.",
  "retrieval": {
    "llm_fallback": true,
    "fallback_reason": "OpenAI API error: ..."
  },
  "structured_answer": {
    "summary": "LLM 응답 생성에 실패해 후보정 결과를 기준으로 추천 후보를 반환합니다.",
    "recommendations": [],
    "warnings": ["OpenAI API error: ..."]
  }
}
```

## Error Cases

raw JSON 경로가 없을 때:

```json
{
  "detail": "Product raw data directory was not found: ..."
}
```
