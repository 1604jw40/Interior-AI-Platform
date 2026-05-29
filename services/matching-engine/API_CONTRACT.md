# Matching Engine API Contract

이 문서는 백엔드/Unity/AI 서비스가 `matching-engine` LLM 서버와 붙을 때 사용하는 API 계약입니다.

## Base URL

Docker 네트워크 내부:

```text
http://matching-engine:8010
```

호스트 PC/Postman:

```text
http://localhost:8010
```

## Full Flow

권장 전체 흐름은 아래 순서입니다.

```text
1. ai-perception이 객체 감지
2. 감지 객체 label: class|ply_filename|scale_w|scale_h
3. 백엔드가 /rag/perception-query 호출
4. matching-engine이 상품 후보 로드
5. x/y/z 치수 필터링
6. text-embedding-3-small로 후보 유사도 정렬
7. 상위 후보만 RAG context로 정리
8. gpt-4.1-mini로 추천 답변 생성
9. 백엔드가 answer + fit.products를 Unity/UI에 전달
```

`scale_w`, `scale_h`는 현재 `ai-perception`의 `DepthEstimator` 기준 미터 단위입니다. 서버는 `PERCEPTION_SCALE_UNIT=m`일 때 cm로 변환합니다.

## Recommendation Policy

LLM은 아래 규칙으로 답변합니다.

- 응답 문장은 기본적으로 항상 한국어입니다.
- `fit.products` 안에 있는 상품만 추천합니다.
- 치수상 들어가지 않는 상품은 LLM에 전달하지 않습니다.
- 추천 우선순위는 `물리적 fit > category match > 남는 공간 > relevance_score > price`입니다.
- 상품명, 치수, 가격, URL, 이미지 경로는 제공된 context 밖에서 추측하지 않습니다.
- 응답에는 가능하면 추천 상품의 `id`, `name`, `dimensions`, `remaining_cm`, `category`, `storage_uri`, `product_url` 근거가 포함됩니다.
- RAG 후보는 기본적으로 상위 `MAX_RAG_CANDIDATES=5`개만 LLM에 전달합니다.

## Trace Policy

백엔드가 추천 결과를 추적할 수 있도록 각 상품 후보에는 `trace`가 포함됩니다.

```json
{
  "trace": {
    "source": "request_or_db",
    "product_id": 1,
    "product_name": "MICKE desk",
    "category": "desk",
    "category_code": "desk",
    "storage_path": "products/desk/micke.jpg",
    "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg",
    "product_url": "https://example.com/products/micke-desk",
    "dimensions_source": {
      "width_x": 105,
      "depth_y": 50,
      "height_z": 72
    }
  }
}
```

`/rag/perception-query`의 `retrieval`에도 아래 추적 필드가 포함됩니다.

```json
{
  "retrieval": {
    "mode": "perception_dimension_embedding_rag",
    "embedding_model": "text-embedding-3-small",
    "product_source": "db",
    "perception_category": "desk",
    "target_object_id": 0,
    "candidate_count": 3,
    "fit_count": 2,
    "max_rag_candidates": 5
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
  "model": "gpt-4.1-mini",
  "embedding_model": "text-embedding-3-small",
  "perception_scale_unit": "m"
}
```

## 2. Dimension Fit Only

LLM 호출 없이 상품이 공간에 들어가는지만 검사합니다.

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
      "id": 1,
      "product_name": "MICKE desk",
      "category_code": "desk",
      "price": 89900,
      "width_x": 105,
      "depth_y": 50,
      "height_z": 72,
      "storage_path": "products/desk/micke.jpg"
    }
  ]
}
```

Response:

```json
{
  "status": "success",
  "space": { "x": 118, "y": 58, "z": 73 },
  "clearance_cm": 2,
  "firebase_storage_base_uri": "gs://interiorplatform-d58e0.firebasestorage.app",
  "fit_count": 1,
  "products": [
    {
      "id": 1,
      "name": "MICKE desk",
      "dimensions": { "x": 105, "y": 50, "z": 72 },
      "fits": true,
      "orientation": "xyz",
        "remaining_cm": { "x": 13, "y": 8, "z": 1 },
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg",
        "price": 89900,
        "category": "desk",
        "product_url": "https://example.com/products/micke-desk",
        "trace": {
          "product_id": 1,
          "category_code": "desk",
          "storage_path": "products/desk/micke.jpg",
          "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg"
        },
        "relevance_score": null
      }
  ]
}
```

## 3. Fit + LLM RAG

백엔드가 상품 후보를 직접 넘기는 테스트/수동 모드입니다.

```http
POST /rag/fit-query
```

Request:

```json
{
  "query": "이 공간에 들어가는 책상 중 가장 적합한 상품을 추천해줘",
  "space": { "x": 120, "y": 60, "z": 75 },
  "clearance_cm": 2,
  "allow_xy_rotation": true,
  "products": [
    {
      "id": 1,
      "product_name": "MICKE desk",
      "category_code": "desk",
      "price": 89900,
      "width_x": 105,
      "depth_y": 50,
      "height_z": 72,
      "storage_path": "products/desk/micke.jpg"
    }
  ]
}
```

Response 주요 필드:

```json
{
  "status": "success",
  "answer": "LLM 추천 답변",
  "model": "gpt-4.1-mini",
  "retrieval": { "mode": "provided_context" },
  "fit": {
    "fit_count": 1,
    "products": [
      {
        "id": 1,
        "name": "MICKE desk",
        "relevance_score": 0.18,
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg"
      }
    ]
  },
  "perception": null
}
```

## 4. Perception + Product DB + LLM RAG

전체 백엔드 연동용 메인 API입니다.

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

위 요청에서 `products`를 생략하면 `matching-engine`이 DB에서 상품 후보를 조회합니다.

Request with explicit products:

```json
{
  "query": "감지된 객체 자리에 들어갈 수 있는 상품을 추천해줘",
  "target_object_id": 0,
  "clearance_cm": 2,
  "perception_objects": [
    {
      "id": 0,
      "label": "desk|asset_desk_0.ply|1.20|0.75",
      "confidence": 0.92
    }
  ],
  "products": [
    {
      "id": 1,
      "product_name": "MICKE desk",
      "category_code": "desk",
      "price": 89900,
      "width_x": 105,
      "depth_y": 50,
      "height_z": 72,
      "storage_path": "products/desk/micke.jpg"
    }
  ]
}
```

Response 주요 필드:

```json
{
  "status": "success",
  "answer": "LLM 추천 답변",
  "model": "gpt-4.1-mini",
  "retrieval": {
    "mode": "perception_dimension_embedding_rag",
    "embedding_model": "text-embedding-3-small",
    "product_source": "db",
    "perception_category": "desk",
    "target_object_id": 0,
    "candidate_count": 3,
    "fit_count": 1,
    "max_rag_candidates": 5
  },
  "fit": {
    "space": { "x": 120, "y": 120, "z": 75 },
    "fit_count": 1,
    "products": [
      {
        "id": 1,
        "name": "MICKE desk",
        "remaining_cm": { "x": 13, "y": 68, "z": 1 },
        "relevance_score": 0.18,
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg",
        "trace": {
          "product_id": 1,
          "category_code": "desk",
          "storage_path": "products/desk/micke.jpg",
          "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg",
          "product_url": "https://example.com/products/micke-desk"
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

## Product Candidate Schema

상품 후보는 두 가지 형식을 모두 지원합니다.

DB row style:

```json
{
  "id": 1,
  "product_name": "MICKE desk",
  "category_code": "desk",
  "price": 89900,
  "width_x": 105,
  "depth_y": 50,
  "height_z": 72,
  "storage_path": "products/desk/micke.jpg",
  "product_url": "https://www.ikea.com/..."
}
```

Normalized style:

```json
{
  "id": 1,
  "name": "MICKE desk",
  "category": "desk",
  "price": 89900,
  "dimensions": { "x": 105, "y": 50, "z": 72 },
  "storage_path": "products/desk/micke.jpg"
}
```

## Error/Empty Cases

OpenAI key missing:

```json
{
  "detail": "OPENAI_API_KEY is not configured for matching-engine."
}
```

No fitting products:

```json
{
  "status": "success",
  "answer": "No product candidates fit the detected object's available space.",
  "retrieval": {
    "mode": "no_fit_candidates"
  },
  "fit": {
    "fit_count": 0,
    "products": []
  }
}
```

## Backend Integration Notes

- `OPENAI_VECTOR_STORE_ID`는 현재 비워둬도 됩니다.
- JPEG만 있어도 `storage_path`에 저장하면 추천 결과의 대표 이미지 URI로 반환됩니다.
- Unity에서 실제 3D 모델 배치가 필요하면 추후 `image_path`, `model_path`를 분리하는 것을 권장합니다.
- `ai-perception`이 depth를 추정하지 않기 때문에 perception label만 쓰면 `x=width`, `y=width`, `z=height`로 계산합니다.
- 더 정확한 배치를 원하면 백엔드가 명시적으로 `space: { x, y, z }`를 넣어 호출하세요.
