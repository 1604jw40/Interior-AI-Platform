# Matching Engine Backend Integration Guide

백엔드에서 LLM/RAG 서버를 호출하기 위한 최소 연동 가이드입니다.

## 서버 역할

`matching-engine`은 별도 FastAPI 서버입니다.

처리 흐름:

```text
1. 백엔드가 ai-perception 결과를 /rag/perception-query로 전달
2. matching-engine이 perception label에서 class, scale_w, scale_h 파싱
3. infrastructure/product_db/data/raw/*.json에서 IKEA 상품 정보 로딩
4. 카테고리 매핑과 x/y/z 치수 검사
5. 통과/탈락 사유 및 점수 기록
6. text-embedding-3-small로 후보 재정렬
7. gpt-5-mini로 한국어 추천 JSON 생성
8. 추천 상품, Firebase Storage 경로, trace 정보를 JSON으로 반환
```

## Base URL

Docker 네트워크 내부:

```text
http://matching-engine:8010
```

로컬 테스트:

```text
http://localhost:8010
```

ngrok 사용:

```text
https://{ngrok-domain}/rag/perception-query
```

## Main API

```http
POST /rag/perception-query
Content-Type: application/json
```

## Request

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
      "bbox": {
        "x_min": 100,
        "y_min": 120,
        "x_max": 420,
        "y_max": 520
      },
      "position_3d": {
        "x": 0.1,
        "y": 0.0,
        "z": 2.1
      }
    }
  ]
}
```

`label` 형식:

```text
class|ply_filename|scale_w|scale_h
```

`scale_w`, `scale_h`는 기본적으로 meter 단위로 보고 cm로 변환합니다. 명시 공간을 직접 주고 싶으면 request에 `space`를 넣으면 됩니다.

```json
"space": { "x": 120, "y": 60, "z": 75 }
```

## Product Source

백엔드가 `products`를 보내지 않으면 서버가 자동으로 raw JSON에서 상품을 읽습니다.

```text
infrastructure/product_db/data/raw/*.json
```

Firebase 이미지 경로:

```text
gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/{raw_category}/{image_filename}
```

## Response

백엔드는 `structured_answer.recommendations`를 우선 사용하면 됩니다. `answer`는 화면 표시용 요약입니다.

```json
{
  "status": "success",
  "answer": "가장 적합한 상품은 PERJOHAN Stool with storage - pine입니다.",
  "structured_answer": {
    "summary": "가장 적합한 상품은 PERJOHAN Stool with storage - pine입니다.",
    "recommendations": [
      {
        "product_id": "4026381ff131",
        "product_name": "PERJOHAN Stool with storage - pine",
        "rank": 1,
        "reason": "감지된 책상 주변 공간에 들어가며 남는 공간이 충분합니다.",
        "fit_reason": "x/y/z 기준 통과: orientation=xyz, 상품 크기 49.0x28.0x45.0cm, 남은 공간 69.0x90.0x33.0cm",
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
    "status": "success",
    "space": { "x": 118, "y": 118, "z": 73 },
    "clearance_cm": 2,
    "fit_count": 5,
    "rejected_count": 95,
    "products": [
      {
        "id": "4026381ff131",
        "name": "PERJOHAN Stool with storage - pine",
        "fits": true,
        "orientation": "xyz",
        "fit_reason": "x/y/z 기준 통과: orientation=xyz, ...",
        "fit_score": 0.388535,
        "semantic_score": 0.8123,
        "clearance_score": 0.590178,
        "intent_score": 0.85,
        "size_score": 0.440741,
        "final_score": 0.662,
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/Desks_office_chairs/Desks office chairs_00002.jpg",
        "trace": {
          "source": "raw_json",
          "product_id": "4026381ff131",
          "category_code": "desks_office_chairs",
          "storage_path": "furniture_images/Desks_office_chairs/Desks office chairs_00002.jpg"
        },
        "metadata": {
          "source": "raw_json",
          "raw_category_label": "Desks_office_chairs",
          "image_filename": "Desks office chairs_00002.jpg"
        }
      }
    ],
    "rejected_products": [
      {
        "id": "741a93ca0014",
        "fits": false,
        "reject_reason": "공간 부족: orientation=xyz, z축 10.0cm 초과"
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

## 백엔드 추적용 필드

```text
structured_answer.recommendations[].product_id
fit.products[].id
fit.products[].name
fit.products[].category
fit.products[].product_url
fit.products[].storage_uri
fit.products[].trace.storage_path
fit.products[].trace.category_code
fit.products[].metadata.raw_category_label
fit.products[].metadata.image_filename
```

## Fallback

OpenAI 호출 실패 시에도 치수 기반 후보는 반환됩니다.

```json
{
  "status": "degraded",
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
