# Matching Engine Backend Integration Guide

아래 내용은 백엔드에서 LLM/RAG 서버를 호출하기 위한 최소 연동 가이드입니다.

## 1. 서버 역할

`matching-engine`은 별도 FastAPI LLM 서버입니다.

백엔드는 `ai-perception` 결과를 `matching-engine`에 HTTP POST로 전달하면 됩니다.

`matching-engine` 내부 처리 흐름:

```text
1. ai-perception 객체 label 파싱
2. Postgres DB에서 상품 후보 조회
3. x/y/z 치수 검사
4. text-embedding-3-small로 후보 유사도 정렬
5. gpt-4.1-mini로 한국어 추천 답변 생성
6. 추천 결과와 추적 정보 JSON 반환
```

## 2. 실행 전제

최소 실행 컨테이너:

```text
postgres_db
matching-engine
```

실행 명령:

```bash
docker compose up -d postgres_db matching-engine
```

Health check:

```http
GET http://localhost:8010/health
```

Docker Compose 내부에서 호출할 때:

```text
http://matching-engine:8010
```

호스트 PC/Postman에서 호출할 때:

```text
http://localhost:8010
```

## 3. 메인 API

```http
POST /rag/perception-query
Content-Type: application/json
```

Docker 내부 URL:

```text
http://matching-engine:8010/rag/perception-query
```

로컬 테스트 URL:

```text
http://localhost:8010/rag/perception-query
```

## 4. 입력 JSON

백엔드는 아래 JSON 형태로 호출하면 됩니다.

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

### 입력 필드 설명

```text
query
  LLM에게 전달할 사용자/백엔드 질의입니다.

target_object_id
  추천 기준으로 사용할 perception object id입니다.
  null이면 confidence가 가장 높은 객체를 사용합니다.

clearance_cm
  여유 공간입니다.
  예: 2면 x/y/z에서 각각 2cm를 뺀 공간에 들어가는지 검사합니다.

allow_xy_rotation
  true면 상품의 x/y를 바꿔서 들어가는지도 검사합니다.

limit
  DB에서 조회할 상품 후보 최대 개수입니다.

perception_objects
  ai-perception이 감지한 객체 배열입니다.

perception_objects[].label
  형식: class|ply_filename|scale_w|scale_h
  예: desk|asset_desk_0.ply|1.20|0.75
  현재 scale_w, scale_h는 미터 단위이며 matching-engine 내부에서 cm로 변환합니다.
```

## 5. 출력 JSON

응답은 아래 구조입니다.

```json
{
  "status": "success",
  "answer": "가장 적합한 상품은 MICKE desk입니다. 상품 id는 1이며, 크기는 105x50x72cm이고 남은 공간은 13x68x1cm입니다. storage_uri는 gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg 입니다.",
  "model": "gpt-4.1-mini",
  "retrieval": {
    "mode": "perception_dimension_embedding_rag",
    "embedding_model": "text-embedding-3-small",
    "product_source": "db",
    "perception_category": "desk",
    "target_object_id": 0,
    "candidate_count": 3,
    "fit_count": 2,
    "max_rag_candidates": 5
  },
  "fit": {
    "status": "success",
    "space": {
      "x": 118,
      "y": 118,
      "z": 73
    },
    "clearance_cm": 2,
    "firebase_storage_base_uri": "gs://interiorplatform-d58e0.firebasestorage.app",
    "fit_count": 2,
    "products": [
      {
        "id": 1,
        "name": "MICKE desk",
        "dimensions": {
          "x": 105,
          "y": 50,
          "z": 72
        },
        "fits": true,
        "orientation": "xyz",
        "remaining_cm": {
          "x": 13,
          "y": 68,
          "z": 1
        },
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.jpg",
        "price": 89900,
        "category": "desk",
        "product_url": "https://example.com/products/micke-desk",
        "relevance_score": 0.309186,
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
        },
        "metadata": {
          "image_description": "Compact white desk suitable for small rooms."
        }
      }
    ]
  },
  "perception": {
    "id": 0,
    "category": "desk",
    "asset_filename": "asset_desk_0.ply",
    "estimated_width": 1.2,
    "estimated_height": 0.75,
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
}
```

## 6. 출력 필드 설명

```text
answer
  LLM이 생성한 한국어 추천 답변입니다.

retrieval
  RAG/DB/embedding 처리 추적 정보입니다.

fit.space
  clearance_cm이 반영된 유효 공간입니다.
  perception label의 scale 값은 cm로 변환됩니다.

fit.products
  실제로 치수상 들어가는 상품 후보입니다.
  LLM은 이 배열 안의 상품만 추천합니다.

fit.products[].trace
  백엔드 추적용 정보입니다.
  DB product id, category_code, storage_path, storage_uri, product_url을 확인할 수 있습니다.

perception
  추천 기준으로 사용된 감지 객체 정보입니다.
```

## 7. 백엔드에서 사용하는 핵심 값

UI/Unity에 넘기기 좋은 필드:

```text
answer
fit.products[0].id
fit.products[0].name
fit.products[0].dimensions
fit.products[0].remaining_cm
fit.products[0].storage_uri
fit.products[0].product_url
fit.products[0].trace.category_code
fit.products[0].trace.storage_path
```

## 8. 에러/빈 결과

치수에 맞는 상품이 없는 경우:

```json
{
  "status": "success",
  "answer": "No product candidates fit the detected object's available space.",
  "retrieval": {
    "mode": "no_fit_candidates",
    "perception_category": "desk"
  },
  "fit": {
    "fit_count": 0,
    "products": []
  }
}
```

OpenAI API key가 없는 경우:

```json
{
  "detail": "OPENAI_API_KEY is not configured for matching-engine."
}
```

DB 조회 실패:

```json
{
  "detail": "Product DB query failed: ..."
}
```

## 9. 참고

- `OPENAI_VECTOR_STORE_ID`는 현재 비워둬도 됩니다.
- 상품 데이터는 Postgres의 `products`, `categories`, `product_images` 테이블에서 조회합니다.
- `product_images.image_filename`은 Firebase Storage 내부 경로로 사용됩니다.
- 예: `products/desk/micke.jpg`
- 응답에서는 `FIREBASE_STORAGE_BASE_URI`와 합쳐져 `storage_uri`로 반환됩니다.
