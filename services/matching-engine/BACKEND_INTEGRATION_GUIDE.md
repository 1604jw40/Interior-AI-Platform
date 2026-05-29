# Matching Engine Backend Integration Guide

이 문서는 백엔드에서 LLM/RAG 서버를 호출하기 위한 최소 연동 가이드입니다.

## 서버 역할

`matching-engine`은 별도 FastAPI 서버입니다.

처리 흐름:

```text
1. 백엔드가 ai-perception 결과를 /rag/perception-query로 전달
2. matching-engine이 perception label에서 class, scale_w, scale_h 파싱
3. infrastructure/product_db/data/raw/*.json에서 IKEA 상품 후보 로드
4. x/y/z 치수 검사
5. text-embedding-3-small로 후보 정렬
6. gpt-4.1-mini로 한국어 추천 답변 생성
7. 추천 상품, Firebase Storage 경로, trace 정보를 JSON으로 반환
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

ngrok 사용 시:

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

Firebase 이미지 경로는 raw JSON의 `image_filename`으로 생성합니다.

```text
gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/{raw_category}/{image_filename}
```

응답에서 추적할 필드:

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
```

## Response

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
    "status": "success",
    "space": { "x": 118, "y": 118, "z": 73 },
    "clearance_cm": 2,
    "firebase_storage_base_uri": "gs://interiorplatform-d58e0.firebasestorage.app",
    "fit_count": 5,
    "products": [
      {
        "id": "e4d6f1a8c912",
        "name": "VATTENKAR Desktop shelf - white 49x15 cm",
        "dimensions": { "x": 49, "y": 15, "z": 36 },
        "fits": true,
        "orientation": "xyz",
        "remaining_cm": { "x": 69, "y": 103, "z": 37 },
        "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/Storage_accessories/Storage accessories_00001.jpg",
        "price": 19900,
        "category": "Storage accessories",
        "product_url": "https://www.ikea.com/kr/en/p/vattenkar-desktop-shelf-white-00541569/",
        "trace": {
          "source": "request_or_db",
          "product_id": "e4d6f1a8c912",
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
