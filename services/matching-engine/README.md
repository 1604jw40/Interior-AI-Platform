# Matching Engine RAG API

`matching-engine`은 FastAPI 기반 LLM/RAG 서버입니다. ai-perception 결과에서 감지된 객체의 x/y/z 공간을 계산하고, IKEA raw 상품 JSON에서 들어갈 수 있는 상품만 후보정한 뒤 OpenAI로 한국어 추천 응답을 생성합니다.

백엔드 연동 상세 문서는 [BACKEND_INTEGRATION_GUIDE.md](./BACKEND_INTEGRATION_GUIDE.md)를 참고하세요.

## Product Source

기본 상품 정보는 Postgres가 아니라 아래 raw JSON에서 읽습니다.

```text
infrastructure/product_db/data/raw/ikea_Beds_mattresses.json
infrastructure/product_db/data/raw/ikea_Desks_office_chairs.json
infrastructure/product_db/data/raw/ikea_Sofas_armchairs.json
infrastructure/product_db/data/raw/ikea_Storage_accessories.json
infrastructure/product_db/data/raw/ikea_Tables_chairs.json
```

이미지 URI는 `image_filename`을 기준으로 생성합니다.

```text
gs://interiorplatform-d58e0.firebasestorage.app/furniture_images/{raw_category}/{image_filename}
```

## Environment

- `OPENAI_API_KEY`: 필수. OpenAI API 호출에 사용합니다.
- `OPENAI_MODEL`: 선택. 기본값은 `gpt-5-mini`입니다.
- `OPENAI_EMBEDDING_MODEL`: 선택. 기본값은 `text-embedding-3-small`입니다.
- `OPENAI_VECTOR_STORE_ID`: 선택. 설정하면 OpenAI file search를 함께 사용할 수 있습니다.
- `FIREBASE_STORAGE_BASE_URI`: 선택. 기본값은 `gs://interiorplatform-d58e0.firebasestorage.app`입니다.
- `FIREBASE_STORAGE_PREFIX`: 선택. 기본값은 `furniture_images`입니다.
- `PRODUCT_RAW_DATA_DIR`: 선택. 기본값은 `infrastructure/product_db/data/raw`입니다.
- `EMBEDDING_CACHE_PATH`: 선택. 기본값은 `/tmp/matching_engine_embeddings.json`입니다.
- `PERCEPTION_SCALE_UNIT`: 선택. 기본값은 `m`이며, perception scale 값을 cm로 변환합니다.
- `MAX_RAG_CANDIDATES`: 선택. LLM에 전달할 상위 후보 수입니다.

## Endpoints

### `GET /health`

서버 설정, raw JSON 경로, 모델명, 임베딩 캐시 경로를 반환합니다.

### `POST /products/fit`

요청에 포함된 상품 목록을 기준으로 x/y/z 치수 검사만 수행합니다. 통과 상품은 `products`, 탈락 상품은 `rejected_products`에 사유와 함께 반환합니다.

### `POST /rag/fit-query`

요청에 포함된 상품 목록을 치수 검사한 뒤 임베딩 재정렬과 LLM 추천을 수행합니다.

### `POST /rag/perception-query`

백엔드 메인 연동 API입니다. `products`를 생략하면 `PRODUCT_RAW_DATA_DIR`의 IKEA raw JSON에서 상품 후보를 읽습니다.

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
      "confidence": 0.92
    }
  ]
}
```

응답의 `structured_answer`는 백엔드 파싱용 고정 JSON입니다. `answer`는 화면 표시용 요약 문자열입니다.

## 후보정

현재 RAG는 raw JSON 전체를 LLM에 바로 넘기지 않습니다.

1. 카테고리 매핑
2. x/y/z 치수 검사
3. fit/reject 사유 기록
4. fit/semantic/clearance/final 점수 계산
5. 임베딩 캐시 기반 재정렬
6. 상위 후보만 LLM 전달
7. OpenAI 실패 시 fallback JSON 반환
