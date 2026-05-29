# Matching Engine RAG API

`matching-engine`는 FastAPI 기반 LLM/RAG 서버입니다. ai-perception 결과에서 감지된 객체의 x/y/z 공간을 계산하고, IKEA raw 상품 JSON에서 들어갈 수 있는 상품만 골라 OpenAI로 한국어 추천 답변을 생성합니다.

백엔드 연동용 상세 문서는 [BACKEND_INTEGRATION_GUIDE.md](./BACKEND_INTEGRATION_GUIDE.md)를 참고하세요.

## Product Source

기본 상품 후보는 Postgres가 아니라 아래 raw JSON 파일에서 읽습니다.

```text
infrastructure/product_db/data/raw/ikea_Beds_mattresses.json
infrastructure/product_db/data/raw/ikea_Desks_office_chairs.json
infrastructure/product_db/data/raw/ikea_Sofas_armchairs.json
infrastructure/product_db/data/raw/ikea_Storage_accessories.json
infrastructure/product_db/data/raw/ikea_Tables_chairs.json
```

각 상품은 `product_name`, `price`, `width_x`, `depth_y`, `height_z`, `top_category_code`, `product_url`, `image_filename`을 사용합니다.

이미지 URI는 raw JSON의 `image_filename`을 기준으로 아래 형식으로 생성합니다.

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
- `PERCEPTION_SCALE_UNIT`: 선택. 기본값은 `m`이며, perception scale 값을 cm로 변환합니다.
- `MAX_RAG_CANDIDATES`: 선택. LLM에 전달할 상위 후보 수입니다.

## Endpoints

### `GET /health`

서버 설정 상태와 raw JSON 경로를 반환합니다.

### `POST /products/fit`

요청에 포함된 상품 목록을 기준으로 x/y/z 치수 필터링만 수행합니다.

### `POST /rag/fit-query`

요청에 포함된 상품 목록을 치수 필터링한 뒤 OpenAI 답변을 생성합니다.

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

응답의 `retrieval.product_source`가 `raw_json`이면 raw 상품 JSON에서 후보를 읽은 것입니다.
