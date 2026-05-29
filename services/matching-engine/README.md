# Matching Engine RAG API

`matching-engine`는 상품 치수 필터링과 OpenAI RAG 응답을 담당하는 백엔드 HTTP API입니다.

백엔드 연동 계약은 [API_CONTRACT.md](./API_CONTRACT.md)를 기준으로 보면 됩니다.
팀 공유용 짧은 연동 가이드는 [BACKEND_INTEGRATION_GUIDE.md](./BACKEND_INTEGRATION_GUIDE.md)를 참고하세요.

## Environment

- `OPENAI_API_KEY`: 필수. OpenAI API 호출에 사용합니다.
- `OPENAI_MODEL`: 선택. 기본값은 `gpt-4.1-mini`입니다.
- `OPENAI_EMBEDDING_MODEL`: 선택. 기본값은 `text-embedding-3-small`입니다.
- `OPENAI_VECTOR_STORE_ID`: 선택. 설정하면 `/rag/query`, `/rag/fit-query`에서 OpenAI file search RAG를 사용합니다.
- `FIREBASE_STORAGE_BASE_URI`: 선택. 기본값은 `gs://interiorplatform-d58e0.firebasestorage.app`입니다.
- `PERCEPTION_SCALE_UNIT`: 선택. `ai-perception`의 `scale_w`, `scale_h` 단위입니다. 기본값은 `m`이며 내부에서 cm로 변환합니다.
- `MAX_RAG_CANDIDATES`: 선택. LLM에 전달할 상위 후보 개수입니다. 기본값은 `5`입니다.
- `DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME`, `DB_PORT`: `/rag/perception-query`에서 `products`를 생략하면 상품 스키마에서 후보를 읽어옵니다.

## Endpoints

### `GET /health`

서비스 설정 상태를 반환합니다.

### `POST /products/fit`

`space.x/y/z`에 들어갈 수 있는 상품만 필터링합니다. 단위는 현재 상품 DB 스키마에 맞춰 `cm` 기준입니다.

Request:

```json
{
  "space": { "x": 120, "y": 60, "z": 75 },
  "clearance_cm": 2,
  "allow_xy_rotation": true,
  "products": [
    {
      "id": 1,
      "name": "MICKE desk",
      "category": "desk",
      "price": 89900,
      "dimensions": { "x": 105, "y": 50, "z": 72 },
      "storage_path": "products/desk/micke.glb",
      "product_url": "https://www.ikea.com/..."
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
      "relevance_score": null,
      "storage_uri": "gs://interiorplatform-d58e0.firebasestorage.app/products/desk/micke.glb"
    }
  ]
}
```

### `POST /rag/query`

일반 RAG 질의입니다. `OPENAI_VECTOR_STORE_ID`가 있으면 OpenAI file search를 사용하고, 없으면 요청의 `context`와 `fit_candidates`를 근거로 답변합니다.

### `POST /rag/fit-query`

치수 검사를 먼저 수행한 뒤, 들어갈 수 있는 상품 후보만 LLM에 전달해 설명을 생성합니다.
통과한 상품 후보는 `OPENAI_EMBEDDING_MODEL`로 질의와의 의미 유사도를 계산해 `relevance_score` 높은 순으로 정렬됩니다.

Request:

```json
{
  "query": "이 공간에 들어가는 책상 중 가장 적합한 상품을 골라줘",
  "space": { "x": 120, "y": 60, "z": 75 },
  "clearance_cm": 2,
  "allow_xy_rotation": true,
  "room_context": {
    "style": "minimal",
    "wall_color": "white"
  },
  "products": [
    {
      "id": 1,
      "name": "MICKE desk",
      "category": "desk",
      "price": 89900,
      "dimensions": { "x": 105, "y": 50, "z": 72 },
      "storage_path": "products/desk/micke.glb"
    }
  ]
}
```

Local test:

```bash
curl -X POST http://localhost:8010/rag/fit-query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"이 공간에 들어가는 책상 추천해줘\",\"space\":{\"x\":120,\"y\":60,\"z\":75},\"products\":[{\"id\":1,\"name\":\"MICKE desk\",\"dimensions\":{\"x\":105,\"y\":50,\"z\":72},\"storage_path\":\"products/desk/micke.glb\"}]}"
```

### `POST /rag/perception-query`

`ai-perception`의 `DetectedObject.label` 형식인 `class|ply_filename|scale_w|scale_h`를 파싱해서 RAG를 수행합니다.
현재 `scale_w`, `scale_h`는 `DepthEstimator` 기준 미터 단위이므로 `PERCEPTION_SCALE_UNIT=m`일 때 cm로 변환됩니다.
상품은 기존 DB 스키마의 `product_name`, `width_x`, `depth_y`, `height_z`, `category_code` 필드 그대로 넣을 수 있습니다.
`products`를 생략하면 `products`, `categories`, `product_images` 테이블에서 후보를 조회합니다.

Request:

```json
{
  "query": "감지된 객체 자리에 들어갈 수 있는 상품을 추천해줘",
  "target_object_id": 0,
  "clearance_cm": 2,
  "limit": 100,
  "perception_objects": [
    {
      "id": 0,
      "label": "desk|asset_desk_0.ply|1.20|0.75",
      "confidence": 0.92,
      "bbox": { "x_min": 100, "y_min": 120, "x_max": 420, "y_max": 520 },
      "position_3d": { "x": 0.1, "y": 0.0, "z": 2.1 }
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
      "storage_path": "products/desk/micke.glb",
      "product_url": "https://www.ikea.com/..."
    }
  ]
}
```

Response에는 선택된 perception 객체, 치수 필터 결과, 임베딩 유사도 정렬 결과, 그리고 `gpt-4.1-mini` 답변이 함께 포함됩니다.
