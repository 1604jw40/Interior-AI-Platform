# 상품 DB 개발 구성

상품 크기·가격·이미지 정보를 저장하기 위한 별도 MySQL 개발 구성입니다. 루트 PostgreSQL 서비스와 다른 구성입니다.

## 파일

- `docker-compose.yml`: MySQL 8, 호스트 3308번 포트, 이름 있는 데이터 볼륨
- `.env.example`: 로컬 연결 변수의 빈 예시
- `db/schema.sql`: 카테고리·상품·이미지 테이블 정의
- `crawler/ikea_to_json.py`: IKEA 상품 페이지와 이미지를 수집해 로컬 JSON으로 저장하는 코드
- `db/insert_from_json.py`: 현재 빈 파일. JSON→DB 적재 기능은 구현되어 있지 않음

## 비밀정보와 실행

이 폴더에서 `.env.example`을 `.env`로 복사하고 `DB_PASSWORD`를 로컬에서 입력합니다. Compose는 이 값을 MySQL 초기 root 비밀번호로 사용하며 값이 없거나 비어 있으면 오류를 반환합니다.

```bash
docker compose config --quiet
docker compose up -d mysql
```

위 명령은 구성 문법 확인과 DB 시작 단계입니다. 이번 정비에서는 Docker 실행·DB 연결을 테스트하지 않았습니다. 최초 빈 볼륨을 만들 때만 초기 비밀번호가 적용되며, 기존 볼륨의 계정 비밀번호는 환경변수를 바꿔도 자동 변경되지 않습니다.

## 데이터 보존과 구현 범위

- `db/schema.sql`은 기존 테이블을 `DROP TABLE`하므로 운영 DB나 보존할 데이터가 있는 DB에 그대로 실행하지 않습니다. Compose가 이 SQL을 자동 적용하도록 구성되어 있지도 않습니다.
- 일반 종료는 `docker compose stop`을 사용합니다. `down -v`는 데이터 볼륨을 삭제하므로 초기화 의도가 없으면 사용하지 않습니다.
- 크롤러 실행은 외부 사이트에 접속하고 이미지·JSON·로그를 저장합니다. 사이트 이용 조건과 이미지 재배포 권한을 확인한 후 실행해야 합니다. 문서 정비에서 크롤링은 수행하지 않았습니다.
- 수집 데이터의 DB 적재와 전체 서비스 연동은 검증된 완료 기능으로 보지 않습니다.
- 키·비밀번호·`.env`·서비스 계정 파일과 수집 원본은 커밋하지 않습니다.
