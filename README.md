# Market Ripple MVP v0.1

특정 국내 상장기업(기본: SK하이닉스 000660)을 중심으로 OpenDART 공시를 수집·선별·임베딩하고, 관련 기업을 자동으로 발견해 RAG 범위를 넓힌다. 그런 다음 입력 이벤트의 직접·간접 산업 파급효과를 **원문 근거로 검증된 경로**로 탐색하는 리서치 파이프라인이다.

> AI가 가설을 발견하고, 공시 원문과 실제 금융 데이터가 그 가설을 검증한다.

## 빠른 시작

```bash
# 0) 사전 준비: Python 3.11, uv, Node.js 18+ (Korea Stock MCP용)
cp .env.example .env        # OPENAI_API_KEY, DART_API_KEY, KRX_API_KEY 입력
uv sync

# 1) 환경 점검 (+ Korea Stock MCP 설치)
uv run python -m scripts.check_environment --install-mcp
uv run python -m scripts.check_krx        # KRX 인증키·서비스별 이용 승인 상태 진단

# 2) (권장) 실제 사업보고서 XML 구조 확인 → 섹션 선택 결과 점검
uv run python -m scripts.inspect_xml --latest-annual 000660

# 3) Seed + 관련 기업 데이터 구축
uv run python -m scripts.bootstrap --seed 000660 --max-companies 20 --max-hops 2

# 4) 분석
uv run python -m scripts.run_research --seed 000660 \
    --query "HBM 수요가 증가할 경우 국내 관련 기업과 산업의 파급효과를 분석해줘" --financial --open
# --open: 분석이 끝나면 Streamlit을 띄워 해당 결과(?run=<run_id>)를 바로 표시

# 5) 과거 시나리오 평가 (A/B/C 비교)
uv run python -m scripts.run_evaluation --event-file data/evaluation/events.json

# 6) UI / 테스트
uv run streamlit run app.py
uv run pytest              # 단위 테스트 (외부 API 불필요)
uv run pytest -m live      # 실제 MCP 서버 연결 테스트
```

> 임베딩 저장 방식과 분석 시 RAG·LLM 호출 흐름은 [docs/PIPELINE.md](docs/PIPELINE.md)에 자세히 정리되어 있다.

## 구조

```text
config/      settings.py(환경변수), sections.yaml(섹션 선택 규칙)
src/
  ingestion/  dart_client · corp_registry · collector · xml_parser · section_selector · chunker
  retrieval/  embedding(캐시) · vector_store(FAISS IndexIDMap2+FlatIP) · retriever(Vector+Keyword RRF)
  discovery/  company_discovery(문서 근거 + 가설 기반) · entity_resolution(상장사 식별)
  graph/      graph_store(NetworkX MultiDiGraph) · traversal(경로) · d3_view(d3-force 시각화)
  research/   event_extractor · query_generator · relationship_extractor · validator · ripple_engine
  financial/  mcp_client(Korea Stock MCP) · market_data(Provider) · metrics(이벤트 스터디) · analysis
  evaluation/ baseline(A/B/C) · historical_replay · evaluator
  pipeline.py 구성요소 조립 · 단계별 재실행 함수
prompts/     LLM 프롬프트 (공시 텍스트는 지시가 아닌 데이터로 취급)
scripts/     CLI
docs/korea_stock_mcp_1.4.1_tools.json   실제 list_tools 결과
```

## 핵심 설계

| 요구사항 | 구현 |
|---|---|
| 기업 수 | 기본 20개 (`MAX_COMPANIES`), 탐색 노드당 후보 5개 (`MAX_DISCOVERY_PER_NODE`). 상한을 올려 다시 실행하면 이미 수집·임베딩한 기업은 재사용하고 새 기업만 추가 |
| 그래프 시각화 | d3-force Disjoint force-directed graph (forceX/forceY, 드래그·줌·연결 강조). Pyvis 대신 사용. 브라우저에서 cdn.jsdelivr.net 접근 필요 |
| 하드코딩 금지 | Seed는 `SEED_STOCK_CODE`, 관련 기업·경로는 탐색 결과로만 생성 |
| 섹션 선택 | 태그 경로를 가정하지 않고 `TITLE` 자식을 가진 컨테이너를 섹션으로 인식, 제목 정규화(로마숫자·번호·공백) 후 YAML 규칙 매칭. 실패 시 제목 목록 LLM 분류, 문서 전체 임베딩 금지 |
| 표 | 제목·단위·열 이름 유지 Markdown, 큰 표 분할 시 헤더/단위 반복 |
| 중복 방지 | 원본 SHA-256, Chunk 정규화 해시, (기업, 본문해시) 기준 임베딩 재사용, 출처는 문서별 Chunk로 모두 유지 |
| 정정공시 | `version_group`으로 버전 관리, 기준일 시점 최신 버전만 검색 |
| 미래 문서 차단 | 기준일 이하 벡터만으로 **임시 FAISS 인덱스**를 만들어 검색 (사후 필터링 아님) |
| 검증 | Structural → Quote(원문 존재) → Semantic(LLM) 분리. evidence 없는 VERIFIED 상태는 Pydantic이 거부. 간접 관계를 SUPPLIES_TO로 확정하지 않음 |
| 경로 상태 | 가장 약한 Edge 상태. 검증 실패 경로는 확장 중단 |
| 예산 | Hop·분기·LLM·검색 호출 수 상한, 동일 질문 캐시, 순환 감지 |
| 금융 | 지표는 Python 계산. 시장조정 CAR과 시장모형 CAR 구분, 데이터 없으면 미계산 + 상태 표시 |
| 평가 | 예측을 먼저 저장한 뒤 사후 데이터 비교, 정답 라벨은 사람이 작성 |

## Korea Stock MCP

`korea-stock-mcp@1.4.1`에서 실제 `list_tools`로 확인한 Tool은 다음 8개다 (`docs/korea_stock_mcp_1.4.1_tools.json`).
`get_corp_code`, `get_disclosure_list`, `get_disclosure`, `get_financial_statement`, `get_market_type`, `get_stock_base_info`, `get_stock_trade_info`, `get_today_date`

- 주가는 `get_stock_trade_info(basDdList, market, codeList)`로 조회한다. 기준일 1개당 KRX API를 1번 호출하므로 일자별로 로컬 캐시한다. 반환값은 **수정주가가 아닌** 일별 종가다.
- MCP에는 지수(벤치마크) Tool이 없다. 그래서 `CsvBenchmarkProvider`(`data/market/benchmarks/KOSPI.csv`, `KOSDAQ.csv`: `date,close`)를 먼저 쓰고, 없으면 `KrxIndexBenchmarkProvider`(KRX Open API 지수, **미검증**)를 쓴다. 둘 다 실패하면 시장 대비 지표를 계산하지 않는다.
- 패키지 bin의 shebang(`#!/usr/bin/env node --max-old-space-size=4096`)은 Linux에서 실행되지 않는다. 그래서 기본 설정은 `.mcp/`에 설치한 패키지를 `node`로 직접 실행한다.
- 서버 프로세스에는 필요한 환경변수(DART/KRX Key, PATH, 프록시)만 넘긴다.
- LLM Function Calling은 `run_tool_agent`로 연결되어 있다. 허용 Tool, 날짜 범위, 호출 횟수는 Python에서 통제한다.

## 검증 현황 (정직한 목록)

개발 환경에서 OpenDART, OpenAI, KRX에 접속할 수 없었고 API Key도 없었다. 그래서 아래 항목은 상태를 나눠 적는다.

**실제로 확인한 것**
- 단위·통합 테스트 63개 통과 (Mock: OpenDART HTTP, 임베딩, LLM / 실제: FAISS, SQLite, lxml, 지표 계산)
- Korea Stock MCP 1.4.1 실제 서버 기동, MCP SDK로 initialize와 list_tools 실행, Tool 스키마 확인, `get_today_date` 호출 성공, Key 없을 때 오류가 명시적으로 표시되는지 확인
- Streamlit 앱 headless 기동 (5개 탭, Key 미설정 경고 표시)
- d3-force 그래프: jsdom에서 실제 d3 7.9.0으로 렌더링해 시뮬레이션 실행, 분리 컴포넌트 배치, 다중 관계 곡선, 한글 라벨 표시 확인

**검증되지 않은 것 (실제 Key로 실행 필요)**
- OpenDART 실제 응답(corpCode, list, document ZIP)과 **실제 SK하이닉스 사업보고서 XML에 대한 섹션 선택 품질**. 테스트 XML은 공개된 DART 구조를 모사한 가상 문서다. 먼저 `scripts.inspect_xml`로 확인하고 `config/sections.yaml`을 조정할 것
- OpenAI Responses API Structured Outputs 실제 호출, 모델 사용 가능 여부 점검, 임베딩 API
- 프롬프트 품질(관계 추출, 의미 검증, 기업 발견), 실제 관련 기업 발견 결과
- KRX 주가 조회(`get_stock_trade_info`)와 KRX 지수 API(`KrxIndexBenchmarkProvider`의 엔드포인트·필드명)
- 3개 과거 이벤트 평가 실행. `events.json`의 ground_truth는 비어 있고, 사람이 공시 원문을 보고 작성해야 한다
- 검색 성능 비교(NumPy / FlatIP / HNSW)는 선택 항목이라 구현하지 않았다

## 알려진 한계

- 주요사항·거래소 공시의 정정본은 원공시와 자동으로 연결하지 않는다 (접수번호 단위로 관리).
- 공시 ZIP 안의 첨부 문서(감사보고서 등)는 본문 분석에서 제외하고 로그에 남긴다.
- 유사도 점수와 검증 상태는 투자 성공 확률이 아니며, 종합 투자 점수는 만들지 않는다.
- 과거 재현 결과는 LLM 사전학습 데이터 누출 가능성이 있어 순수한 미래 예측 성능으로 볼 수 없다.
