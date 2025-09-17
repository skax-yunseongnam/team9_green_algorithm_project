
# Green Runner

Streamlit 기반의 Python 코드 실행/계측 + Azure OpenAI를 활용한 그린 최적화 도구이다.

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env  # 키/엔드포인트 채우기
streamlit run app.py
```

## 브랜치 가이드

- `main`: 안정화된 릴리스
- `develop`: 통합 테스트
- 기능별: `feat/*`
- 버그픽스: `fix/*`
- 리팩토링: `refactor/*`
- 문서: `docs/*`
