# Document Assistant

LangChain 기반 RAG 문서 질의응답 시스템

## 설치

```bash
# Anaconda 가상환경 생성
conda create -n docassist python=3.11 -y
conda activate docassist

# 패키지 설치
pip install -r requirements.txt

# GPU 가속용 llama-cpp-python 설치 (선택)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

## 실행

```bash
python main.py
```

## 지원 파일 형식

- HWP, HWPX (한글)
- PDF
- DOCX, DOC (Word)
- TXT, MD
- CSV, XLSX

## 모델 사용

`data/models/` 폴더에 GGUF 형식 모델 파일을 넣으면 자동으로 인식됩니다.

추천 모델 (RTX 3060 12GB):
- Qwen2.5-14B-Instruct-Q4_K_M.gguf
- Mistral-7B-Instruct-Q4_K_M.gguf
