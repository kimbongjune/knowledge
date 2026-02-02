"""
Document Assistant 설정
"""
import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
METADATA_PATH = DATA_DIR / "metadata"
MODELS_PATH = DATA_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"  # 업로드 파일 저장 디렉토리

# 디렉토리 생성
for path in [DATA_DIR, CHROMA_DB_PATH, METADATA_PATH, MODELS_PATH, UPLOAD_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# 문서 처리 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 지원 확장자
SUPPORTED_EXTENSIONS = {
    '.hwp', '.hwpx',     # 한글
    '.pdf',              # PDF
    '.docx', '.doc',     # Word
    '.txt', '.md',       # 텍스트
    '.csv', '.xlsx',     # 표 형식
}

# 임베딩 모델 설정
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"  # 한국어 특화

# LLM 설정
DEFAULT_MODEL_PARAMS = {
    "n_ctx": 4096,       # 컨텍스트 길이
    "n_gpu_layers": -1,  # 모든 레이어 GPU 사용
    "n_threads": 8,      # CPU 스레드 수
}

# 서버 설정
HOST = "127.0.0.1"
PORT = 7860
