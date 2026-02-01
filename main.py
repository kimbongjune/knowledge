"""
Document Assistant - 메인 진입점
문서 기반 RAG 질의응답 시스템
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 필요한 디렉토리 생성
REQUIRED_DIRS = [
    PROJECT_ROOT / "data" / "models",
    PROJECT_ROOT / "data" / "chroma_db",
    PROJECT_ROOT / "data" / "metadata",
    PROJECT_ROOT / "src" / "ui" / "static",
    PROJECT_ROOT / "src" / "ui" / "templates",
]

for dir_path in REQUIRED_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)


def main():
    print()
    print("=" * 60)
    print("  Document Assistant")
    print("  문서 기반 RAG 질의응답 시스템")
    print("=" * 60)
    print()
    print("  서버 시작 중... http://127.0.0.1:7860")
    print("  모델 폴더: data/models/ (GGUF 파일을 여기에 넣으세요)")
    print("  종료하려면 Ctrl+C를 누르세요")
    print()
    print("=" * 60)
    
    from src.ui.web_app import run_server
    run_server()


if __name__ == "__main__":
    main()
