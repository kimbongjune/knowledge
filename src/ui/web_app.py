"""
Document Assistant Web UI
FastAPI + Jinja2 기반 웹 인터페이스
"""
import os
import json
import asyncio
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.document.document_processor import DocumentProcessor
from src.vector.vector_manager import VectorManager
from src.vector.incremental_manager import IncrementalManager
from src.llm.model_manager import ModelManager
from src.rag.rag_pipeline import RAGPipeline
from src.storage.chat_storage import ChatStorage
from config.settings import MODELS_PATH, HOST, PORT
import httpx
import json


# 전역 상태
class AppState:
    def __init__(self):
        self.doc_processor: Optional[DocumentProcessor] = None
        self.vector_manager: Optional[VectorManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.chat_storage: Optional[ChatStorage] = None
        self.current_collection: Optional[str] = None
        self.current_session_id: Optional[str] = None
        self.indexing_progress: dict = {"current": 0, "total": 0, "status": "", "done": True}

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 초기화
    print("초기화 중...")
    app_state.doc_processor = DocumentProcessor()
    app_state.vector_manager = VectorManager()
    app_state.model_manager = ModelManager()
    app_state.chat_storage = ChatStorage()
    print("초기화 완료!")
    yield
    # 종료 시 정리
    if app_state.model_manager:
        app_state.model_manager.unload_model()


app = FastAPI(title="Document Assistant", lifespan=lifespan)

# 템플릿 및 정적 파일 설정
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# === API 엔드포인트 ===

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지"""
    # 실제 설치된 모델만 표시
    models = app_state.model_manager.list_installed_models() if app_state.model_manager else []
    collections = app_state.vector_manager.list_collections() if app_state.vector_manager else []
    
    collection_stats = []
    for coll in collections:
        # 유효한 컬렉션 이름만 처리 (영문/숫자만)
        if not coll or not coll.isascii():
            continue
        try:
            stats = app_state.vector_manager.get_collection_stats(coll)
            collection_stats.append({
                "name": coll,
                "count": stats.get("document_count", 0)
            })
        except Exception:
            # 잘못된 컬렉션은 건너뛰기
            continue
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": [{"name": m.name, "size": m.size, "is_vision": m.is_vision} for m in models],
        "collections": collection_stats,
        "current_model": app_state.model_manager._current_model_name if app_state.model_manager else None,
        "current_collection": app_state.current_collection
    })


@app.get("/api/browse")
async def browse_folder(path: str = ""):
    """폴더 탐색 API"""
    try:
        if not path:
            # 드라이브 목록 반환 (Windows)
            if os.name == 'nt':
                import string
                drives = []
                for letter in string.ascii_uppercase:
                    drive = f"{letter}:/"
                    if os.path.exists(drive):
                        drives.append({"name": drive, "path": drive, "is_dir": True})
                return {"items": drives, "current": ""}
            else:
                path = "/"
        
        path = Path(path)
        if not path.exists():
            return {"error": "경로가 존재하지 않습니다.", "items": [], "current": str(path)}
        
        items = []
        
        # 상위 폴더
        if path.parent != path:
            items.append({"name": "..", "path": str(path.parent), "is_dir": True})
        
        # 하위 항목
        try:
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "is_dir": True
                    })
        except PermissionError:
            pass
        
        return {"items": items, "current": str(path)}
    except Exception as e:
        return {"error": str(e), "items": [], "current": path}


@app.post("/api/index")
async def start_indexing(folder_path: str = Form(...), collection_name: str = Form("")):
    """인덱싱 시작"""
    if not folder_path or not os.path.exists(folder_path):
        return JSONResponse({"error": "유효한 폴더 경로가 아닙니다."}, status_code=400)
    
    if not collection_name:
        collection_name = Path(folder_path).name
    
    # 컬렉션 이름 정리 (영문, 숫자, 언더스코어만 허용)
    # 한글 등은 제거하고 영문/숫자만 유지
    cleaned_name = "".join(c if c.isascii() and (c.isalnum() or c == "_") else "" for c in collection_name)
    
    # 빈 문자열이면 기본값 사용
    if not cleaned_name or len(cleaned_name) < 3:
        import hashlib
        hash_suffix = hashlib.md5(folder_path.encode()).hexdigest()[:8]
        cleaned_name = f"collection_{hash_suffix}"
    
    collection_name = cleaned_name
    
    try:
        inc_manager = IncrementalManager(collection_name)
        changes = inc_manager.get_changes(folder_path)
        
        if not changes.has_changes():
            return {
                "success": True,
                "message": "변경 사항 없음. 이미 최신 상태입니다.",
                "indexed_count": inc_manager.get_indexed_count()
            }
        
        # 삭제된 파일 처리
        if changes.deleted:
            app_state.vector_manager.remove_documents_by_source(changes.deleted, collection_name)
            inc_manager.remove_file_metadata(changes.deleted)
        
        # 추가/수정된 파일 처리
        files_to_process = changes.added + changes.modified
        
        all_documents = []
        for file_path in files_to_process:
            if file_path in changes.modified:
                app_state.vector_manager.remove_documents_by_source([str(file_path)], collection_name)
            docs = app_state.doc_processor.load_document(file_path)
            all_documents.extend(docs)
        
        # 청크 분할 및 저장
        added_count = 0
        if all_documents:
            chunked_docs = app_state.doc_processor.text_splitter.split_documents(all_documents)
            added_count = app_state.vector_manager.add_documents(chunked_docs, collection_name)
            inc_manager.update_files_metadata(files_to_process)
        
        app_state.current_collection = collection_name
        
        return {
            "success": True,
            "message": "인덱싱 완료",
            "collection": collection_name,
            "added": len(changes.added),
            "modified": len(changes.modified),
            "deleted": len(changes.deleted),
            "chunks": added_count,
            "indexed_count": inc_manager.get_indexed_count()
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/load-model")
async def load_model(model_name: str = Form(...)):
    """모델 로드"""
    try:
        app_state.model_manager.load_model(model_name)
        return {"success": True, "model": model_name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/setup-rag")
async def setup_rag(collection_name: str = Form(...)):
    """RAG 파이프라인 설정"""
    try:
        if not app_state.model_manager._current_model:
            return JSONResponse({"error": "먼저 모델을 로드하세요."}, status_code=400)
        
        app_state.rag_pipeline = RAGPipeline(app_state.vector_manager, app_state.model_manager)
        app_state.rag_pipeline.setup_chain(collection_name)
        app_state.current_collection = collection_name
        
        return {"success": True, "collection": collection_name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/collections/rename")
async def rename_collection(old_name: str = Form(...), new_name: str = Form(...)):
    """컬렉션 이름 변경"""
    try:
        # 이름 유효성 검사 (영문, 숫자, 언더스코어만 허용)
        cleaned_name = "".join(c for c in new_name if c.isascii() and (c.isalnum() or c == "_"))
        if not cleaned_name:
            return JSONResponse({"error": "유효하지 않은 컬렉션 이름입니다. 영문, 숫자, _ 만 사용 가능합니다."}, status_code=400)
            
        app_state.vector_manager.rename_collection(old_name, cleaned_name)
        
        # 현재 선택된 컬렉션이면 업데이트
        if app_state.current_collection == old_name:
            app_state.current_collection = cleaned_name
            
        return {"success": True, "old_name": old_name, "new_name": cleaned_name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/chat")
async def chat(
    question: str = Form(...), 
    attachments: str = Form(default="[]"),
    options: str = Form(default="{}")
):
    """질의응답 (스트리밍 + 대화 기록 + 파일 첨부 + 고급 설정)"""
    if not app_state.rag_pipeline:
        return JSONResponse({"error": "RAG가 설정되지 않았습니다."}, status_code=400)
    
    # 설정 파싱
    try:
        opts = json.loads(options)
    except:
        opts = {}
        
    k_value = int(opts.get("k", 7))
    num_predict = int(opts.get("num_predict", 2048))
    system_prompt = opts.get("system_prompt", "")
    
    # 문서 검색 제한 확인 (컬렉션 문서 수보다 크면 조정)
    if app_state.vector_manager and app_state.current_collection:
        try:
            stats = app_state.vector_manager.get_collection_stats(app_state.current_collection)
            max_docs = stats.get("document_count", 0)
            if k_value > max_docs:
                k_value = max_docs
        except:
            pass
    
    # 첨부파일 파싱
    try:
        attachment_list = json.loads(attachments)
    except:
        attachment_list = []
    
    async def generate():
        try:
            # 1. 관련 문서 검색 (사용자 설정 k 적용)
            docs = app_state.vector_manager.similarity_search(
                question, 
                app_state.current_collection, 
                k=k_value
            ) if k_value > 0 else []
            
            # 문서 정보 전송
            sources = []
            for doc in docs:
                source = doc.metadata.get('source', '')
                if source and source not in [s.get('path') for s in sources]:
                    sources.append({
                        "filename": doc.metadata.get('filename', 'Unknown'),
                        "path": source
                    })
            
            yield f"data: {json.dumps({'type': 'sources', 'data': sources[:5]})}\n\n"
            
            # 2. 컨텍스트 구성
            context = "\n\n---\n\n".join(
                f"[문서: {doc.metadata.get('filename', 'Unknown')}]\n{doc.page_content}" 
                for doc in docs
            )
            
            # 3. 대화 기록 구성 (SQLite에서 로드)
            history_text = ""
            if app_state.current_session_id:
                recent_messages = app_state.chat_storage.get_recent_messages(app_state.current_session_id, count=10)
                if recent_messages:
                    history_text = "\n\n[이전 대화]\n"
                    for m in recent_messages:
                        if m.role == "user":
                            history_text += f"사용자: {m.content}\n"
                        else:
                            content = m.content[:200] + "..." if len(m.content) > 200 else m.content
                            history_text += f"AI: {content}\n\n"
            
            # 4. 프롬프트 구성 (사용자 시스템 프롬프트 반영)
            base_system = system_prompt if system_prompt else """당신은 문서 분석 전문 AI 어시스턴트입니다.
제공된 문서를 참고하여 질문에 정확하고 상세하게 답변해주세요.

## 답변 규칙
1. 문서에 있는 내용만 사용하세요
2. 문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변
3. 가능하면 구체적인 수치, 날짜, 이름 등을 포함
4. 답변은 명확하고 구조적으로 작성 (필요시 번호나 글머리 사용)
5. 참고한 문서명을 언급하면 더 좋음"""

            prompt = f"""{base_system}
{history_text}
[참고 문서]
{context}

[질문]
{question}

[답변]"""
            
            # 5. 첨부 문서가 있으면 프롬프트에 추가
            doc_attachments = [a for a in attachment_list if a.get('type') == 'document']
            if doc_attachments:
                attached_docs = "\n\n".join([f"[첨부파일: {a['name']}]\n{a['data'][:3000]}" for a in doc_attachments])
                prompt = prompt.replace("[참고 문서]", f"[첨부 문서]\n{attached_docs}\n\n[참고 문서]")
            
            # 6. 이미지 첨부 (비전 모델용)
            image_attachments = [a['data'] for a in attachment_list if a.get('type') == 'image']
            is_vision = app_state.model_manager.is_current_model_vision()
            
            # 7. GPU/CPU 감지 및 최적화 설정
            import os
            has_gpu = False
            try:
                # Ollama API로 GPU 사용 가능 여부 확인
                async with httpx.AsyncClient(timeout=5.0) as check_client:
                    ps_response = await check_client.get("http://localhost:11434/api/ps")
                    if ps_response.status_code == 200:
                        ps_data = ps_response.json()
                        # 실행 중인 모델이 GPU를 사용하는지 확인
                        for model in ps_data.get("models", []):
                            if model.get("size_vram", 0) > 0:
                                has_gpu = True
                                break
            except:
                # API 호출 실패 시 환경변수로 판단
                has_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") is not None
            
            # CPU 코어 수 확인
            cpu_count = os.cpu_count() or 4
            
            # 8. Ollama 비동기 스트리밍 호출
            model_options = {
                "temperature": float(opts.get("temperature", 0.3)),
                "top_p": 0.9,
                "num_predict": num_predict
            }
            
            # GPU가 없으면 CPU 스레드 수 설정
            if not has_gpu:
                model_options["num_thread"] = cpu_count
                model_options["num_gpu"] = 0  # GPU 레이어 비활성화
            
            request_body = {
                "model": app_state.model_manager._current_model_name or "qwen2.5:7b",
                "prompt": prompt,
                "stream": True,
                "options": model_options
            }
            
            if is_vision and image_attachments:
                request_body["images"] = image_attachments
            
            # 세션 생성 (필요시)
            if not app_state.current_session_id:
                session = app_state.chat_storage.create_session(
                    collection_name=app_state.current_collection,
                    model_name=app_state.model_manager._current_model_name
                )
                app_state.current_session_id = session.id
            
            # 사용자 메시지 저장
            app_state.chat_storage.add_message(
                app_state.current_session_id, 
                "user", 
                question
            )
            
            full_answer = ""
            # CPU 추론은 매우 느릴 수 있으므로 타임아웃을 충분히 설정
            timeout = httpx.Timeout(
                connect=60.0,    # 연결 타임아웃 60초
                read=1800.0,     # 읽기 타임아웃 30분 (CPU 추론용)
                write=60.0,      # 쓰기 타임아웃 60초
                pool=60.0        # 연결 풀 타임아웃 60초
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", "http://localhost:11434/api/generate", json=request_body) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                if token:
                                    full_answer += token
                                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
                                if data.get("done"):
                                    break
                            except:
                                pass
            
            # AI 응답 저장
            app_state.chat_storage.add_message(
                app_state.current_session_id, 
                "assistant", 
                full_answer,
                sources=sources[:3]
            )
            
            # 자동 제목 생성
            session = app_state.chat_storage.get_session(app_state.current_session_id)
            if session and session.message_count <= 2:
                app_state.chat_storage.auto_title_from_first_message(app_state.current_session_id)
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """모델 파일 업로드"""
    try:
        if not file.filename.endswith('.gguf'):
            return JSONResponse({"error": "GGUF 파일만 업로드 가능합니다."}, status_code=400)
        
        # 모델 폴더에 저장
        models_path = Path(MODELS_PATH)
        models_path.mkdir(parents=True, exist_ok=True)
        
        file_path = models_path / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {"success": True, "filename": file.filename, "path": str(file_path)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/models")
async def get_models():
    """모델 목록"""
    models = app_state.model_manager.list_models() if app_state.model_manager else []
    return {
        "models": [{"name": m.name, "size": m.size_gb, "loaded": m.loaded} for m in models],
        "current": app_state.model_manager._current_model_name if app_state.model_manager else None
    }


@app.get("/api/collections")
async def get_collections():
    """컬렉션 목록"""
    collections = app_state.vector_manager.list_collections() if app_state.vector_manager else []
    
    result = []
    for coll in collections:
        stats = app_state.vector_manager.get_collection_stats(coll)
        result.append({
            "name": coll,
            "count": stats.get("document_count", 0)
        })
    
    return {"collections": result, "current": app_state.current_collection}


@app.post("/api/clear-chat")
async def clear_chat():
    """새 대화 시작 (새 세션 생성)"""
    session = app_state.chat_storage.create_session(
        collection_name=app_state.current_collection,
        model_name=app_state.model_manager._current_model_name if app_state.model_manager else None
    )
    app_state.current_session_id = session.id
    return {"success": True, "session_id": session.id}


@app.post("/api/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """파일에서 텍스트 추출"""
    try:
        import tempfile
        import os
        
        # 임시 파일로 저장
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # 문서 처리기로 텍스트 추출
            docs = app_state.doc_processor.process_file(Path(tmp_path))
            text = "\n\n".join([doc.page_content for doc in docs])
            
            return {"success": True, "text": text[:10000], "filename": file.filename}  # 최대 10000자
        finally:
            os.unlink(tmp_path)  # 임시 파일 삭제
            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/open-folder")
async def open_folder(path: str = Form(...)):
    """파일이 있는 폴더를 서버에서 열기 (서버가 윈도우/리눅스일 때 모두 지원)"""
    import subprocess
    import platform
    from pathlib import Path
    
    try:
        file_path = Path(path)
        folder_path = file_path.parent if file_path.is_file() else file_path
        
        if not folder_path.exists():
            return JSONResponse({"error": "폴더가 존재하지 않습니다."}, status_code=404)
        
        system = platform.system()
        if system == "Windows":
            # 윈도우: 폴더 열고 파일 선택
            if file_path.is_file():
                subprocess.run(["explorer", "/select,", str(file_path)], check=False)
            else:
                subprocess.run(["explorer", str(folder_path)], check=False)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(folder_path)], check=False)
        else:  # Linux
            subprocess.run(["xdg-open", str(folder_path)], check=False)
        
        return {"success": True, "message": f"폴더 열기: {folder_path}"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# === 모델 관리 API ===

@app.get("/api/models/installed")
async def get_installed_models():
    """설치된 모델 목록"""
    models = app_state.model_manager.list_installed_models()
    return {"models": [{"name": m.name, "size": m.size, "is_vision": m.is_vision} for m in models]}


@app.get("/api/models/search")
async def search_models(q: str = ""):
    """모델 검색 (Ollama 라이브러리)"""
    results = app_state.model_manager.search_models(q)
    return {"models": results}


@app.post("/api/models/pull")
async def pull_model(model_name: str = Form(...)):
    """모델 다운로드 (스트리밍 진행률)"""
    async def generate():
        for progress in app_state.model_manager.pull_model_stream(model_name):
            yield f"data: {json.dumps(progress)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/models/{model_name:path}")
async def delete_model(model_name: str):
    """모델 삭제"""
    success = app_state.model_manager.delete_model(model_name)
    if success:
        return {"success": True, "message": f"{model_name} 삭제 완료"}
    else:
        return JSONResponse({"error": "모델 삭제 실패"}, status_code=500)


# === 세션 관리 API ===

@app.get("/api/sessions")
async def get_sessions():
    """세션 목록"""
    sessions = app_state.chat_storage.list_sessions()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "collection": s.collection_name,
                "model": s.model_name,
                "updated_at": s.updated_at[:16].replace("T", " "),
                "message_count": s.message_count
            } for s in sessions
        ],
        "current_session_id": app_state.current_session_id
    }


@app.post("/api/sessions")
async def create_session():
    """새 세션 생성"""
    session = app_state.chat_storage.create_session(
        title="새 대화",
        collection_name=app_state.current_collection,
        model_name=app_state.model_manager._current_model_name if app_state.model_manager else None
    )
    app_state.current_session_id = session.id
    return {"success": True, "session_id": session.id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """세션 로드 (메시지 포함)"""
    session = app_state.chat_storage.get_session(session_id)
    if not session:
        return JSONResponse({"error": "세션을 찾을 수 없습니다"}, status_code=404)
    
    messages = app_state.chat_storage.get_messages(session_id)
    app_state.current_session_id = session_id
    
    # 세션의 컬렉션과 모델 정보로 복원
    if session.collection_name:
        app_state.current_collection = session.collection_name
    
    return {
        "session": {
            "id": session.id,
            "title": session.title,
            "collection": session.collection_name,
            "model": session.model_name
        },
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources
            } for m in messages
        ]
    }


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, title: str = Form(None)):
    """세션 이름 변경"""
    if title:
        app_state.chat_storage.update_session(session_id, title=title)
    return {"success": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    success = app_state.chat_storage.delete_session(session_id)
    if session_id == app_state.current_session_id:
        app_state.current_session_id = None
    return {"success": success}


def run_server():
    """서버 실행"""
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run_server()
