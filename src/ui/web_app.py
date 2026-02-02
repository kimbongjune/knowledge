"""
Document Assistant Web UI
FastAPI + Jinja2 기반 웹 인터페이스
"""
import os
import json
import asyncio
import secrets
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks, Response, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.document.document_processor import DocumentProcessor
from src.vector.vector_manager import VectorManager
from src.vector.incremental_manager import IncrementalManager
from src.llm.model_manager import ModelManager
from src.rag.rag_pipeline import RAGPipeline
from src.storage.chat_storage import ChatStorage
from config.settings import MODELS_PATH, HOST, PORT, UPLOAD_DIR
import httpx
import hashlib



# === Trash Helper Functions ===
def _add_to_trash(name):
    try:
        from config.settings import DATA_DIR
        import json
        trash_file = DATA_DIR / "trash.json"
        trash = set()
        if trash_file.exists():
            try: trash = set(json.loads(trash_file.read_text(encoding='utf-8')))
            except: pass
        trash.add(name)
        trash_file.write_text(json.dumps(list(trash)), encoding='utf-8')
    except: pass

def _cleanup_trash():
    try:
        from config.settings import DATA_DIR, CHROMA_DB_PATH
        import json
        import shutil
        trash_file = DATA_DIR / "trash.json"
        if not trash_file.exists(): return
        
        trash = set(json.loads(trash_file.read_text(encoding='utf-8')))
        new_trash = set()
        
        for name in trash:
            path = CHROMA_DB_PATH / name
            if not path.exists(): continue
            try:
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
            except:
                new_trash.add(name)
        
        if not new_trash:
            trash_file.unlink()
        else:
            trash_file.write_text(json.dumps(list(new_trash)), encoding='utf-8')
    except: pass
# ==============================

# === Resource Management Helpers ===
def _get_managed_file():
    from config.settings import DATA_DIR
    return DATA_DIR / "managed_paths.json"

def _load_managed_paths():
    try:
        import json
        f = _get_managed_file()
        if f.exists():
            return set(json.loads(f.read_text(encoding='utf-8')))
    except: pass
    return set()

def _save_managed_paths(paths):
    try:
        import json
        f = _get_managed_file()
        f.write_text(json.dumps(list(paths)), encoding='utf-8')
    except: pass

def _register_path(path_str):
    try:
        from pathlib import Path
        p = str(Path(path_str).resolve())
        paths = _load_managed_paths()
        paths.add(p)
        _save_managed_paths(paths)
    except: pass

def _unregister_path(path_str):
    try:
        from pathlib import Path
        p = str(Path(path_str).resolve())
        paths = _load_managed_paths()
        if p in paths:
            paths.remove(p)
            _save_managed_paths(paths)
    except: pass

def _is_managed(target_path):
    try:
        from pathlib import Path
        target = Path(target_path).resolve()
        target_str = str(target)
        paths = _load_managed_paths()
        
        # 1. 직접 등록 여부
        if target_str in paths: return True
        
        # 2. 상위 경로 등록 여부 (부모가 관리되면 자식도 관리됨)
        # 단, 루트 경로 등은 제외해야 함
        for parent in target.parents:
            if str(parent) in paths:
                return True
    except: pass
    return False

def _update_managed_path(old_path, new_path):
    try:
        from pathlib import Path
        old_p = str(Path(old_path).resolve())
        new_p = str(Path(new_path).resolve())
        
        paths = _load_managed_paths()
        # 직접 등록된 경로라면 갱신
        if old_p in paths:
            paths.remove(old_p)
            paths.add(new_p)
            _save_managed_paths(paths)
    except: pass
# ==============================

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
        self.current_user_id: Optional[str] = None  # 로그인 사용자 ID
        self.pending_session: bool = False  # 새 대화 클릭 시 True, 첫 메시지 시 세션 생성
        self.indexing_progress: dict = {"current": 0, "total": 0, "status": "", "done": True}

app_state = AppState()


def hash_password(password: str) -> str:
    """간단한 비밀번호 해싱 (bcrypt 대신 SHA256 + salt 사용)"""
    salt = "document_assistant_salt_2026"
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    return hash_password(password) == hashed

def setup_services():
    """애플리케이션 서비스 초기화"""
    app_state.doc_processor = DocumentProcessor()
    app_state.vector_manager = VectorManager()
    app_state.model_manager = ModelManager()
    app_state.chat_storage = ChatStorage()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 초기화
    print("초기화 중...")
    setup_services()
    try: _cleanup_trash() # 시작 시 쓰레기 파일 정리 시도
    except Exception as e: print(f"쓰레기 파일 정리 중 오류 발생: {e}")
    # [설정] 기본 업로드 폴더를 관리 대상으로 등록 (Whitelist 초기화)
    try: _register_path(UPLOAD_DIR)
    except: pass
    
    print("초기화 완료!")
    yield
    # 종료 시 정리
    if app_state.model_manager:
        app_state.model_manager.unload_model()


app = FastAPI(title="Document Assistant", lifespan=lifespan)

# 세션 미들웨어 추가 (쿠키 기반 인증)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

# 템플릿 및 정적 파일 설정
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# === 인증 API ===

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """로그인 페이지"""
    user_id = request.session.get("user_id")
    if user_id:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/api/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    """회원 가입"""
    if not username or not password:
        return JSONResponse({"error": "아이디와 비밀번호를 입력하세요"}, status_code=400)
    
    if len(username) < 2 or len(password) < 4:
        return JSONResponse({"error": "아이디 2자, 비밀번호 4자 이상"}, status_code=400)
    
    password_hash = hash_password(password)
    user = app_state.chat_storage.create_user(username, password_hash)
    
    if not user:
        return JSONResponse({"error": "이미 존재하는 아이디입니다"}, status_code=400)
    
    # 자동 로그인
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    app_state.current_user_id = user.id
    
    return {"success": True, "user": {"id": user.id, "username": user.username}}


@app.post("/api/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """로그인"""
    user_data = app_state.chat_storage.get_user_by_username(username)
    
    if not user_data:
        return JSONResponse({"error": "아이디 또는 비밀번호가 잘못되었습니다"}, status_code=401)
    
    user_id, user_name, password_hash, created_at = user_data
    
    if not verify_password(password, password_hash):
        return JSONResponse({"error": "아이디 또는 비밀번호가 잘못되었습니다"}, status_code=401)
    
    request.session["user_id"] = user_id
    request.session["username"] = user_name
    app_state.current_user_id = user_id
    
    return {"success": True, "user": {"id": user_id, "username": user_name}}


@app.post("/api/logout")
async def logout(request: Request):
    """로그아웃"""
    request.session.clear()
    app_state.current_user_id = None
    app_state.current_session_id = None
    return {"success": True}


@app.get("/api/me")
async def get_current_user(request: Request):
    """현재 로그인 사용자 확인"""
    user_id = request.session.get("user_id")
    username = request.session.get("username")
    
    if user_id:
        return {"logged_in": True, "user": {"id": user_id, "username": username}}
    return {"logged_in": False, "user": None}


# === API 엔드포인트 ===

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지"""
    # 세션에서 사용자 ID 가져오기
    user_id = request.session.get("user_id")
    username = request.session.get("username")
    
    if user_id:
        app_state.current_user_id = user_id
    
    # 실제 설치된 모델만 표시
    models = app_state.model_manager.list_installed_models() if app_state.model_manager else []
    
    # 컬렉션 목록 가져오기 (쓰레기통 필터링 포함)
    collections = []
    if app_state.vector_manager:
        all_collections = app_state.vector_manager.list_collections()
        # 쓰레기통 로직 직접 구현 (함수 호출 문제 방지)
        trash = set()
        try:
            from config.settings import DATA_DIR
            import json
            trash_file = DATA_DIR / "trash.json"
            if trash_file.exists():
                trash = set(json.loads(trash_file.read_text(encoding='utf-8')))
        except:
            pass
            
        collections = [c for c in all_collections if c not in trash]
    
    collection_stats = []
    for coll in collections:
        # 유효한 컬렉션 이름만 처리 (영문/숫자만) -> coll은 문자열임
        if not coll or not isinstance(coll, str) or not coll.isascii():
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
        "current_collection": app_state.current_collection,
        "user": {"id": user_id, "username": username} if user_id else None
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
            items.append({"name": "..", "path": str(path.parent), "is_dir": True, "can_delete": False})
        
        # 하위 항목
        try:
            # 폴더 먼저
            dirs = []
            files = []
            
            for item in sorted(path.iterdir()):
                # [수정] 관리 권한 체크 (Whitelist 기반)
                is_managed_item = _is_managed(item)

                if item.is_dir():
                    dirs.append({
                        "name": item.name,
                        "path": str(item),
                        "is_dir": True,
                        "can_delete": is_managed_item
                    })
                elif item.is_file():
                     # 지원하는 확장자만 표시 (또는 모든 파일 표시하고 아이콘으로 구분)
                     from config.settings import SUPPORTED_EXTENSIONS
                     if item.suffix.lower() in SUPPORTED_EXTENSIONS or item.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                         files.append({
                            "name": item.name,
                            "path": str(item), 
                            "is_dir": False,
                            "size": item.stat().st_size,
                            "can_delete": is_managed_item
                        })
            
            items.extend(dirs)
            items.extend(files)
            
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
                "collection": collection_name,
                "chunks": 0,
                "deleted_chunks": 0,
                "indexed_count": inc_manager.get_indexed_count()
            }
        
        # 삭제된 파일 처리
        deleted_chunks_count = 0
        if changes.deleted:
            deleted_chunks_count = app_state.vector_manager.remove_documents_by_source(changes.deleted, collection_name)
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
            "deleted_chunks": deleted_chunks_count,
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


    try:
        if not app_state.model_manager._current_model:
            return JSONResponse({"error": "먼저 모델을 로드하세요."}, status_code=400)
        
        app_state.rag_pipeline = RAGPipeline(app_state.vector_manager, app_state.model_manager)
        app_state.rag_pipeline.setup_chain(collection_name)
        app_state.current_collection = collection_name
        
        return {"success": True, "collection": collection_name}
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
            
            # 세션 생성 (로그인한 사용자만, 첫 메시지 시)
            if app_state.current_user_id and not app_state.current_session_id:
                session = app_state.chat_storage.create_session(
                    collection_name=app_state.current_collection,
                    model_name=app_state.model_manager._current_model_name,
                    user_id=app_state.current_user_id
                )
                app_state.current_session_id = session.id
                app_state.pending_session = False
            
            # 사용자 메시지 저장 (로그인한 사용자만)
            if app_state.current_user_id and app_state.current_session_id:
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
            
            # AI 응답 저장 (로그인한 사용자만)
            if app_state.current_user_id and app_state.current_session_id:
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
    
    # Trash 필터링 (직접 구현)
    try:
        from config.settings import DATA_DIR
        import json
        trash_file = DATA_DIR / "trash.json"
        if trash_file.exists():
            trash = set(json.loads(trash_file.read_text(encoding='utf-8')))
            collections = [c for c in collections if c not in trash]
    except: pass
    
    result = []
    for coll in collections:
        try:
            stats = app_state.vector_manager.get_collection_stats(coll)
            result.append({
                "name": coll,
                "count": stats.get("document_count", 0)
            })
        except Exception as e:
            # 보이지 않는 불량 컬렉션(이름 규칙 위반 등)은 건너뜀
            print(f"[Warn] Skipping invalid collection '{coll}': {e}")
            continue
    
    return {"collections": result, "current": app_state.current_collection}


@app.post("/api/clear-chat")
async def clear_chat():
    """새 대화 시작 (DB에 저장하지 않음 - 첫 메시지 전송 시 세션 생성)"""
    app_state.current_session_id = None
    app_state.pending_session = True  # 첫 메시지에서 세션 생성 예약
    return {"success": True, "session_id": None}



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
async def get_sessions(request: Request):
    """세션 목록 (로그인한 사용자의 세션만)"""
    user_id = request.session.get("user_id")
    
    # 로그인하지 않은 경우 빈 목록 반환
    if not user_id:
        return {"sessions": [], "current_session_id": None}
    
    sessions = app_state.chat_storage.list_sessions(user_id=user_id)
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
async def create_session(request: Request):
    """새 세션 생성"""
    user_id = request.session.get("user_id")
    
    # 로그인하지 않은 경우 세션 생성 안 함
    if not user_id:
        return {"success": False, "error": "로그인이 필요합니다", "session_id": None}
    
    session = app_state.chat_storage.create_session(
        title="새 대화",
        collection_name=app_state.current_collection,
        model_name=app_state.model_manager._current_model_name if app_state.model_manager else None,
        user_id=user_id
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


# === 파일 업로드 API ===

@app.get("/api/upload-dir")
async def get_upload_dir():
    """업로드 디렉토리 경로 및 파일 목록"""
    try:
        files = []
        if UPLOAD_DIR.exists():
            for item in UPLOAD_DIR.iterdir():
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "size": item.stat().st_size,
                        "path": str(item)
                    })
                elif item.is_dir():
                    files.append({
                        "name": item.name,
                        "is_dir": True,
                        "path": str(item)
                    })
        return {
            "path": str(UPLOAD_DIR),
            "files": files
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/upload-files")
async def upload_files(files: List[UploadFile] = File(...), target_path: str = Form(None)):
    """파일 업로드 (지정된 경로 또는 서버의 uploads 폴더로)"""
    try:
        # 업로드 대상 디렉토리 결정
        if target_path:
            upload_dir = Path(target_path)
            if not upload_dir.exists():
                return JSONResponse({"error": "대상 경로가 존재하지 않습니다"}, status_code=400)
            if not upload_dir.is_dir():
                return JSONResponse({"error": "대상 경로는 폴더여야 합니다"}, status_code=400)
        else:
            upload_dir = UPLOAD_DIR
            
        uploaded = []
        for file in files:
            file_path = upload_dir / file.filename
            
            # 중복 파일명 처리
            counter = 1
            original_stem = file_path.stem
            while file_path.exists():
                file_path = upload_dir / f"{original_stem}_{counter}{file_path.suffix}"
                counter += 1
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # [등록] 생성된 파일 관리 대상에 추가
            _register_path(file_path)
            
            uploaded.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": len(content)
            })
        
        return {"success": True, "files": uploaded, "upload_dir": str(upload_dir)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/mkdir")
async def create_folder(path: str = Form(...), name: str = Form(...)):
    """새 폴더 생성"""
    try:
        base_path = Path(path)
        if not base_path.exists() or not base_path.is_dir():
             return JSONResponse({"error": "상위 경로가 유효하지 않습니다"}, status_code=400)
             
        new_folder = base_path / name
        if new_folder.exists():
             return JSONResponse({"error": "이미 존재하는 폴더 이름입니다"}, status_code=400)
             
        new_folder.mkdir()
        # [등록] 생성된 폴더 관리 대상에 추가
        _register_path(new_folder)
        return {"success": True, "path": str(new_folder)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/delete-item")
async def delete_item(path: str = Form(...)):
    """파일/폴더 삭제 (업로드 폴더 내에서만 허용)"""
    import shutil
    try:
        target = Path(path).resolve()
        
        # [수정] 사용자 요청: 웹에서 생성된 리소스만 삭제 가능 (Whitelist)
        if not _is_managed(target):
             return JSONResponse({"error": "권한이 없습니다. 웹에서 생성된 항목만 삭제할 수 있습니다."}, status_code=403)

        # 시스템 루트 등 2차 방어
        import os
        if len(target.parts) <= 1 or str(target).upper() == os.environ.get("SystemRoot", "C:\\WINDOWS").upper():
             return JSONResponse({"error": "시스템 경로는 삭제할 수 없습니다"}, status_code=403)
        # except ValueError:
        #      return JSONResponse({"error": "유효하지 않은 경로입니다"}, status_code=400)
        
        if not target.exists():
             return JSONResponse({"error": "항목이 존재하지 않습니다"}, status_code=404)
             
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
            
        return {"success": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/rename-item")
async def rename_item(path: str = Form(...), new_name: str = Form(...)):
    """파일/폴더 이름 변경"""
    try:
        target = Path(path).resolve()
        
        # [수정] 사용자 요청: 웹에서 생성된 항목만 수정 가능
        if not _is_managed(target):
            return JSONResponse({"error": "권한이 없습니다. 웹에서 생성된 항목만 수정할 수 있습니다."}, status_code=403)

        # [주의] 존재 여부 체크 시 권한 문제 있을 수 있으나 try로 커버
        if not target.exists():
             return JSONResponse({"error": "항목이 존재하지 않습니다"}, status_code=404)
        
        new_path = target.parent / new_name
        if new_path.exists():
             return JSONResponse({"error": "이미 존재하는 이름입니다"}, status_code=400)
             
        target.rename(new_path)
        # [갱신] 관리 경로 업데이트
        _update_managed_path(target, new_path)
        
        return {"success": True, "path": str(new_path)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/go-upload-dir")
async def go_upload_dir():
    """업로드 디렉토리로 이동 (폴더 경로 반환)"""
    return {"path": str(UPLOAD_DIR)}





# === SQLite Administration (H2-style Admin Console) ===

@app.get("/admin", response_class=HTMLResponse)
@app.get("/admin/", response_class=HTMLResponse)
async def admin_page(request: Request):
    """SQLite Admin 페이지 - H2 스타일 데이터베이스 콘솔"""
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/api/admin/databases")
async def get_admin_databases():
    """사용 가능한 데이터베이스 목록 반환"""
    from config.settings import DATA_DIR
    try:
        db_files = []
        
        # DATA_DIR 내의 모든 DB 파일 스캔
        if DATA_DIR.exists():
            for f in os.listdir(DATA_DIR):
                if f.endswith(('.db', '.sqlite3')):
                    db_files.append({
                        "name": f,
                        "path": str(DATA_DIR / f)
                    })
        
        # chroma_db 내의 sqlite 파일도 스캔
        chroma_dir = DATA_DIR / "chroma_db"
        if chroma_dir.exists():
            for root, dirs, files in os.walk(chroma_dir):
                for f in files:
                    if f.endswith('.sqlite3'):
                        rel_path = os.path.relpath(os.path.join(root, f), DATA_DIR)
                        db_files.append({
                            "name": f"chroma: {rel_path}",
                            "path": os.path.join(root, f)
                        })
        
        return {"databases": db_files}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/admin/tables")
async def get_admin_tables(db: str = "metadata.db"):
    """선택된 데이터베이스의 테이블 목록 반환"""
    from config.settings import DATA_DIR
    import sqlite3
    
    try:
        # DB 경로 결정
        if db.startswith("/") or db.startswith("C:") or db.startswith("c:"):
            db_path = db
        else:
            db_path = str(DATA_DIR / db)
        
        if not os.path.exists(db_path):
            return {"error": f"Database not found: {db}", "tables": []}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록 조회
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return {"tables": tables, "db": db}
    except Exception as e:
        return {"error": str(e), "tables": []}


@app.post("/api/admin/query")
async def execute_admin_query(request: Request):
    """SQL 쿼리 실행"""
    from config.settings import DATA_DIR
    import sqlite3
    
    try:
        body = await request.json()
        db = body.get("db", "metadata.db")
        query = body.get("query", "").strip()
        
        if not query:
            return {"error": "Query is empty", "columns": [], "rows": []}
        
        # DB 경로 결정
        if db.startswith("/") or db.startswith("C:") or db.startswith("c:"):
            db_path = db
        else:
            db_path = str(DATA_DIR / db)
        
        if not os.path.exists(db_path):
            return {"error": f"Database not found: {db}", "columns": [], "rows": []}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 쿼리 타입 판별
        query_upper = query.upper().strip()
        is_select = query_upper.startswith("SELECT") or query_upper.startswith("PRAGMA") or query_upper.startswith("EXPLAIN")
        
        cursor.execute(query)
        
        if is_select:
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            return {"columns": columns, "rows": rows, "affected": len(rows)}
        else:
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return {"columns": ["Result"], "rows": [[f"{affected} row(s) affected"]], "affected": affected}
            
    except sqlite3.Error as e:
        return {"error": f"SQL Error: {str(e)}", "columns": [], "rows": []}
    except Exception as e:
        return {"error": str(e), "columns": [], "rows": []}


@app.get("/api/admin/table-info")
async def get_table_info(db: str, table: str):
    """테이블 구조 정보 반환"""
    from config.settings import DATA_DIR
    import sqlite3
    
    try:
        # DB 경로 결정
        if db.startswith("/") or db.startswith("C:") or db.startswith("c:"):
            db_path = db
        else:
            db_path = str(DATA_DIR / db)
        
        if not os.path.exists(db_path):
            return {"error": f"Database not found: {db}"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 정보 조회
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        
        # 인덱스 정보
        cursor.execute(f"PRAGMA index_list('{table}')")
        indexes = cursor.fetchall()
        
        # 행 수
        cursor.execute(f"SELECT COUNT(*) FROM '{table}'")
        row_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "table": table,
            "columns": [
                {
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "notnull": col[3],
                    "default": col[4],
                    "pk": col[5]
                } for col in columns
            ],
            "indexes": indexes,
            "row_count": row_count
        }
    except Exception as e:
        return {"error": str(e)}

def run_server():
    """서버 실행"""
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    run_server()
