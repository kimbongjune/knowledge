"""
Document Assistant Web UI
Gradio 기반 사용자 인터페이스
"""
import os
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.document.document_processor import DocumentProcessor
from src.vector.vector_manager import VectorManager
from src.vector.incremental_manager import IncrementalManager
from src.llm.model_manager import ModelManager
from src.rag.rag_pipeline import RAGPipeline
from config.settings import MODELS_PATH


class DocumentAssistantUI:
    """문서 도우미 UI 클래스"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_manager = VectorManager()
        self.model_manager = ModelManager()
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.current_collection: Optional[str] = None
        
    def get_model_choices(self) -> List[str]:
        """사용 가능한 모델 목록"""
        models = self.model_manager.list_models()
        if not models:
            return ["모델 없음 - data/models/ 폴더에 GGUF 파일을 추가하세요"]
        return [f"{m.name} ({m.size_gb}GB)" for m in models]
    
    def get_collection_choices(self) -> List[str]:
        """인덱싱된 컬렉션 목록"""
        collections = self.vector_manager.list_collections()
        if not collections:
            return []
        
        result = []
        for coll in collections:
            stats = self.vector_manager.get_collection_stats(coll)
            doc_count = stats.get('document_count', 0)
            result.append(f"{coll} ({doc_count} chunks)")
        return result
    
    def scan_and_index(
        self, 
        folder_path: str, 
        collection_name: str,
        progress=gr.Progress()
    ) -> str:
        """폴더 스캔 및 벡터화"""
        if not folder_path or not os.path.exists(folder_path):
            return "❌ 유효한 폴더 경로를 입력하세요."
        
        if not collection_name:
            # 폴더 이름을 컬렉션 이름으로 사용
            collection_name = Path(folder_path).name
        
        # 컬렉션 이름 정리 (특수문자 제거)
        collection_name = "".join(c if c.isalnum() or c == "_" else "_" for c in collection_name)
        
        try:
            # 증분 관리자 초기화
            inc_manager = IncrementalManager(collection_name)
            changes = inc_manager.get_changes(folder_path)
            
            if not changes.has_changes():
                return f"✅ 변경 사항 없음. 이미 최신 상태입니다.\n📊 인덱싱된 파일: {inc_manager.get_indexed_count()}개"
            
            progress(0, desc="파일 스캔 중...")
            
            # 삭제된 파일 처리
            if changes.deleted:
                progress(0.1, desc=f"삭제된 파일 처리 중... ({len(changes.deleted)}개)")
                self.vector_manager.remove_documents_by_source(changes.deleted, collection_name)
                inc_manager.remove_file_metadata(changes.deleted)
            
            # 추가/수정된 파일 처리
            files_to_process = changes.added + changes.modified
            total_files = len(files_to_process)
            
            if total_files == 0:
                return f"✅ 삭제된 파일만 처리됨 ({len(changes.deleted)}개)"
            
            all_documents = []
            for idx, file_path in enumerate(files_to_process):
                progress((idx + 1) / total_files, desc=f"처리 중: {file_path.name}")
                
                # 수정된 파일은 기존 벡터 삭제
                if file_path in changes.modified:
                    self.vector_manager.remove_documents_by_source([str(file_path)], collection_name)
                
                docs = self.doc_processor.load_document(file_path)
                all_documents.extend(docs)
            
            # 청크 분할
            progress(0.9, desc="문서 청크 분할 중...")
            if all_documents:
                chunked_docs = self.doc_processor.text_splitter.split_documents(all_documents)
                
                # 벡터 DB에 저장
                progress(0.95, desc="벡터 DB에 저장 중...")
                added_count = self.vector_manager.add_documents(chunked_docs, collection_name)
                
                # 메타데이터 업데이트
                inc_manager.update_files_metadata(files_to_process)
            else:
                added_count = 0
            
            progress(1.0, desc="완료!")
            
            result = f"""✅ 인덱싱 완료!

📊 처리 결과:
- 컬렉션: {collection_name}
- 추가된 파일: {len(changes.added)}개
- 수정된 파일: {len(changes.modified)}개
- 삭제된 파일: {len(changes.deleted)}개
- 생성된 청크: {added_count}개
- 총 인덱싱 파일: {inc_manager.get_indexed_count()}개"""
            
            self.current_collection = collection_name
            return result
            
        except Exception as e:
            return f"❌ 오류 발생: {str(e)}"
    
    def load_model(self, model_selection: str, progress=gr.Progress()) -> str:
        """모델 로드"""
        if not model_selection or "모델 없음" in model_selection:
            return "❌ 먼저 모델을 선택하세요."
        
        try:
            # 모델 이름 추출 (크기 정보 제거)
            model_name = model_selection.split(" (")[0]
            
            progress(0.3, desc="모델 로딩 중... (시간이 걸릴 수 있습니다)")
            self.model_manager.load_model(model_name)
            progress(1.0, desc="완료!")
            
            return f"✅ 모델 로드 완료: {model_name}"
        except Exception as e:
            return f"❌ 모델 로드 실패: {str(e)}"
    
    def setup_rag(self, collection_selection: str) -> str:
        """RAG 파이프라인 설정"""
        if not collection_selection:
            return "❌ 컬렉션을 선택하세요."
        
        try:
            # 컬렉션 이름 추출
            collection_name = collection_selection.split(" (")[0]
            
            self.rag_pipeline = RAGPipeline(self.vector_manager, self.model_manager)
            self.rag_pipeline.setup_chain(collection_name)
            self.current_collection = collection_name
            
            return f"✅ RAG 설정 완료: {collection_name}"
        except Exception as e:
            return f"❌ RAG 설정 실패: {str(e)}"
    
    def chat(
        self, 
        message: str, 
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        """채팅 응답"""
        if not message:
            return history, ""
        
        if not self.rag_pipeline:
            history.append((message, "❌ 먼저 모델을 로드하고 컬렉션을 선택해주세요."))
            return history, ""
        
        try:
            # 질의응답 (소스 포함)
            result = self.rag_pipeline.query_with_sources(message)
            answer = result["answer"]
            
            # 소스 정보 추가
            if result["sources"]:
                sources_text = "\n\n📎 **참고 문서:**\n"
                for src in result["sources"][:3]:  # 상위 3개만
                    sources_text += f"- {src['filename']}\n"
                answer += sources_text
            
            history.append((message, answer))
            return history, ""
        except Exception as e:
            history.append((message, f"❌ 오류: {str(e)}"))
            return history, ""
    
    def search_docs(self, query: str) -> str:
        """문서 검색"""
        if not query:
            return "검색어를 입력하세요."
        
        if not self.current_collection:
            return "❌ 먼저 컬렉션을 선택하세요."
        
        try:
            if not self.rag_pipeline:
                self.rag_pipeline = RAGPipeline(self.vector_manager, self.model_manager)
            
            results = self.rag_pipeline.search_similar_documents(
                query, 
                self.current_collection,
                k=5
            )
            
            if not results:
                return "검색 결과가 없습니다."
            
            output = "🔍 **검색 결과:**\n\n"
            for i, r in enumerate(results, 1):
                output += f"**{i}. {r['filename']}** (유사도: {r['score']:.4f})\n"
                output += f"```\n{r['preview']}\n```\n\n"
            
            return output
        except Exception as e:
            return f"❌ 검색 실패: {str(e)}"
    
    def import_model_file(self, file) -> str:
        """모델 파일 업로드"""
        if file is None:
            return "파일을 선택하세요."
        
        try:
            result = self.model_manager.import_model(file.name)
            return f"✅ 모델 업로드 완료: {Path(result).name}"
        except Exception as e:
            return f"❌ 업로드 실패: {str(e)}"


def create_app() -> gr.Blocks:
    """Gradio 앱 생성"""
    ui = DocumentAssistantUI()
    
    with gr.Blocks(title="📚 Document Assistant") as app:
        gr.Markdown("# 📚 Document Assistant")
        gr.Markdown("문서 기반 RAG 질의응답 시스템 - 기획서, 설계서 분석 및 작성 지원")
        
        with gr.Row():
            # 왼쪽 패널: 설정
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 설정")
                
                # 폴더 인덱싱
                with gr.Group():
                    gr.Markdown("### 📁 문서 폴더")
                    folder_input = gr.Textbox(
                        label="폴더 경로",
                        placeholder="C:/Documents/Projects",
                        info="인덱싱할 문서 폴더 경로"
                    )
                    collection_input = gr.Textbox(
                        label="컬렉션 이름 (선택)",
                        placeholder="my_project",
                        info="비워두면 폴더 이름 사용"
                    )
                    scan_btn = gr.Button("🔍 스캔 및 인덱싱", variant="primary")
                    index_status = gr.Textbox(
                        label="인덱싱 상태",
                        lines=8,
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                
                # 모델 선택
                with gr.Group():
                    gr.Markdown("### 🤖 LLM 모델")
                    model_dropdown = gr.Dropdown(
                        choices=ui.get_model_choices(),
                        label="모델 선택",
                        info="data/models/ 폴더의 GGUF 파일"
                    )
                    refresh_models_btn = gr.Button("🔄 새로고침")
                    load_model_btn = gr.Button("📥 모델 로드", variant="primary")
                    model_status = gr.Textbox(
                        label="모델 상태",
                        interactive=False
                    )
                    
                    with gr.Accordion("모델 업로드", open=False):
                        model_upload = gr.File(
                            label="GGUF 파일 업로드",
                            file_types=[".gguf", ".bin"]
                        )
                        upload_btn = gr.Button("업로드")
                        upload_status = gr.Textbox(label="업로드 상태", interactive=False)
                
                # 컬렉션 선택
                with gr.Group():
                    gr.Markdown("### 📂 컬렉션")
                    collection_dropdown = gr.Dropdown(
                        choices=ui.get_collection_choices(),
                        label="활성 컬렉션 선택"
                    )
                    refresh_collections_btn = gr.Button("🔄 새로고침")
                    setup_rag_btn = gr.Button("⚡ RAG 활성화", variant="primary")
                    rag_status = gr.Textbox(label="RAG 상태", interactive=False)
            
            # 오른쪽 패널: 채팅
            with gr.Column(scale=2):
                gr.Markdown("## 💬 질의응답")
                
                chatbot = gr.Chatbot(
                    label="대화",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="질문 입력",
                        placeholder="예: 기획서 양식은 어떻게 되나요?",
                        scale=4
                    )
                    send_btn = gr.Button("전송", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ 대화 초기화")
                
                # 문서 검색
                with gr.Accordion("🔍 문서 검색", open=False):
                    search_input = gr.Textbox(
                        label="검색어",
                        placeholder="검색할 키워드 입력"
                    )
                    search_btn = gr.Button("검색")
                    search_results = gr.Markdown(label="검색 결과")
        
        # 이벤트 핸들러
        scan_btn.click(
            ui.scan_and_index,
            inputs=[folder_input, collection_input],
            outputs=[index_status]
        ).then(
            lambda: ui.get_collection_choices(),
            outputs=[collection_dropdown]
        )
        
        refresh_models_btn.click(
            lambda: gr.update(choices=ui.get_model_choices()),
            outputs=[model_dropdown]
        )
        
        load_model_btn.click(
            ui.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        upload_btn.click(
            ui.import_model_file,
            inputs=[model_upload],
            outputs=[upload_status]
        ).then(
            lambda: gr.update(choices=ui.get_model_choices()),
            outputs=[model_dropdown]
        )
        
        refresh_collections_btn.click(
            lambda: gr.update(choices=ui.get_collection_choices()),
            outputs=[collection_dropdown]
        )
        
        setup_rag_btn.click(
            ui.setup_rag,
            inputs=[collection_dropdown],
            outputs=[rag_status]
        )
        
        # 채팅 이벤트
        msg_input.submit(
            ui.chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        send_btn.click(
            ui.chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        # 검색 이벤트
        search_btn.click(
            ui.search_docs,
            inputs=[search_input],
            outputs=[search_results]
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
