"""
문서 처리기
다양한 형식의 문서를 로드하고 청크로 분할
"""
import os
from pathlib import Path
from typing import List, Dict, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

from .hwp_loader import HWPLoader, HWPXLoader

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import SUPPORTED_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """문서 처리 총괄 클래스"""
    
    # 확장자별 로더 매핑
    LOADER_MAPPING = {
        '.hwp': HWPLoader,
        '.hwpx': HWPXLoader,
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.txt': TextLoader,
        '.md': TextLoader,
        '.csv': CSVLoader,
        '.xlsx': UnstructuredExcelLoader,
    }
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
    
    def scan_folder(self, folder_path: str) -> List[Path]:
        """폴더 내 지원되는 모든 문서 파일 스캔"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
        
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(folder.rglob(f"*{ext}"))
        
        return sorted(files)
    
    def load_document(self, file_path: Path) -> List[Document]:
        """단일 문서 로드"""
        ext = file_path.suffix.lower()
        
        if ext not in self.LOADER_MAPPING:
            print(f"지원하지 않는 파일 형식: {file_path}")
            return []
        
        loader_class = self.LOADER_MAPPING[ext]
        
        try:
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # 메타데이터 보강
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": ext[1:],  # 확장자에서 점 제거
                })
            
            return documents
        except Exception as e:
            print(f"문서 로드 실패 ({file_path}): {e}")
            return []
    
    def process_folder(
        self, 
        folder_path: str, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Document]:
        """
        폴더 내 모든 문서 처리
        
        Args:
            folder_path: 문서 폴더 경로
            progress_callback: 진행률 콜백 함수 (현재, 전체, 파일명)
        
        Returns:
            청크로 분할된 문서 목록
        """
        files = self.scan_folder(folder_path)
        total = len(files)
        
        if total == 0:
            return []
        
        all_documents = []
        
        for idx, file_path in enumerate(files, 1):
            if progress_callback:
                progress_callback(idx, total, file_path.name)
            
            docs = self.load_document(file_path)
            all_documents.extend(docs)
        
        # 청크로 분할
        if all_documents:
            chunked_docs = self.text_splitter.split_documents(all_documents)
            return chunked_docs
        
        return []
    
    def process_folder_parallel(
        self, 
        folder_path: str, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 4
    ) -> List[Document]:
        """병렬 처리로 폴더 내 모든 문서 처리"""
        files = self.scan_folder(folder_path)
        total = len(files)
        
        if total == 0:
            return []
        
        all_documents = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.load_document, f): f for f in files}
            
            for future in as_completed(futures):
                file_path = futures[future]
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, file_path.name)
                
                try:
                    docs = future.result()
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"처리 실패 ({file_path}): {e}")
        
        # 청크로 분할
        if all_documents:
            chunked_docs = self.text_splitter.split_documents(all_documents)
            return chunked_docs
        
        return []
    
    def get_supported_extensions(self) -> List[str]:
        """지원되는 파일 확장자 목록 반환"""
        return list(SUPPORTED_EXTENSIONS)
