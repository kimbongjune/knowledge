"""
벡터 데이터베이스 관리자
ChromaDB 통합 및 문서 임베딩 관리
"""
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CHROMA_DB_PATH, EMBEDDING_MODEL


class VectorManager:
    """ChromaDB 벡터 데이터베이스 관리"""
    
    def __init__(self, persist_path: Optional[Path] = None, embedding_model: Optional[str] = None):
        self.persist_path = persist_path or CHROMA_DB_PATH
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        
        # GPU 사용 가능 여부 확인
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        
        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 컬렉션 캐시
        self._collections: Dict[str, Chroma] = {}
    
    def get_or_create_collection(self, collection_name: str) -> Chroma:
        """컬렉션 가져오기 또는 생성"""
        if collection_name not in self._collections:
            collection_path = self.persist_path / collection_name
            self._collections[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(collection_path)
            )
        return self._collections[collection_name]
    
    def add_documents(
        self, 
        documents: List[Document], 
        collection_name: str,
        batch_size: int = 100
    ) -> int:
        """
        문서를 벡터 DB에 추가
        
        Args:
            documents: 추가할 문서 목록
            collection_name: 컬렉션 이름
            batch_size: 배치 크기
        
        Returns:
            추가된 문서 수
        """
        if not documents:
            return 0
        
        collection = self.get_or_create_collection(collection_name)
        
        # 배치 단위로 추가
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            collection.add_documents(batch)
            total_added += len(batch)
        
        return total_added
    
    def remove_documents_by_source(self, source_paths: List[str], collection_name: str) -> int:
        """
        소스 경로 기준으로 문서 삭제
        
        Args:
            source_paths: 삭제할 파일 경로 목록
            collection_name: 컬렉션 이름
        
        Returns:
            삭제된 문서 수
        """
        if not source_paths:
            return 0
        
        collection = self.get_or_create_collection(collection_name)
        
        deleted_count = 0
        for source_path in source_paths:
            try:
                # 해당 소스의 모든 문서 삭제
                results = collection.get(where={"source": source_path})
                if results and results.get('ids'):
                    collection.delete(ids=results['ids'])
                    deleted_count += len(results['ids'])
            except Exception as e:
                print(f"문서 삭제 실패 ({source_path}): {e}")
        
        return deleted_count
    
    def similarity_search(
        self, 
        query: str, 
        collection_name: str, 
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            collection_name: 컬렉션 이름
            k: 반환할 문서 수
            filter_dict: 필터 조건
        
        Returns:
            관련 문서 목록
        """
        collection = self.get_or_create_collection(collection_name)
        
        if filter_dict:
            return collection.similarity_search(query, k=k, filter=filter_dict)
        return collection.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        collection_name: str, 
        k: int = 5
    ) -> List[tuple]:
        """유사도 점수와 함께 검색"""
        collection = self.get_or_create_collection(collection_name)
        return collection.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, collection_name: str, search_kwargs: Optional[Dict] = None):
        """LangChain Retriever 반환"""
        collection = self.get_or_create_collection(collection_name)
        kwargs = search_kwargs or {"k": 5}
        return collection.as_retriever(search_kwargs=kwargs)
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """컬렉션 통계 정보"""
        collection = self.get_or_create_collection(collection_name)
        try:
            count = collection._collection.count()
            return {
                "name": collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            return {
                "name": collection_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def list_collections(self) -> List[str]:
        """저장된 컬렉션 목록"""
        if not self.persist_path.exists():
            return []
        
        collections = []
        for item in self.persist_path.iterdir():
            if item.is_dir():
                collections.append(item.name)
        return collections

    def rename_collection(self, old_name: str, new_name: str) -> bool:
        """컬렉션 이름 변경 (폴더명 및 메타데이터 파일명 변경)"""
        old_path = self.persist_path / old_name
        new_path = self.persist_path / new_name
        
        if not old_path.exists():
            raise ValueError(f"컬렉션 '{old_name}'이 존재하지 않습니다.")
        if new_path.exists():
            raise ValueError(f"컬렉션 '{new_name}'이 이미 존재합니다.")
            
        # 메모리에서 제거 시도
        if old_name in self._collections:
            del self._collections[old_name]
            import gc
            gc.collect()
            
        try:
            # 1. 컬렉션 폴더명 변경
            old_path.rename(new_path)
            
            # 2. 메타데이터 파일명 변경
            from config.settings import METADATA_PATH
            old_meta = METADATA_PATH / f"{old_name}_metadata.json"
            new_meta = METADATA_PATH / f"{new_name}_metadata.json"
            
            if old_meta.exists():
                old_meta.rename(new_meta)
                
            return True
        except PermissionError:
            # 윈도우에서 파일이 사용 중일 경우
            raise OSError("컬렉션이 사용 중이어서 이름을 변경할 수 없습니다. 잠시 후 다시 시도하거나 서버를 재시작하세요.")
    
    def delete_collection(self, collection_name: str):
        """컬렉션 삭제"""
        import shutil
        
        collection_path = self.persist_path / collection_name
        if collection_path.exists():
            shutil.rmtree(collection_path)
        
        self._collections.pop(collection_name, None)
