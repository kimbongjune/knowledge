"""
RAG 파이프라인
LangChain 기반 검색 증강 생성
"""
from typing import List, Optional, Dict, Any, Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.vector.vector_manager import VectorManager
from src.llm.model_manager import ModelManager


# 한국어 프롬프트 템플릿
DEFAULT_PROMPT_TEMPLATE = """아래의 문맥(Context)을 참고하여 질문에 답변해주세요.
문맥에 관련 정보가 있으면 우선적으로 활용하고, 없으면 당신의 지식을 활용하여 답변해주세요.
문서 기반 답변인지 일반 지식 기반 답변인지 구분해서 알려주세요.

문맥(Context):
{context}

질문: {question}

답변:"""

DOCUMENT_ASSISTANT_PROMPT = """당신은 문서 분석 및 작성을 도와주는 AI 어시스턴트입니다.
기존 기획서, 설계서, 분석서를 참고하여 새로운 문서 작성을 지원합니다.

제공된 문서들을 기반으로 질문에 답변해주세요:
- 문맥에 관련 정보가 있으면 우선적으로 참조합니다
- 문맥에 정보가 부족하면 당신의 일반 지식을 활용하여 답변합니다
- 기존 문서의 양식이나 구조를 참고할 수 있습니다
- 유사한 프로젝트의 사례를 찾아 제안할 수 있습니다
- 문서 작성에 필요한 항목들을 안내할 수 있습니다

문맥(Context):
{context}

질문: {question}

답변:"""


class RAGPipeline:
    """검색 증강 생성 파이프라인"""
    
    def __init__(
        self, 
        vector_manager: VectorManager, 
        model_manager: ModelManager,
        prompt_template: Optional[str] = None
    ):
        self.vector_manager = vector_manager
        self.model_manager = model_manager
        self.prompt_template = prompt_template or DOCUMENT_ASSISTANT_PROMPT
        
        self._chain = None
        self._collection_name = None
    
    def setup_chain(
        self, 
        collection_name: str, 
        model_name: Optional[str] = None,
        k: int = 5
    ):
        """
        RAG 체인 초기화
        
        Args:
            collection_name: 검색할 컬렉션 이름
            model_name: 사용할 LLM 모델 (None이면 현재 로드된 모델)
            k: 검색할 문서 수
        """
        # LLM 로드
        llm = self.model_manager.get_llm(model_name)
        
        # Retriever 생성
        retriever = self.vector_manager.get_retriever(
            collection_name, 
            search_kwargs={"k": k}
        )
        
        # 프롬프트 설정
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # LCEL 체인 구성
        def format_docs(docs: List[Document]) -> str:
            return "\n\n---\n\n".join(
                f"[출처: {doc.metadata.get('filename', 'Unknown')}]\n{doc.page_content}" 
                for doc in docs
            )
        
        self._chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        self._collection_name = collection_name
        print(f"✅ RAG 체인 초기화 완료 (컬렉션: {collection_name})")
    
    def query(self, question: str) -> str:
        """
        질의응답 수행
        
        Args:
            question: 질문
        
        Returns:
            LLM 응답
        """
        if not self._chain:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다. setup_chain()을 먼저 호출하세요.")
        
        return self._chain.invoke(question)
    
    def query_with_sources(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        소스 문서와 함께 질의응답
        
        Args:
            question: 질문
            k: 검색할 문서 수
        
        Returns:
            {"answer": str, "sources": List[Dict]}
        """
        if not self._collection_name:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다.")
        
        # 관련 문서 검색
        docs = self.vector_manager.similarity_search(
            question, 
            self._collection_name, 
            k=k
        )
        
        # 컨텍스트 구성
        context = "\n\n---\n\n".join(
            f"[출처: {doc.metadata.get('filename', 'Unknown')}]\n{doc.page_content}" 
            for doc in docs
        )
        
        # LLM 응답 생성
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        llm = self.model_manager.get_llm()
        formatted_prompt = prompt.format(context=context, question=question)
        
        print(f"🔍 질문: {question}")
        print(f"📚 검색된 문서: {len(docs)}개")
        
        # Ollama는 max_tokens 대신 num_predict 사용 (모델 설정에서 지정)
        answer = llm.invoke(formatted_prompt)
        
        print(f"✅ 응답 생성 완료 ({len(answer)}자)")
        
        # 소스 정보 추출
        sources = []
        seen_sources = set()
        for doc in docs:
            source = doc.metadata.get('source', '')
            if source and source not in seen_sources:
                seen_sources.add(source)
                sources.append({
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "path": source,
                    "file_type": doc.metadata.get('file_type', 'unknown'),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def stream_query(self, question: str) -> Generator[str, None, None]:
        """스트리밍 응답 생성"""
        if not self._chain:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다.")
        
        for chunk in self._chain.stream(question):
            yield chunk
    
    def search_similar_documents(
        self, 
        query: str, 
        collection_name: Optional[str] = None,
        k: int = 10
    ) -> List[Dict]:
        """유사 문서 검색 (RAG 없이 검색만)"""
        coll = collection_name or self._collection_name
        if not coll:
            raise RuntimeError("컬렉션이 지정되지 않았습니다.")
        
        docs_with_scores = self.vector_manager.similarity_search_with_score(
            query, coll, k=k
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "filename": doc.metadata.get('filename', 'Unknown'),
                "path": doc.metadata.get('source', ''),
                "score": float(score),
                "preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        return results
    
    def set_prompt_template(self, template: str):
        """프롬프트 템플릿 변경"""
        self.prompt_template = template
        # 체인이 이미 초기화되어 있으면 다시 설정
        if self._collection_name:
            self.setup_chain(self._collection_name)
