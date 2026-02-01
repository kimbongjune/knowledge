"""
LLM 모델 관리자
Ollama를 사용한 로컬 모델 관리 (검색/다운로드/삭제)
"""
import json
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass, asdict

from langchain_community.llms import Ollama


@dataclass
class ModelInfo:
    """모델 정보"""
    name: str
    size: str  # 예: "4.7 GB"
    modified: str  # 수정일
    family: str = ""
    parameter_size: str = ""
    is_vision: bool = False  # 비전 모델 여부


class ModelManager:
    """Ollama 기반 LLM 모델 관리"""
    
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # 비전 모델 이름 패턴
    VISION_MODEL_PATTERNS = ["llava", "bakllava", "moondream", "cogvlm"]
    
    def __init__(self, models_path: Optional[Path] = None):
        self._current_model: Optional[Ollama] = None
        self._current_model_name: Optional[str] = None
    
    def _check_ollama_running(self) -> bool:
        """Ollama 서버 실행 여부 확인"""
        try:
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _is_vision_model(self, model_name: str) -> bool:
        """비전 모델인지 확인"""
        name_lower = model_name.lower()
        return any(pattern in name_lower for pattern in self.VISION_MODEL_PATTERNS)
    
    def is_current_model_vision(self) -> bool:
        """현재 로드된 모델이 비전 모델인지 확인"""
        if not self._current_model_name:
            return False
        return self._is_vision_model(self._current_model_name)
    
    def _format_size(self, size_bytes: int) -> str:
        """바이트를 GB로 포맷"""
        if size_bytes >= 1024**3:
            return f"{size_bytes / (1024**3):.1f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / (1024**2):.1f} MB"
        else:
            return f"{size_bytes / 1024:.1f} KB"
    
    def list_installed_models(self) -> List[ModelInfo]:
        """Ollama에 설치된 모델 목록 (실제 설치된 것만)"""
        models = []
        try:
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                for m in data.get("models", []):
                    name = m.get("name", "")
                    details = m.get("details", {})
                    models.append(ModelInfo(
                        name=name,
                        size=self._format_size(m.get("size", 0)),
                        modified=m.get("modified_at", "")[:10],  # 날짜만
                        family=details.get("family", ""),
                        parameter_size=details.get("parameter_size", ""),
                        is_vision=self._is_vision_model(name)
                    ))
        except Exception as e:
            print(f"모델 목록 조회 실패: {e}")
        return models
    
    def search_models(self, query: str) -> List[Dict]:
        """
        Ollama 라이브러리에서 모델 검색
        Note: Ollama는 공식 검색 API가 없어서 인기 모델 목록 제공
        """
        # 인기/추천 모델 목록 (수동 관리)
        popular_models = [
            {"name": "qwen2.5:7b", "description": "Qwen 2.5 7B - 빠른 한국어 지원", "size": "4.7 GB"},
            {"name": "qwen2.5:14b", "description": "Qwen 2.5 14B - 균형잡힌 성능", "size": "9.0 GB"},
            {"name": "qwen2.5:32b", "description": "Qwen 2.5 32B - 고성능", "size": "19 GB"},
            {"name": "qwen2.5:72b", "description": "Qwen 2.5 72B - 최고 성능", "size": "41 GB"},
            {"name": "llama3.2:3b", "description": "Llama 3.2 3B - 경량 모델", "size": "2.0 GB"},
            {"name": "llama3.2:11b", "description": "Llama 3.2 11B Vision - 이미지 분석", "size": "7.9 GB", "vision": True},
            {"name": "llama3.2:90b", "description": "Llama 3.2 90B Vision - 고성능 비전", "size": "55 GB", "vision": True},
            {"name": "llava:7b", "description": "LLaVA 7B - 이미지 분석 기본", "size": "4.5 GB", "vision": True},
            {"name": "llava:13b", "description": "LLaVA 13B - 이미지 분석 성능", "size": "8.0 GB", "vision": True},
            {"name": "llava:34b", "description": "LLaVA 34B - 고성능 이미지 분석", "size": "20 GB", "vision": True},
            {"name": "gemma2:9b", "description": "Gemma 2 9B - Google 경량 모델", "size": "5.4 GB"},
            {"name": "gemma2:27b", "description": "Gemma 2 27B - Google 고성능", "size": "16 GB"},
            {"name": "mistral:7b", "description": "Mistral 7B - 빠르고 효율적", "size": "4.1 GB"},
            {"name": "codellama:7b", "description": "Code Llama 7B - 코드 특화", "size": "3.8 GB"},
            {"name": "deepseek-coder:6.7b", "description": "DeepSeek Coder - 코딩 전문", "size": "3.8 GB"},
        ]
        
        # 설치된 모델 목록
        installed = [m.name for m in self.list_installed_models()]
        
        # 검색어로 필터링
        query_lower = query.lower()
        results = []
        for model in popular_models:
            if query_lower in model["name"].lower() or query_lower in model.get("description", "").lower():
                model["installed"] = model["name"] in installed
                results.append(model)
        
        # 검색어가 비어있으면 전체 반환
        if not query:
            for model in popular_models:
                model["installed"] = model["name"] in installed
            return popular_models
        
        return results
    
    def pull_model_stream(self, model_name: str) -> Generator[Dict, None, None]:
        """모델 다운로드 (스트리밍 진행률)"""
        try:
            response = requests.post(
                f"{self.OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=3600  # 1시간 타임아웃
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        
                        progress = 0
                        if total > 0:
                            progress = int(completed / total * 100)
                        
                        yield {
                            "status": status,
                            "progress": progress,
                            "completed": completed,
                            "total": total
                        }
                        
                        if status == "success":
                            break
                    except:
                        pass
                        
        except Exception as e:
            yield {"status": "error", "message": str(e)}
    
    def delete_model(self, model_name: str) -> bool:
        """모델 삭제"""
        try:
            response = requests.delete(
                f"{self.OLLAMA_BASE_URL}/api/delete",
                json={"name": model_name},
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            print(f"모델 삭제 실패: {e}")
            return False
    
    def load_model(self, model_name: str, **kwargs) -> Ollama:
        """모델 로드"""
        if not self._check_ollama_running():
            raise RuntimeError("Ollama가 실행되고 있지 않습니다. Ollama를 먼저 실행해주세요.")
        
        # 이미 같은 모델이 로드되어 있으면 재사용
        if self._current_model_name == model_name and self._current_model:
            return self._current_model
        
        # LangChain Ollama 초기화
        self._current_model = Ollama(
            base_url=self.OLLAMA_BASE_URL,
            model=model_name,
            temperature=0.7,
            num_ctx=4096,
        )
        self._current_model_name = model_name
        
        print(f"✅ 모델 로드 완료: {model_name}")
        return self._current_model
    
    def get_llm(self, model_name: Optional[str] = None) -> Ollama:
        """현재 로드된 LLM 또는 지정된 모델 반환"""
        if model_name:
            return self.load_model(model_name)
        
        if self._current_model:
            return self._current_model
        
        raise RuntimeError("로드된 모델이 없습니다.")
    
    def unload_model(self):
        """현재 모델 언로드"""
        if self._current_model:
            self._current_model = None
            self._current_model_name = None
            print("🔄 모델 언로드 완료")
    
    def get_current_model_info(self) -> Optional[Dict]:
        """현재 로드된 모델 정보"""
        if not self._current_model_name:
            return None
        
        return {
            "name": self._current_model_name,
            "backend": "ollama",
            "is_vision": self._is_vision_model(self._current_model_name)
        }
    
    def is_current_model_vision(self) -> bool:
        """현재 모델이 비전 모델인지"""
        return self._current_model_name and self._is_vision_model(self._current_model_name)
