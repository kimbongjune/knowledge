"""
증분 업데이트 관리자
파일 변경 감지 및 메타데이터 관리
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import METADATA_PATH, SUPPORTED_EXTENSIONS


@dataclass
class ChangeSet:
    """변경된 파일 세트"""
    added: List[Path] = field(default_factory=list)
    modified: List[Path] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)
    
    def total_to_process(self) -> int:
        return len(self.added) + len(self.modified)


class IncrementalManager:
    """파일 변경 감지 및 증분 업데이트 관리"""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.metadata_file = METADATA_PATH / f"{collection_name}_metadata.json"
        self.metadata: Dict[str, Dict] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """저장된 메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        """메타데이터 저장"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """파일 해시값 계산 (MD5)"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def _get_file_info(self, file_path: Path) -> Dict:
        """파일 정보 수집"""
        stat = file_path.stat()
        return {
            "hash": self._get_file_hash(file_path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "indexed_at": datetime.now().isoformat()
        }
    
    def get_changes(self, folder_path: str) -> ChangeSet:
        """
        폴더 스캔하여 변경된 파일 감지
        
        Returns:
            ChangeSet: 추가/수정/삭제된 파일 목록
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
        
        # 현재 폴더의 모든 지원 파일 스캔
        current_files: Dict[str, Path] = {}
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in folder.rglob(f"*{ext}"):
                current_files[str(file_path)] = file_path
        
        # 이전에 인덱싱된 파일 목록
        indexed_files = set(self.metadata.keys())
        current_paths = set(current_files.keys())
        
        changes = ChangeSet()
        
        # 새로 추가된 파일
        for path_str in current_paths - indexed_files:
            changes.added.append(current_files[path_str])
        
        # 삭제된 파일
        for path_str in indexed_files - current_paths:
            # 해당 폴더에 있던 파일만 삭제 처리
            if path_str.startswith(str(folder)):
                changes.deleted.append(path_str)
        
        # 수정된 파일 (해시 비교)
        for path_str in current_paths & indexed_files:
            file_path = current_files[path_str]
            old_info = self.metadata.get(path_str, {})
            
            # 수정 시간 먼저 확인 (빠른 체크)
            current_mtime = file_path.stat().st_mtime
            if current_mtime != old_info.get("mtime"):
                # 해시로 실제 변경 확인
                current_hash = self._get_file_hash(file_path)
                if current_hash != old_info.get("hash"):
                    changes.modified.append(file_path)
        
        return changes
    
    def update_file_metadata(self, file_path: Path):
        """단일 파일 메타데이터 업데이트"""
        path_str = str(file_path)
        self.metadata[path_str] = self._get_file_info(file_path)
        self._save_metadata()
    
    def update_files_metadata(self, file_paths: List[Path]):
        """여러 파일 메타데이터 일괄 업데이트"""
        for file_path in file_paths:
            path_str = str(file_path)
            self.metadata[path_str] = self._get_file_info(file_path)
        self._save_metadata()
    
    def remove_file_metadata(self, file_paths: List[str]):
        """삭제된 파일의 메타데이터 제거"""
        for path_str in file_paths:
            self.metadata.pop(path_str, None)
        self._save_metadata()
    
    def get_indexed_count(self) -> int:
        """인덱싱된 파일 수 반환"""
        return len(self.metadata)
    
    def clear_metadata(self):
        """메타데이터 초기화"""
        self.metadata = {}
        if self.metadata_file.exists():
            self.metadata_file.unlink()
