"""
HWP 문서 로더
olefile을 사용하여 HWP 파일에서 텍스트 추출
"""
import os
import zlib
import struct
from typing import List, Optional
from pathlib import Path

try:
    import olefile
except ImportError:
    olefile = None

from langchain_core.documents import Document


class HWPLoader:
    """HWP 파일 로더 (한글 v5 포맷)"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        if olefile is None:
            raise ImportError("olefile 패키지가 필요합니다: pip install olefile")
    
    def load(self) -> List[Document]:
        """HWP 파일에서 텍스트 추출"""
        try:
            text = self._extract_text()
            if text:
                return [Document(
                    page_content=text,
                    metadata={
                        "source": str(self.file_path),
                        "filename": self.file_path.name,
                        "file_type": "hwp"
                    }
                )]
            return []
        except Exception as e:
            print(f"HWP 로드 실패 ({self.file_path}): {e}")
            return []
    
    def _extract_text(self) -> str:
        """HWP 파일에서 텍스트 추출 (OLE 구조 파싱)"""
        if not olefile.isOleFile(str(self.file_path)):
            # HWP가 아닌 경우 (HWPX일 수 있음)
            return self._try_hwpx_extraction()
        
        texts = []
        
        with olefile.OleFileIO(str(self.file_path)) as ole:
            # 방법 1: PrvText 스트림에서 미리보기 텍스트 추출
            if ole.exists("PrvText"):
                try:
                    prv_text = ole.openstream("PrvText").read()
                    text = prv_text.decode('utf-16-le', errors='ignore')
                    text = text.replace('\x00', '').strip()
                    if text:
                        texts.append(text)
                except Exception:
                    pass
            
            # 방법 2: BodyText 섹션에서 본문 추출
            body_texts = self._extract_body_text(ole)
            if body_texts:
                texts.extend(body_texts)
        
        return '\n\n'.join(texts) if texts else ""
    
    def _extract_body_text(self, ole) -> List[str]:
        """BodyText 스트림에서 본문 텍스트 추출"""
        texts = []
        
        # BodyText 스토리지 내의 모든 Section 스트림 처리
        for entry in ole.listdir():
            if len(entry) >= 2 and entry[0] == "BodyText":
                stream_path = "/".join(entry)
                try:
                    data = ole.openstream(stream_path).read()
                    
                    # zlib 압축 해제 시도
                    try:
                        decompressed = zlib.decompress(data, -15)
                        text = self._parse_body_text(decompressed)
                    except zlib.error:
                        # 압축되지 않은 경우
                        text = self._parse_body_text(data)
                    
                    if text:
                        texts.append(text)
                except Exception as e:
                    continue
        
        return texts
    
    def _parse_body_text(self, data: bytes) -> str:
        """바이너리 데이터에서 텍스트 파싱"""
        # HWP 레코드 구조 파싱
        text_parts = []
        pos = 0
        
        while pos < len(data):
            if pos + 4 > len(data):
                break
            
            # 레코드 헤더 읽기 (4바이트)
            header = struct.unpack('<I', data[pos:pos+4])[0]
            tag_id = header & 0x3FF
            level = (header >> 10) & 0x3FF
            size = (header >> 20) & 0xFFF
            
            # 크기가 0xFFF면 다음 4바이트에서 실제 크기 읽기
            if size == 0xFFF:
                if pos + 8 > len(data):
                    break
                size = struct.unpack('<I', data[pos+4:pos+8])[0]
                pos += 8
            else:
                pos += 4
            
            # PARA_TEXT 태그 (tag_id = 67)에서 텍스트 추출
            if tag_id == 67 and pos + size <= len(data):
                try:
                    text_data = data[pos:pos+size]
                    # UTF-16LE로 디코딩
                    text = text_data.decode('utf-16-le', errors='ignore')
                    # 제어 문자 제거
                    text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
                    if text.strip():
                        text_parts.append(text.strip())
                except Exception:
                    pass
            
            pos += size
        
        return '\n'.join(text_parts)
    
    def _try_hwpx_extraction(self) -> str:
        """HWPX 형식 시도 (ZIP 기반)"""
        import zipfile
        import xml.etree.ElementTree as ET
        
        try:
            with zipfile.ZipFile(str(self.file_path), 'r') as zf:
                texts = []
                # Contents 폴더 내의 섹션 XML 파일 처리
                for name in zf.namelist():
                    if name.startswith('Contents/') and name.endswith('.xml'):
                        try:
                            content = zf.read(name).decode('utf-8')
                            root = ET.fromstring(content)
                            # 모든 텍스트 노드 추출
                            for elem in root.iter():
                                if elem.text and elem.text.strip():
                                    texts.append(elem.text.strip())
                        except Exception:
                            continue
                return '\n'.join(texts)
        except zipfile.BadZipFile:
            return ""
        except Exception:
            return ""


class HWPXLoader:
    """HWPX 파일 로더 (신규 XML 기반 포맷)"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    def load(self) -> List[Document]:
        """HWPX 파일에서 텍스트 추출"""
        import zipfile
        import xml.etree.ElementTree as ET
        
        try:
            texts = []
            with zipfile.ZipFile(str(self.file_path), 'r') as zf:
                for name in zf.namelist():
                    if 'section' in name.lower() and name.endswith('.xml'):
                        try:
                            content = zf.read(name).decode('utf-8')
                            root = ET.fromstring(content)
                            
                            # hp:t 태그에서 텍스트 추출 (HWPX 네임스페이스)
                            ns = {'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph'}
                            for t_elem in root.iter():
                                if t_elem.text and t_elem.text.strip():
                                    texts.append(t_elem.text.strip())
                        except Exception:
                            continue
            
            text = '\n'.join(texts)
            if text:
                return [Document(
                    page_content=text,
                    metadata={
                        "source": str(self.file_path),
                        "filename": self.file_path.name,
                        "file_type": "hwpx"
                    }
                )]
            return []
        except Exception as e:
            print(f"HWPX 로드 실패 ({self.file_path}): {e}")
            return []
