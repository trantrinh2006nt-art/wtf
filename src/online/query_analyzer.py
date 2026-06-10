"""
query_analyzer.py — Tầng 1: LLM phân rã query thành sub-queries đa phương thức.

Sử dụng Gemini Flash để phân rã query người dùng thành:
- visual_query:   tìm kiếm hình ảnh (SigLIP2)
- temporal_query:  hành động / chuyển động (InternVideo2)
- audio_query:    âm thanh môi trường (CLAP)
- ocr_query:      chữ trên màn hình (VietOCR)
- text_query:     lời nói phiên âm (PhoWhisper ASR)
- entities:       thực thể có tên (người, địa danh, tổ chức)
- query_type:     phân loại query → điều chỉnh trọng số retrieval

Hỗ trợ:
- Entity Grounding: bổ sung tri thức cho thực thể Việt Nam
- HyDE: sinh hypothetical document để cải thiện dense retrieval
- Fallback: nếu Gemini lỗi → trả về query gốc cho tất cả sub-queries
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    AnalyzedQuery, log_step, log_success, log_warning, log_error, log_info, timer
)

try:
    from google import genai
except ImportError:
    genai = None
    log_warning("google-genai chưa cài. Chạy: pip install google-genai")


class QueryAnalyzer:
    """
    Tầng 1 — LLM Query Decomposition.
    
    Phân rã câu hỏi người dùng thành nhiều sub-queries chuyên biệt
    cho từng modality, giúp retrieval chính xác hơn so với dùng
    nguyên query gốc cho mọi index.
    """

    def __init__(self):
        """Khởi tạo Gemini client."""
        self.client = None
        self.model_name = GEMINI_MODEL

        if genai is None:
            log_warning("QueryAnalyzer: google-genai không khả dụng, sẽ dùng fallback")
            return

        if not GEMINI_API_KEY:
            log_warning("QueryAnalyzer: GEMINI_API_KEY chưa cấu hình, sẽ dùng fallback")
            return

        try:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            log_success(f"QueryAnalyzer: Gemini client khởi tạo ({self.model_name})")
        except Exception as e:
            log_error(f"QueryAnalyzer: Lỗi khởi tạo Gemini client: {e}")
            self.client = None

    # ================================================================
    # PUBLIC: analyze()
    # ================================================================

    @timer
    def analyze(self, query: str) -> AnalyzedQuery:
        """
        Phân rã query thành AnalyzedQuery đa phương thức.

        Args:
            query: Câu hỏi từ người dùng (tiếng Việt hoặc tiếng Anh)

        Returns:
            AnalyzedQuery chứa sub-queries cho từng modality
        """
        log_step("Query Analyzer", f"Phân rã query: '{query}'")

        # Fallback nếu không có Gemini
        if self.client is None:
            log_warning("Không có Gemini → dùng fallback (query gốc cho mọi modality)")
            return self._fallback(query)

        try:
            # Bước 1: Gọi Gemini phân rã query
            analyzed = self._decompose_query(query)

            # Bước 2: Entity Grounding — bổ sung tri thức cho thực thể
            if analyzed.entities:
                log_info(f"Phát hiện {len(analyzed.entities)} thực thể: {analyzed.entities}")
                analyzed.entity_knowledge = self._entity_grounding(analyzed.entities)
                if analyzed.entity_knowledge:
                    log_success("Entity grounding hoàn thành")
                    # Bổ sung tri thức vào visual_query để cải thiện visual search
                    if analyzed.query_type == "entity_specific" and analyzed.entity_knowledge:
                        knowledge_snippet = analyzed.entity_knowledge[:200]
                        analyzed.visual_query = f"{analyzed.visual_query}. {knowledge_snippet}"

            # Bước 3: HyDE — sinh hypothetical document
            if HYDE_ENABLED:
                analyzed.hyde_document = self._generate_hyde(query)
                if analyzed.hyde_document:
                    log_success(f"HyDE document: {len(analyzed.hyde_document)} ký tự")

            self._log_analysis(analyzed)
            return analyzed

        except Exception as e:
            log_error(f"Query analysis thất bại: {e}")
            return self._fallback(query)

    # ================================================================
    # PRIVATE: _decompose_query() — gọi Gemini phân rã
    # ================================================================

    def _decompose_query(self, query: str) -> AnalyzedQuery:
        """Gửi query đến Gemini Flash, yêu cầu phân rã thành JSON."""

        prompt = f"""Bạn là chuyên gia phân tích query cho hệ thống tìm kiếm video bằng chứng.
Hãy phân rã câu hỏi sau thành các thành phần tìm kiếm đa phương thức.

QUERY: "{query}"

Trả về JSON (KHÔNG có markdown code block, CHỈ JSON thuần):
{{
    "visual_query": "mô tả hình ảnh cần tìm (đối tượng, cảnh vật, màu sắc, bối cảnh)",
    "temporal_query": "mô tả hành động, chuyển động, sự kiện diễn ra theo thời gian",
    "audio_query": "mô tả âm thanh (tiếng nổ, nhạc, còi xe, tiếng người)",
    "ocr_query": "text/chữ có thể xuất hiện trên màn hình (banner, biển báo, phụ đề)",
    "text_query": "từ khóa/nội dung lời nói có thể được phiên âm trong ASR transcript",
    "entities": ["danh sách thực thể: tên người, địa danh, tổ chức, sự kiện cụ thể"],
    "query_type": "entity_specific|action|audio|general"
}}

QUY TẮC:
- visual_query: Mô tả CỤ THỂ những gì mắt nhìn thấy trong video
- temporal_query: Tập trung vào HÀNH ĐỘNG, CHUYỂN ĐỘNG, DIỄN BIẾN
- audio_query: Mô tả ÂM THANH đặc trưng (để trống "" nếu không liên quan)
- ocr_query: Text CÓ THỂ HIỂN THỊ trên màn hình (để trống "" nếu không liên quan)
- text_query: Từ khóa phục vụ tìm kiếm trong bản phiên âm tiếng Việt (ASR)
- entities: CHỈ liệt kê thực thể CÓ TÊN RIÊNG (proper nouns)
- query_type: 
  + "entity_specific" nếu query nhắc đến người/địa danh/tổ chức cụ thể
  + "action" nếu query mô tả hành động/sự kiện
  + "audio" nếu query tập trung vào âm thanh
  + "general" cho các trường hợp còn lại

Ví dụ: query "Tìm đoạn Thủ tướng Phạm Minh Chính phát biểu về biến đổi khí hậu"
→ visual_query: "người đàn ông mặc vest đứng trên bục phát biểu, hội trường lớn, quốc kỳ"
→ temporal_query: "người phát biểu, cử chỉ tay, chuyển slide"
→ audio_query: "giọng nói nam giới, tiếng micro, vỗ tay"
→ ocr_query: "Phạm Minh Chính, biến đổi khí hậu, climate change"
→ text_query: "Thủ tướng Phạm Minh Chính biến đổi khí hậu phát biểu"
→ entities: ["Phạm Minh Chính"]
→ query_type: "entity_specific"
"""

        response = self._call_gemini(prompt)
        if not response:
            return self._fallback(query)

        return self._parse_decomposition(query, response)

    # ================================================================
    # PRIVATE: _parse_decomposition() — parse JSON từ Gemini
    # ================================================================

    def _parse_decomposition(self, query: str, response_text: str) -> AnalyzedQuery:
        """Parse JSON response từ Gemini thành AnalyzedQuery."""
        try:
            # Xử lý trường hợp Gemini trả về markdown code block
            cleaned = response_text.strip()

            # Loại bỏ ```json ... ``` wrapper nếu có
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
            if json_match:
                cleaned = json_match.group(1).strip()

            # Tìm JSON object trong response
            brace_start = cleaned.find('{')
            brace_end = cleaned.rfind('}')
            if brace_start != -1 and brace_end != -1:
                cleaned = cleaned[brace_start:brace_end + 1]

            data = json.loads(cleaned)

            # Validate query_type
            valid_types = {"entity_specific", "action", "audio", "general"}
            query_type = data.get("query_type", "general")
            if query_type not in valid_types:
                query_type = "general"

            # Build AnalyzedQuery
            analyzed = AnalyzedQuery(
                original_query=query,
                visual_query=data.get("visual_query", query) or query,
                temporal_query=data.get("temporal_query", query) or query,
                audio_query=data.get("audio_query", "") or "",
                ocr_query=data.get("ocr_query", "") or "",
                text_query=data.get("text_query", query) or query,
                entities=data.get("entities", []) or [],
                query_type=query_type,
            )

            log_success("Parse JSON decomposition thành công")
            return analyzed

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log_warning(f"Parse JSON thất bại ({e}), dùng fallback")
            return self._fallback(query)

    # ================================================================
    # PRIVATE: _entity_grounding() — bổ sung tri thức cho thực thể
    # ================================================================

    @timer
    def _entity_grounding(self, entities: list) -> str:
        """
        Entity Grounding: dùng Gemini sinh tri thức bổ sung cho thực thể.
        
        Thay vì gọi Wikidata API (hay lỗi với thực thể Việt Nam),
        tận dụng kiến thức sẵn có trong LLM để mô tả ngoại hình,
        đặc điểm nhận dạng của thực thể → cải thiện visual retrieval.

        Args:
            entities: Danh sách tên thực thể

        Returns:
            Chuỗi mô tả tri thức bổ sung
        """
        if not entities:
            return ""

        entities_str = ", ".join(entities)
        prompt = f"""Bạn là chuyên gia về Việt Nam. Hãy cung cấp thông tin NGẮN GỌN cho các thực thể sau,
tập trung vào đặc điểm NHẬN DẠNG TRỰC QUAN (ngoại hình, trang phục thường thấy, bối cảnh xuất hiện).

Thực thể: {entities_str}

Yêu cầu:
- Mỗi thực thể tối đa 2-3 câu
- Tập trung vào: ngoại hình, đặc điểm nhận dạng qua hình ảnh
- Nếu là địa danh: mô tả cảnh quan đặc trưng
- Nếu là tổ chức: mô tả logo, trụ sở đặc trưng
- Nếu không biết thực thể: bỏ qua, không bịa

Trả về text thuần (KHÔNG JSON, KHÔNG markdown).
"""

        response = self._call_gemini(prompt)
        if response:
            # Cắt ngắn nếu quá dài (giới hạn 500 ký tự)
            return response.strip()[:500]
        return ""

    # ================================================================
    # PRIVATE: _generate_hyde() — sinh hypothetical document
    # ================================================================

    @timer
    def _generate_hyde(self, query: str) -> str:
        """
        HyDE: sinh hypothetical document để cải thiện dense retrieval.
        
        Ý tưởng: thay vì encode query ngắn → encode 1 đoạn văn giả lập
        giống transcript thật → embedding sẽ gần hơn với document thật
        trong vector space.

        Args:
            query: Câu hỏi gốc

        Returns:
            Đoạn văn giả lập phong cách bản tin/phiên âm tiếng Việt
        """
        prompt = f"""Hãy viết một đoạn phiên âm video tin tức tiếng Việt (khoảng 50-80 từ) 
mà có thể chứa câu trả lời cho câu hỏi sau:

Câu hỏi: "{query}"

Yêu cầu:
- Viết như TRANSCRIPT (bản phiên âm lời nói), KHÔNG phải bài báo
- Phong cách: bản tin thời sự / phóng sự VTV
- Bao gồm từ khóa liên quan tự nhiên
- KHÔNG ghi nguồn, KHÔNG ghi tiêu đề
- Chỉ trả về đoạn transcript, không giải thích gì thêm
"""

        response = self._call_gemini(prompt)
        if response:
            # Loại bỏ các ký tự đặc biệt không mong muốn
            cleaned = response.strip()
            # Xóa dấu ngoặc kép bao quanh nếu có
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            return cleaned[:500]  # Giới hạn 500 ký tự
        return ""

    # ================================================================
    # PRIVATE: _call_gemini() — gọi Gemini API với retry logic
    # ================================================================

    def _call_gemini(self, prompt: str) -> str:
        """
        Gọi Gemini API với retry logic.

        Args:
            prompt: Prompt gửi đến Gemini

        Returns:
            Response text hoặc "" nếu thất bại
        """
        if self.client is None:
            return ""

        import time as _time

        for attempt in range(API_RETRY_MAX):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                # Trích xuất text từ response
                if response and response.text:
                    return response.text

                log_warning(f"Gemini trả về response rỗng (attempt {attempt + 1})")

            except Exception as e:
                error_str = str(e).lower()

                # Rate limit → đợi lâu hơn
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    wait = API_RETRY_DELAY * (attempt + 2)
                    log_warning(f"Rate limited, đợi {wait:.1f}s...")
                    _time.sleep(wait)
                # Lỗi khác → retry nhanh
                else:
                    log_warning(f"Gemini API lỗi (attempt {attempt + 1}/{API_RETRY_MAX}): {e}")
                    _time.sleep(API_RETRY_DELAY)

        log_error(f"Gemini API thất bại sau {API_RETRY_MAX} lần thử")
        return ""

    # ================================================================
    # PRIVATE: _fallback() — trả về query gốc khi mọi thứ thất bại
    # ================================================================

    def _fallback(self, query: str) -> AnalyzedQuery:
        """
        Fallback: gán query gốc cho tất cả sub-queries.
        Đảm bảo pipeline không bao giờ crash dù Gemini lỗi.
        """
        log_info("Sử dụng fallback: query gốc → tất cả modalities")
        return AnalyzedQuery(
            original_query=query,
            visual_query=query,
            temporal_query=query,
            audio_query=query,
            ocr_query=query,
            text_query=query,
            entities=[],
            entity_knowledge="",
            hyde_document="",
            query_type="general",
        )

    # ================================================================
    # PRIVATE: _log_analysis() — log kết quả phân rã
    # ================================================================

    def _log_analysis(self, analyzed: AnalyzedQuery):
        """In kết quả phân rã ra console để debug."""
        log_info(f"  query_type:     {analyzed.query_type}")
        log_info(f"  visual_query:   {analyzed.visual_query[:80]}...")
        log_info(f"  temporal_query: {analyzed.temporal_query[:80]}...")
        log_info(f"  text_query:     {analyzed.text_query[:80]}...")
        if analyzed.audio_query:
            log_info(f"  audio_query:    {analyzed.audio_query[:80]}...")
        if analyzed.ocr_query:
            log_info(f"  ocr_query:      {analyzed.ocr_query[:80]}...")
        if analyzed.entities:
            log_info(f"  entities:       {analyzed.entities}")
