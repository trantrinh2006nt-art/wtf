"""
reranker.py — LVLM Reranker dùng Gemini 2.5 Flash.
Đọc danh sách kết quả từ tầng Retrieval và chấm điểm lại (Reranking) dựa trên ngữ nghĩa sâu.
"""

import json
from google import genai
from config import GEMINI_MODEL
from utils import SearchResult, log_info, log_error

class LVLMReranker:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = GEMINI_MODEL

    def rerank(self, query: str, candidates: list[SearchResult], video_dir: str = "") -> list[SearchResult]:
        if not candidates:
            return []

        log_info(f"LVLMReranker: Đang chấm điểm lại {len(candidates)} candidates...")
        
        # Tạo prompt cho Gemini
        prompt = f"""
Bạn là một chuyên gia đánh giá độ liên quan của video.
Câu truy vấn của người dùng: "{query}"

Dưới đây là danh sách các đoạn video (được biểu diễn bằng text ASR/OCR/Hành động). 
Hãy đánh giá độ liên quan của từng đoạn đối với câu truy vấn trên thang điểm từ 0.0 đến 1.0.
Trả về KẾT QUẢ DUY NHẤT LÀ ĐỊNH DẠNG JSON MẢNG, KHÔNG CÓ MARKDOWN HAY TEXT NÀO KHÁC.
Format: [{{"id": "segment_id", "score": 0.95}}, ...]

Danh sách:
"""
        for i, c in enumerate(candidates):
            text_info = c.text[:200] + "..." if len(c.text) > 200 else c.text
            prompt += f"- ID: {c.segment_id} | Text: {text_info}\n"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            
            # Xử lý text trả về để lấy JSON
            result_text = response.text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            elif result_text.startswith("```"):
                result_text = result_text[3:-3].strip()
                
            scores_data = json.loads(result_text)
            
            # Cập nhật điểm
            score_map = {item["id"]: item.get("score", 0.0) for item in scores_data}
            for candidate in candidates:
                if candidate.segment_id in score_map:
                    candidate.rerank_score = float(score_map[candidate.segment_id])
                    # Trộn điểm gốc với điểm của LLM (LLM chiếm trọng số cao hơn)
                    candidate.score = (candidate.score * 0.3) + (candidate.rerank_score * 0.7)
            
            # Sắp xếp lại theo điểm mới
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates

        except Exception as e:
            log_error(f"Lỗi khi gọi Gemini Reranker: {e}")
            return candidates # Fallback trả về nguyên gốc nếu lỗi