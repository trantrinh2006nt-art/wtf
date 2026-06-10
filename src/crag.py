"""
crag.py — Corrective RAG Engine cho Video Retrieval.
Đánh giá chất lượng của top kết quả và tự động reformulate query nếu kết quả chưa đạt yêu cầu.
"""

import json
from config import GEMINI_MODEL, CRAG_THRESHOLD_CORRECT, CRAG_THRESHOLD_AMBIGUOUS
from utils import SearchResult, log_info, log_error

class CRAGEngine:
    def __init__(self, client):
        self.client = client
        self.model_name = GEMINI_MODEL

    def evaluate_results(self, query: str, top_results: list[SearchResult]) -> tuple[str, float]:
        """
        Đánh giá kết quả.
        Trả về: Verdict ("CORRECT", "AMBIGUOUS", "INCORRECT") và điểm trung bình.
        """
        if not top_results:
            return "INCORRECT", 0.0

        # Mô phỏng điểm tự tin (Confidence score) dựa trên top 3 kết quả
        scores = [r.score for r in top_results[:3]]
        avg_score = sum(scores) / len(scores)

        if avg_score >= CRAG_THRESHOLD_CORRECT:
            verdict = "CORRECT"
        elif avg_score >= CRAG_THRESHOLD_AMBIGUOUS:
            verdict = "AMBIGUOUS"
        else:
            verdict = "INCORRECT"

        return verdict, avg_score

    def reformulate_query(self, original_query: str, failed_results: list[SearchResult], attempt: int) -> str:
        """Sử dụng LLM để viết lại câu query nếu tìm kiếm lần đầu thất bại."""
        log_info("CRAG: Đang phân tích lỗi và tạo query mới...")
        
        failed_texts = "\n".join([f"- {r.text[:100]}..." for r in failed_results])
        
        prompt = f"""
Câu truy vấn gốc: "{original_query}"
Hệ thống đã thử tìm kiếm nhưng các kết quả trả về không đủ độ tin cậy:
{failed_texts}

Hãy viết lại câu truy vấn (reformulate query) bằng cách:
1. Trích xuất từ khóa cốt lõi nhất.
2. Thêm các từ đồng nghĩa phổ biến trong tiếng Việt.
3. Rút gọn, loại bỏ các từ vô nghĩa.

TRẢ VỀ DUY NHẤT 1 CÂU QUERY MỚI, KHÔNG GIẢI THÍCH THÊM.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            new_query = response.text.strip().replace('"', '')
            return new_query
        except Exception as e:
            log_error(f"Lỗi khi reformulate query: {e}")
            # Fallback nếu lỗi LLM
            return original_query + " chi tiết"