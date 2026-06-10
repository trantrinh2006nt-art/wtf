"""
verifier.py — Verifier Agent.
Bước kiểm tra cuối cùng (Post-retrieval verification) để lọc bỏ các kết quả bị ảo giác (hallucinated) 
trước khi trình bày cho người dùng.
"""

from config import GEMINI_MODEL
from utils import SearchResult, log_info, log_error
import time # Thêm import time

class VerifierAgent:
    def __init__(self, client):
        self.client = client
        self.model_name = GEMINI_MODEL

    def verify_results(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Loại bỏ các kết quả hoàn toàn không khớp logic với query."""
        if not candidates:
            return []
            
        log_info(f"VerifierAgent: Đang kiểm chứng chéo {len(candidates)} kết quả cuối...")
        verified_results = []
        
        # Để tiết kiệm thời gian và API, thường chỉ verify top 3-5
        for candidate in candidates:
            # Nếu điểm gốc vốn đã rất cao thì auto pass (tiết kiệm API)
            if candidate.score > 0.85:
                verified_results.append(candidate)
                continue
                
            prompt = f"""
Câu hỏi tìm kiếm bằng chứng: "{query}"
Nội dung đoạn video trích xuất được: "{candidate.text}"

Đoạn nội dung trên có thực sự chứa bằng chứng thỏa mãn câu hỏi không?
Chỉ trả lời "YES" hoặc "NO".
"""
            try:
                time.sleep(4.0)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                answer = response.text.strip().upper()
                
                # Nếu LLM xác nhận YES hoặc trả về câu trả lời có chứa YES
                if "YES" in answer:
                    verified_results.append(candidate)
                else:
                    log_info(f"Verifier loại bỏ kết quả ID: {candidate.segment_id} do thiếu bằng chứng logic.")
                    
            except Exception as e:
                log_error(f"Lỗi Verifier: {e}")
                # Nếu API lỗi, chấp nhận cho qua thay vì drop mất dữ liệu của người dùng
                verified_results.append(candidate)
                
        # Nếu filter gắt quá dẫn đến mảng rỗng, fallback trả về mảng ban đầu
        if not verified_results:
            log_info("Verifier loại quá nhiều kết quả, fallback giữ lại top-k gốc.")
            return candidates
            
        return verified_results