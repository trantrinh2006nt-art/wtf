"""
semi_track.py — Tầng 3B: Track Bán Tự Động (Human-in-the-loop).

Sử dụng LLM/LVLM (Gemini Vision) để đánh giá chi tiết (Reranking)
và xác minh (Verification). Chấp nhận latency cao (<15s) đổi lấy
độ chính xác tối đa. Tích hợp cơ chế feedback và self-correction (CRAG).
"""

import sys
import time
import re
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils import (
    SearchResult,
    log_step, log_success, log_warning, log_error, log_info,
    timer, extract_frame, Colors
)
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

class SemiTrack:
    def __init__(self):
        """Khởi tạo SemiTrack với Gemini Client."""
        # Bỏ validate_api_key() ở đây để tránh lỗi khi init
        self.client = None
        if "genai" in sys.modules:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                log_warning(f"Lỗi khởi tạo genai.Client: {e}")
        self.model_id = GEMINI_VISION_MODEL

    @timer
    def process(self, query: str, candidates: List[SearchResult], video_dir: str) -> List[SearchResult]:
        """
        Quy trình xử lý đầy đủ cho track bán tự động:
        1. LVLM Reranking trên Top K candidates
        2. Verification cross-check
        3. Phân loại CRAG
        """
        log_info(f"Bắt đầu SemiTrack xử lý {len(candidates)} candidates")
        
        if not candidates:
            return []

        # 1. Lấy Top K Rerank
        top_candidates = candidates[:TOP_K_RERANK]
        
        # 2. LVLM Reranking (Chấm điểm dựa trên hình ảnh + văn bản)
        reranked_results = self._lvlm_rerank(query, top_candidates, video_dir)
        
        # 3. Reranking and Verification
        verified_results = self._verify(query, reranked_results)
        
        # Sắp xếp lại dựa trên điểm sau verify
        verified_results.sort(key=lambda x: x.score, reverse=True)
        
        return verified_results[:TOP_K_FINAL]

    def _lvlm_rerank(self, query: str, candidates: List[SearchResult], video_dir: str) -> List[SearchResult]:
        """Sử dụng Gemini Vision để chấm điểm độ liên quan."""
        log_step("LVLM RERANK", f"Chấm điểm {len(candidates)} đoạn video bằng Gemini Vision")
        
        base_video_dir = Path(video_dir) if video_dir else RAW_VIDEO_DIR
        results = []

        for i, cand in enumerate(candidates):
            retries = 0
            while retries < API_RETRY_MAX:
                try:
                    # Trích xuất khung hình
                    video_path = base_video_dir / cand.video_file
                    if not video_path.exists():
                        log_warning(f"Không tìm thấy video: {video_path}")
                        cand.rerank_score = cand.score
                        results.append(cand)
                        break

                    img = extract_frame(str(video_path), cand.start_time)
                    if img is None:
                        log_warning(f"Không thể cắt khung hình từ {cand.video_file} tại {cand.start_time}s")
                        cand.rerank_score = cand.score
                        results.append(cand)
                        break

                    # Prompt yêu cầu chấm điểm
                    prompt = f"""
Bạn là chuyên gia phân tích bằng chứng video. Hãy xem xét hình ảnh này và đoạn văn bản đi kèm để đánh giá mức độ liên quan so với truy vấn.

Truy vấn: "{query}"

Văn bản từ video (nếu có):
{cand.text}

Hãy đánh giá mức độ phù hợp trên thang điểm từ 0 đến 100.
Chỉ trả về MỘT con số duy nhất, không giải thích.
Ví dụ: 85
                    """
                    
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[img, prompt]
                    )
                    
                    # Parse điểm
                    text_res = response.text.strip()
                    match = re.search(r'\d+', text_res)
                    if match:
                        score_100 = float(match.group(0))
                        cand.rerank_score = min(max(score_100 / 100.0, 0.0), 1.0)
                        
                        # Kết hợp điểm retrieval và rerank
                        cand.score = 0.3 * cand.score + 0.7 * cand.rerank_score
                    else:
                        cand.rerank_score = cand.score

                    results.append(cand)
                    log_info(f"[{i+1}/{len(candidates)}] Score: {cand.rerank_score:.2f} - {cand.video_file}")
                    
                    time.sleep(API_CALL_DELAY)
                    break # Thành công, thoát vòng lặp retry

                except Exception as e:
                    retries += 1
                    log_warning(f"Lỗi gọi API cho {cand.video_file} (thử lại {retries}/{API_RETRY_MAX}): {e}")
                    time.sleep(API_RETRY_DELAY)
                    
            if retries == API_RETRY_MAX:
                cand.rerank_score = cand.score
                results.append(cand)

        # Sắp xếp lại theo điểm mới
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _verify(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Xác minh xem văn bản/kết quả có thực sự đáp ứng truy vấn hay không."""
        if not results:
            return results
            
        log_step("VERIFICATION", "Cross-check tính hợp lý của kết quả")
        
        # Thường chỉ verify top 3 để tiết kiệm thời gian
        verify_top = min(3, len(results))
        
        for i in range(verify_top):
            cand = results[i]
            if not cand.text:
                continue
                
            prompt = f"""
Xác minh xem đoạn văn bản sau có cung cấp bằng chứng cho truy vấn hay không.
Truy vấn: "{query}"
Đoạn văn: "{cand.text}"

Trích xuất thông tin:
1. Có bằng chứng rõ ràng không? (True/False)
2. Độ tin cậy (0.0-1.0)

Trả lời bằng JSON với định dạng chính xác:
{{"verified": true, "confidence": 0.85}}
            """
            
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                
                import json
                data = json.loads(response.text)
                is_verified = data.get("verified", False)
                confidence = data.get("confidence", 0.5)
                
                cand.metadata["verified"] = is_verified
                cand.metadata["verify_confidence"] = confidence
                
                # Phạt điểm nếu không verified
                if not is_verified:
                    cand.score *= 0.8
                    
            except Exception as e:
                log_warning(f"Lỗi verify cho kết quả {i+1}: {e}")
                
        return results

    def evaluate_and_correct(self, query: str, results: List[SearchResult]) -> Tuple[str, List[SearchResult], str]:
        """
        CRAG mechanism: Đánh giá chất lượng tập kết quả hiện tại.
        Trả về (verdict, results, reformulated_query)
        verdict: "CORRECT", "AMBIGUOUS", "INCORRECT"
        """
        if not results:
            return "INCORRECT", results, query

        top_score = results[0].score
        
        if top_score >= CRAG_THRESHOLD_CORRECT:
            return "CORRECT", results, ""
            
        if top_score >= CRAG_THRESHOLD_AMBIGUOUS:
            # AMBIGUOUS -> Cần reformulate query
            prompt = f"""
Tôi đang tìm kiếm video với truy vấn: "{query}"
Tuy nhiên, các kết quả hiện tại có vẻ mập mờ, không chắc chắn (điểm số vừa phải).
Hãy viết lại câu truy vấn này sao cho rõ ràng hơn, thêm các từ khóa đồng nghĩa hoặc tập trung vào chi tiết cụ thể để tìm kiếm chính xác hơn.
Chỉ trả về câu truy vấn mới, không giải thích.
"""
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                new_query = response.text.strip()
                log_warning(f"CRAG: Kết quả mập mờ. Viết lại truy vấn: {new_query}")
                return "AMBIGUOUS", results, new_query
            except Exception as e:
                log_error(f"CRAG lỗi sinh query mới: {e}")
                return "AMBIGUOUS", results, ""
                
        # INCORRECT -> Hoàn toàn không liên quan
        return "INCORRECT", results, ""

    def collect_feedback(self, results: List[SearchResult]) -> List[SearchResult]:
        """Tương tác thu thập phản hồi từ người dùng."""
        print(f"\n{Colors.BOLD}{Colors.HEADER}=== THU THẬP PHẢN HỒI NGƯỜI DÙNG ==={Colors.END}")
        
        for i, res in enumerate(results[:3]): # Chỉ hỏi top 3
            time_str = f"{res.start_time:.1f}s"
            print(f"\n{Colors.CYAN}Kết quả #{i+1} - {res.video_file} tại {time_str}{Colors.END}")
            if res.text:
                print(f"Nội dung: {res.text[:100]}...")
                
            feedback = input(f"Kết quả này có phù hợp không? (y/n/skip): ").strip().lower()
            
            if feedback == 'y':
                res.score *= 1.3
                res.metadata["user_feedback"] = "positive"
            elif feedback == 'n':
                res.score *= 0.5
                res.metadata["user_feedback"] = "negative"
            else:
                res.metadata["user_feedback"] = "skipped"
                
        # Sắp xếp lại
        results.sort(key=lambda x: x.score, reverse=True)
        return results
