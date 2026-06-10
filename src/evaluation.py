"""
evaluation.py — Đánh giá hiệu năng hệ thống Video Evidence Retrieval.

Bao gồm các metrics:
- Temporal IoU: Đo độ trùng khớp thời gian giữa dự đoán và ground truth
- Aspect Recall: Đo mức độ bao phủ các khía cạnh nội dung
- MAP@K (Mean Average Precision): Đánh giá chất lượng xếp hạng
- MRR (Mean Reciprocal Rank): Đánh giá vị trí kết quả đúng đầu tiên
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Đảm bảo import từ cùng thư mục src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TOP_K_FINAL, TEMPORAL_WINDOW_SEC
from utils import (
    SearchResult, log_step, log_success, log_warning,
    log_error, log_info, format_time
)


class EvaluationEngine:
    """
    Engine đánh giá hiệu năng pipeline VER.

    Hỗ trợ đánh giá theo nhiều metric khác nhau:
    - Temporal IoU: Đo overlap thời gian
    - Aspect Recall: Đo bao phủ khía cạnh nội dung
    - MAP@K: Mean Average Precision
    - MRR: Mean Reciprocal Rank
    """

    def __init__(self):
        """Khởi tạo evaluation engine."""
        log_step("EvaluationEngine", "Khởi tạo module đánh giá hiệu năng")

    # ================================================================
    # METRIC: TEMPORAL IoU
    # ================================================================

    def iou_score(
        self,
        predicted_start: float,
        predicted_end: float,
        gt_start: float,
        gt_end: float
    ) -> float:
        """
        Tính Temporal IoU (Intersection over Union) giữa đoạn dự đoán
        và đoạn ground truth trên trục thời gian.

        IoU = intersection / union
        Giá trị nằm trong khoảng [0.0, 1.0].

        Args:
            predicted_start: Thời điểm bắt đầu dự đoán (giây).
            predicted_end: Thời điểm kết thúc dự đoán (giây).
            gt_start: Thời điểm bắt đầu ground truth (giây).
            gt_end: Thời điểm kết thúc ground truth (giây).

        Returns:
            Điểm IoU trong [0.0, 1.0].
        """
        # Xử lý trường hợp end = 0 (chỉ có start_time)
        if predicted_end <= predicted_start:
            predicted_end = predicted_start + TEMPORAL_WINDOW_SEC
        if gt_end <= gt_start:
            gt_end = gt_start + TEMPORAL_WINDOW_SEC

        # Tính intersection (phần giao)
        inter_start = max(predicted_start, gt_start)
        inter_end = min(predicted_end, gt_end)
        intersection = max(0.0, inter_end - inter_start)

        # Tính union (phần hợp)
        union = (
            (predicted_end - predicted_start)
            + (gt_end - gt_start)
            - intersection
        )

        # Tránh chia cho 0
        if union <= 0:
            return 0.0

        return intersection / union

    # ================================================================
    # METRIC: ASPECT RECALL
    # ================================================================

    def aspect_recall(
        self,
        predicted_results: list[SearchResult],
        ground_truth_aspects: list[str]
    ) -> float:
        """
        Tính Aspect Recall: Tỷ lệ khía cạnh nội dung được bao phủ.

        Mỗi aspect trong ground_truth_aspects được coi là "matched"
        nếu có ít nhất 1 kết quả dự đoán chứa aspect đó trong text.

        Args:
            predicted_results: Danh sách kết quả tìm kiếm.
            ground_truth_aspects: Danh sách khía cạnh cần bao phủ.

        Returns:
            Recall score trong [0.0, 1.0].
        """
        if not ground_truth_aspects:
            log_warning("Danh sách ground truth aspects rỗng")
            return 0.0

        # Gộp tất cả text từ kết quả dự đoán (lower-case để so sánh)
        all_predicted_text = " ".join(
            r.text.lower() for r in predicted_results if r.text
        )

        # Đếm số aspect được matched
        matched_count = 0
        matched_aspects = []
        unmatched_aspects = []

        for aspect in ground_truth_aspects:
            aspect_lower = aspect.lower().strip()
            if aspect_lower in all_predicted_text:
                matched_count += 1
                matched_aspects.append(aspect)
            else:
                unmatched_aspects.append(aspect)

        recall = matched_count / len(ground_truth_aspects)

        # Log chi tiết
        log_info(
            f"Aspect Recall: {matched_count}/{len(ground_truth_aspects)} "
            f"= {recall:.2%}"
        )
        if unmatched_aspects:
            log_warning(f"Aspects chưa bao phủ: {unmatched_aspects}")

        return recall

    # ================================================================
    # METRIC: MAP@K (Mean Average Precision at K)
    # ================================================================

    def _is_relevant(
        self,
        result: SearchResult,
        ground_truth_segments: list[dict],
        iou_threshold: float = 0.3
    ) -> bool:
        """
        Kiểm tra xem một kết quả có liên quan (relevant) hay không.
        Một kết quả được coi là relevant nếu IoU với bất kỳ segment GT nào >= threshold.

        Args:
            result: Kết quả cần kiểm tra.
            ground_truth_segments: Danh sách ground truth segments.
            iou_threshold: Ngưỡng IoU để coi là relevant.

        Returns:
            True nếu relevant, False nếu không.
        """
        for gt in ground_truth_segments:
            gt_start = gt.get("start", 0.0)
            gt_end = gt.get("end", 0.0)

            iou = self.iou_score(
                result.start_time, result.end_time,
                gt_start, gt_end
            )

            if iou >= iou_threshold:
                return True

        return False

    def average_precision(
        self,
        predicted_results: list[SearchResult],
        ground_truth_segments: list[dict],
        k: int = None
    ) -> float:
        """
        Tính Average Precision (AP) cho một query.

        AP = (1/R) * Σ(Precision@i * rel(i))
        R = số lượng relevant documents trong ground truth.

        Args:
            predicted_results: Danh sách kết quả (đã sắp xếp theo score giảm dần).
            ground_truth_segments: Danh sách ground truth segments.
            k: Giới hạn số kết quả xem xét (None = tất cả).

        Returns:
            AP score trong [0.0, 1.0].
        """
        if not ground_truth_segments:
            return 0.0

        # Giới hạn top-k nếu cần
        results = predicted_results[:k] if k else predicted_results
        total_relevant = len(ground_truth_segments)

        if total_relevant == 0:
            return 0.0

        cumulative_relevant = 0
        precision_sum = 0.0

        for i, result in enumerate(results, 1):
            if self._is_relevant(result, ground_truth_segments):
                cumulative_relevant += 1
                precision_at_i = cumulative_relevant / i
                precision_sum += precision_at_i

        # AP = tổng precision / số relevant thực tế
        ap = precision_sum / total_relevant
        return ap

    def mean_average_precision(
        self,
        predicted_results: list[SearchResult],
        ground_truth_segments: list[dict],
        k: int = None
    ) -> float:
        """
        Tính MAP@K (Mean Average Precision at K).

        Đây là trường hợp đơn giản hóa: 1 query → 1 AP.
        Với nhiều queries, gọi hàm này cho từng query rồi lấy trung bình.

        Args:
            predicted_results: Danh sách kết quả tìm kiếm.
            ground_truth_segments: Danh sách ground truth segments.
            k: Giới hạn top-K. Mặc định dùng TOP_K_FINAL từ config.

        Returns:
            MAP@K score trong [0.0, 1.0].
        """
        if k is None:
            k = TOP_K_FINAL

        ap = self.average_precision(predicted_results, ground_truth_segments, k)

        log_info(f"MAP@{k}: {ap:.4f}")
        return ap

    # ================================================================
    # METRIC: MRR (Mean Reciprocal Rank)
    # ================================================================

    def reciprocal_rank(
        self,
        predicted_results: list[SearchResult],
        ground_truth_segments: list[dict]
    ) -> float:
        """
        Tính Reciprocal Rank cho một query.

        RR = 1 / (vị trí kết quả đúng đầu tiên)
        Nếu không có kết quả đúng nào: RR = 0.

        Args:
            predicted_results: Danh sách kết quả (đã sắp xếp).
            ground_truth_segments: Danh sách ground truth segments.

        Returns:
            RR score trong [0.0, 1.0].
        """
        for i, result in enumerate(predicted_results, 1):
            if self._is_relevant(result, ground_truth_segments):
                return 1.0 / i

        return 0.0

    def mean_reciprocal_rank(
        self,
        predicted_results: list[SearchResult],
        ground_truth_segments: list[dict]
    ) -> float:
        """
        Tính MRR (Mean Reciprocal Rank).

        Trường hợp đơn giản hóa: 1 query → 1 RR.
        Với nhiều queries, gọi hàm này cho từng query rồi lấy trung bình.

        Args:
            predicted_results: Danh sách kết quả.
            ground_truth_segments: Danh sách ground truth segments.

        Returns:
            MRR score trong [0.0, 1.0].
        """
        rr = self.reciprocal_rank(predicted_results, ground_truth_segments)
        log_info(f"MRR: {rr:.4f}")
        return rr

    # ================================================================
    # ĐÁNH GIÁ TỔNG HỢP
    # ================================================================

    def evaluate_pipeline(
        self,
        results: list[SearchResult],
        ground_truth: list[dict]
    ) -> dict:
        """
        Chạy toàn bộ metrics và trả về dict tổng hợp.

        Ground truth format:
        [
            {
                "start": 10.5,
                "end": 25.0,
                "text": "Nội dung transcript",
                "aspects": ["khía cạnh 1", "khía cạnh 2"]
            },
            ...
        ]

        Args:
            results: Danh sách kết quả tìm kiếm từ pipeline.
            ground_truth: Danh sách ground truth segments.

        Returns:
            Dict chứa tất cả metric scores.
        """
        log_step("Evaluation", "═══ Bắt đầu đánh giá pipeline ═══")

        # 1. MAP@K
        map_score = self.mean_average_precision(results, ground_truth)

        # 2. MRR
        mrr_score = self.mean_reciprocal_rank(results, ground_truth)

        # 3. Tính IoU trung bình cho các kết quả relevant
        iou_scores = []
        for result in results:
            best_iou = 0.0
            for gt in ground_truth:
                iou = self.iou_score(
                    result.start_time, result.end_time,
                    gt.get("start", 0), gt.get("end", 0)
                )
                best_iou = max(best_iou, iou)
            iou_scores.append(best_iou)

        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

        # 4. Aspect Recall (gộp aspects từ tất cả ground truth)
        all_aspects = []
        for gt in ground_truth:
            all_aspects.extend(gt.get("aspects", []))

        aspect_recall_score = (
            self.aspect_recall(results, all_aspects) if all_aspects else 0.0
        )

        # 5. Precision@K (tỷ lệ kết quả relevant trong top-k)
        relevant_count = sum(
            1 for r in results if self._is_relevant(r, ground_truth)
        )
        precision_at_k = relevant_count / len(results) if results else 0.0

        # 6. Recall (tỷ lệ ground truth được tìm thấy)
        found_gt_count = 0
        for gt in ground_truth:
            gt_start = gt.get("start", 0)
            gt_end = gt.get("end", 0)
            for result in results:
                iou = self.iou_score(
                    result.start_time, result.end_time,
                    gt_start, gt_end
                )
                if iou >= 0.3:
                    found_gt_count += 1
                    break

        recall = found_gt_count / len(ground_truth) if ground_truth else 0.0

        # Tổng hợp kết quả
        eval_results = {
            "map_at_k": round(map_score, 4),
            "mrr": round(mrr_score, 4),
            "avg_iou": round(avg_iou, 4),
            "aspect_recall": round(aspect_recall_score, 4),
            "precision_at_k": round(precision_at_k, 4),
            "recall": round(recall, 4),
            "num_results": len(results),
            "num_ground_truth": len(ground_truth),
            "num_relevant": relevant_count,
        }

        log_success("Đánh giá hoàn tất")
        return eval_results

    # ================================================================
    # BÁO CÁO
    # ================================================================

    def generate_report(self, eval_results: dict) -> str:
        """
        Tạo báo cáo đánh giá hiệu năng với format đẹp.

        Args:
            eval_results: Dict kết quả từ evaluate_pipeline().

        Returns:
            Chuỗi báo cáo đã format.
        """
        separator = "═" * 56
        thin_sep = "─" * 56

        report_lines = [
            "",
            separator,
            "  📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG HỆ THỐNG VER",
            separator,
            "",
            f"  Số kết quả trả về     : {eval_results.get('num_results', 0)}",
            f"  Số ground truth       : {eval_results.get('num_ground_truth', 0)}",
            f"  Số kết quả relevant   : {eval_results.get('num_relevant', 0)}",
            "",
            thin_sep,
            "  RETRIEVAL METRICS",
            thin_sep,
            "",
            f"  MAP@K                 : {eval_results.get('map_at_k', 0):.4f}",
            f"  MRR                   : {eval_results.get('mrr', 0):.4f}",
            f"  Precision@K           : {eval_results.get('precision_at_k', 0):.4f}",
            f"  Recall                : {eval_results.get('recall', 0):.4f}",
            "",
            thin_sep,
            "  TEMPORAL & CONTENT METRICS",
            thin_sep,
            "",
            f"  Average IoU           : {eval_results.get('avg_iou', 0):.4f}",
            f"  Aspect Recall         : {eval_results.get('aspect_recall', 0):.4f}",
            "",
            separator,
        ]

        # Đánh giá tổng thể dựa trên MAP@K
        map_score = eval_results.get("map_at_k", 0)
        if map_score >= 0.7:
            grade = "🟢 XUẤT SẮC"
        elif map_score >= 0.5:
            grade = "🟡 TỐT"
        elif map_score >= 0.3:
            grade = "🟠 TRUNG BÌNH"
        else:
            grade = "🔴 CẦN CẢI THIỆN"

        report_lines.extend([
            f"  Đánh giá tổng thể    : {grade} (MAP@K = {map_score:.4f})",
            separator,
            "",
        ])

        report = "\n".join(report_lines)

        # In ra console
        print(report)

        return report

    def load_ground_truth(self, json_path: str) -> list[dict]:
        """
        Tải ground truth từ file JSON.

        Expected format:
        [
            {
                "start": 10.5,
                "end": 25.0,
                "text": "Nội dung",
                "aspects": ["aspect1", "aspect2"]
            }
        ]

        Args:
            json_path: Đường dẫn tới file ground truth JSON.

        Returns:
            Danh sách ground truth segments.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            log_success(f"Đã tải {len(data)} ground truth segments từ {json_path}")
            return data

        except FileNotFoundError:
            log_error(f"Không tìm thấy file ground truth: {json_path}")
            return []
        except json.JSONDecodeError as e:
            log_error(f"Lỗi parse JSON ground truth: {e}")
            return []
        except Exception as e:
            log_error(f"Lỗi tải ground truth: {e}")
            return []
