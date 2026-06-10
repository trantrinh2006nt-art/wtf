"""
main.py — CLI entry point cho hệ thống Video Evidence Retrieval (AI Challenge 2026).

Commands:
  preprocess  - Tiền xử lý video offline (ASR, OCR, embedding, indexing)
  search      - Tìm kiếm bằng chứng video
  interactive - Chế độ REPL tương tác
  evaluate    - Đánh giá hiệu năng với ground truth
  server      - Khởi chạy web dashboard
  status      - Kiểm tra trạng thái hệ thống
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Đảm bảo import từ cùng thư mục src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import *
from utils import (
    SearchResult, log_step, log_success, log_warning,
    log_error, log_info, format_time, Colors,
)


# ============================================================
# ASCII BANNER
# ============================================================

BANNER = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗   ██╗███████╗██████╗                                   ║
║   ██║   ██║██╔════╝██╔══██╗                                  ║
║   ██║   ██║█████╗  ██████╔╝                                  ║
║   ╚██╗ ██╔╝██╔══╝  ██╔══██╗                                  ║
║    ╚████╔╝ ███████╗██║  ██║                                  ║
║     ╚═══╝  ╚══════╝╚═╝  ╚═╝                                  ║
║                                                              ║
║   VIDEO EVIDENCE RETRIEVAL — AI Challenge 2026               ║
║   Hybrid Multi-Modal Pipeline for Vietnamese News Video      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}"""


# ============================================================
# SYSTEM INFO
# ============================================================

def print_system_info():
    """In thông tin hệ thống: GPU, RAM, models, indexes."""
    import torch

    print(f"\n{Colors.BOLD}{'─' * 60}{Colors.END}")
    print(f"  {Colors.BOLD}SYSTEM INFORMATION{Colors.END}")
    print(f"{'─' * 60}")

    # Device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  {Colors.GREEN}GPU{Colors.END}     : {gpu_name} ({vram:.1f} GB VRAM)")
        vram_used = torch.cuda.memory_allocated() / 1024**2
        print(f"  {Colors.GREEN}VRAM{Colors.END}    : {vram_used:.0f} MB allocated")
    else:
        print(f"  {Colors.YELLOW}GPU{Colors.END}     : Không có (chạy trên CPU)")

    print(f"  {Colors.GREEN}Device{Colors.END}  : {DEVICE}")

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  {Colors.GREEN}RAM{Colors.END}     : {ram.used / 1024**3:.1f} / {ram.total / 1024**3:.1f} GB")
    except ImportError:
        pass

    # Python
    print(f"  {Colors.GREEN}Python{Colors.END}  : {sys.version.split()[0]}")
    print(f"  {Colors.GREEN}PyTorch{Colors.END} : {torch.__version__}")

    # Models
    print(f"\n  {Colors.BOLD}Models:{Colors.END}")
    print(f"    ASR           : {ASR_MODEL}")
    print(f"    Frame Encoder : {FRAME_ENCODER_MODEL}")
    print(f"    Video Encoder : {VIDEO_ENCODER_MODEL}")
    print(f"    Audio Model   : {AUDIO_MODEL}")
    print(f"    Text Embedding: {TEXT_EMBEDDING_MODEL}")
    print(f"    Reranker      : {AUTO_RERANKER_MODEL}")
    print(f"    LLM           : {GEMINI_MODEL}")

    # Indexes
    print(f"\n  {Colors.BOLD}Indexes:{Colors.END}")
    indexes = {
        "Text Dense": TEXT_DENSE_INDEX_PATH,
        "Text Sparse": TEXT_SPARSE_INDEX_DIR,
        "Visual": VISUAL_INDEX_PATH,
        "Temporal": TEMPORAL_INDEX_PATH,
        "Audio": AUDIO_INDEX_PATH,
        "Segments": SEGMENTS_PATH,
    }
    for name, path in indexes.items():
        exists = os.path.exists(path)
        icon = f"{Colors.GREEN}✓{Colors.END}" if exists else f"{Colors.RED}✗{Colors.END}"
        size_str = ""
        if exists and os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1024**2
            size_str = f" ({size_mb:.1f} MB)"
        print(f"    {icon} {name:14s}: {Path(path).name}{size_str}")

    # API Key
    api_status = f"{Colors.GREEN}✓ Configured{Colors.END}" if GEMINI_API_KEY else f"{Colors.RED}✗ Missing{Colors.END}"
    print(f"\n  {Colors.BOLD}API:{Colors.END}")
    print(f"    Gemini API Key: {api_status}")

    print(f"{'─' * 60}\n")


# ============================================================
# KẾT QUẢ TABLE
# ============================================================

def _print_results_table(results: list[SearchResult], query: str = ""):
    """In bảng kết quả đẹp ra console."""
    if not results:
        print(f"\n  {Colors.YELLOW}Không tìm thấy kết quả nào.{Colors.END}\n")
        return

    # Header
    print(f"\n{'═' * 80}")
    if query:
        print(f"  {Colors.BOLD}Query:{Colors.END} {Colors.CYAN}{query}{Colors.END}")
        print(f"  {Colors.DIM}{len(results)} kết quả{Colors.END}")
    print(f"{'═' * 80}")

    # Column headers
    print(
        f"  {Colors.BOLD}{'#':>3}  "
        f"{'Video':15s}  "
        f"{'Time':>8s}  "
        f"{'Score':>7s}  "
        f"{'Source':>8s}  "
        f"{'Text'}{Colors.END}"
    )
    print(f"  {'─' * 74}")

    for i, r in enumerate(results, 1):
        # Thời gian
        time_str = format_time(r.start_time)
        if r.end_time > r.start_time:
            time_str += f"→{format_time(r.end_time)}"

        # Video file (cắt ngắn nếu cần)
        video = r.video_file[:15] if r.video_file else "—"

        # Score bar
        score_pct = min(100, int(r.score * 100))
        if score_pct >= 70:
            score_color = Colors.GREEN
        elif score_pct >= 40:
            score_color = Colors.YELLOW
        else:
            score_color = Colors.RED
        score_str = f"{score_color}{r.score:.4f}{Colors.END}"

        # Source badge
        source_colors = {
            "fused": Colors.CYAN,
            "text": Colors.GREEN,
            "visual": Colors.HEADER,
            "audio": Colors.YELLOW,
            "text_dense": Colors.GREEN,
            "text_sparse": Colors.GREEN,
        }
        src_color = source_colors.get(r.source, Colors.DIM)
        source_str = f"{src_color}{r.source:>8s}{Colors.END}"

        # Text preview
        text = r.text[:40] + "..." if r.text and len(r.text) > 40 else (r.text or "—")

        print(
            f"  {Colors.BOLD}{i:>3}{Colors.END}  "
            f"{video:15s}  "
            f"{time_str:>8s}  "
            f"{score_str}  "
            f"{source_str}  "
            f"{Colors.DIM}{text}{Colors.END}"
        )

        # Component scores (chi tiết) cho top 3
        if i <= 3 and any([r.dense_score, r.sparse_score, r.visual_score, r.audio_score]):
            components = []
            if r.dense_score:
                components.append(f"dense={r.dense_score:.3f}")
            if r.sparse_score:
                components.append(f"sparse={r.sparse_score:.3f}")
            if r.visual_score:
                components.append(f"visual={r.visual_score:.3f}")
            if r.audio_score:
                components.append(f"audio={r.audio_score:.3f}")
            if r.temporal_score:
                components.append(f"temporal={r.temporal_score:.3f}")
            if r.rerank_score:
                components.append(f"rerank={r.rerank_score:.3f}")
            print(f"       {Colors.DIM}└─ {' | '.join(components)}{Colors.END}")

    print(f"{'═' * 80}\n")


# ============================================================
# COMMANDS
# ============================================================

def cmd_preprocess(args):
    """Command: tiền xử lý video offline."""
    print(BANNER)
    log_step("PREPROCESS", "Bắt đầu tiền xử lý offline")

    from pipeline_old import VERPipeline
    pipeline = VERPipeline(track="auto")

    stats = pipeline.preprocess(
        video_path=args.video,
        video_dir=args.dir,
    )

    # In thống kê
    print(f"\n{'═' * 50}")
    print(f"  {Colors.BOLD}THỐNG KÊ TIỀN XỬ LÝ{Colors.END}")
    print(f"{'═' * 50}")
    for key, value in stats.items():
        if key != "errors":
            print(f"  {key:25s}: {value}")
    if stats.get("errors"):
        print(f"\n  {Colors.RED}Errors:{Colors.END}")
        for err in stats["errors"]:
            print(f"    • {err['video']}: {err['error']}")
    print(f"{'═' * 50}\n")


def cmd_search(args):
    """Command: tìm kiếm bằng chứng video."""
    print(BANNER)

    query = args.query
    track = args.track
    top_k = args.top_k

    log_step("SEARCH", f'Track={track} | Top-K={top_k}')
    log_info(f'Query: "{query}"')

    from pipeline_old import VERPipeline
    pipeline = VERPipeline(track=track)

    start_time = time.perf_counter()
    results = pipeline.search(query, top_k=top_k)
    elapsed = time.perf_counter() - start_time

    _print_results_table(results, query=query)
    print(f"  {Colors.DIM}Thời gian tìm kiếm: {elapsed:.2f}s{Colors.END}\n")


def cmd_interactive(args):
    """Command: REPL loop tương tác."""
    print(BANNER)
    print_system_info()

    track = args.track
    log_step("INTERACTIVE MODE", f"Track: {track}")
    print(f"  {Colors.CYAN}Nhập query để tìm kiếm. Gõ 'quit' hoặc 'exit' để thoát.{Colors.END}")
    print(f"  {Colors.DIM}Lệnh: /track auto|semi, /topk N, /status, /help{Colors.END}\n")

    from pipeline_old import VERPipeline
    pipeline = VERPipeline(track=track)

    top_k = TOP_K_FINAL
    search_count = 0

    while True:
        try:
            prompt = f"{Colors.BOLD}{Colors.CYAN}VER [{pipeline.track}]>{Colors.END} "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {Colors.GREEN}Goodbye! 👋{Colors.END}\n")
            break

        if not user_input:
            continue

        # Lệnh thoát
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n  {Colors.GREEN}Goodbye! 👋{Colors.END}\n")
            break

        # Lệnh đặc biệt
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/track" and len(parts) > 1:
                new_track = parts[1]
                if new_track in ("auto", "semi"):
                    pipeline = VERPipeline(track=new_track)
                    print(f"  {Colors.GREEN}Đã chuyển sang track: {new_track}{Colors.END}")
                else:
                    print(f"  {Colors.RED}Track không hợp lệ. Dùng: auto, semi{Colors.END}")

            elif cmd == "/topk" and len(parts) > 1:
                try:
                    top_k = int(parts[1])
                    print(f"  {Colors.GREEN}Top-K = {top_k}{Colors.END}")
                except ValueError:
                    print(f"  {Colors.RED}Giá trị không hợp lệ{Colors.END}")

            elif cmd == "/status":
                status = pipeline.get_status()
                print(f"\n  {Colors.BOLD}Pipeline Status:{Colors.END}")
                print(f"  Track: {status['track']} | Device: {status['device']}")
                print(f"  Engines:")
                for name, active in status["engines"].items():
                    icon = f"{Colors.GREEN}✓{Colors.END}" if active else f"{Colors.RED}✗{Colors.END}"
                    print(f"    {icon} {name}")
                print(f"  Segments: {status['segment_count']}")
                print()

            elif cmd == "/help":
                print(f"""
  {Colors.BOLD}Lệnh:{Colors.END}
    /track auto|semi  — Chuyển track
    /topk N           — Đổi số kết quả (hiện tại: {top_k})
    /status           — Kiểm tra trạng thái
    /help             — Hiện help
    quit/exit         — Thoát
""")
            else:
                print(f"  {Colors.RED}Lệnh không hợp lệ. Gõ /help{Colors.END}")
            continue

        # Tìm kiếm
        search_count += 1
        print(f"\n  {Colors.DIM}[Search #{search_count}]{Colors.END}")

        try:
            start_time = time.perf_counter()
            results = pipeline.search(user_input, top_k=top_k)
            elapsed = time.perf_counter() - start_time

            _print_results_table(results, query=user_input)
            print(f"  {Colors.DIM}⏱ {elapsed:.2f}s{Colors.END}\n")

        except Exception as e:
            print(f"  {Colors.RED}Lỗi: {e}{Colors.END}\n")


def cmd_evaluate(args):
    """Command: đánh giá hiệu năng với ground truth."""
    print(BANNER)

    gt_path = args.ground_truth
    if not os.path.exists(gt_path):
        log_error(f"File ground truth không tồn tại: {gt_path}")
        return

    log_step("EVALUATE", f"Ground truth: {gt_path}")

    # Load ground truth
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)
    except Exception as e:
        log_error(f"Lỗi đọc ground truth: {e}")
        return

    from pipeline_old import VERPipeline
    from evaluation import EvaluationEngine

    track = args.track
    pipeline = VERPipeline(track=track)
    evaluator = EvaluationEngine()

    # Format: list of {query, segments: [{start, end, text, aspects}]}
    all_ap_scores = []
    all_rr_scores = []
    all_iou_scores = []

    queries = ground_truth_data if isinstance(ground_truth_data, list) else [ground_truth_data]

    for i, item in enumerate(queries, 1):
        query = item.get("query", "")
        gt_segments = item.get("segments", [])

        if not query:
            log_warning(f"Item #{i}: không có query, bỏ qua")
            continue

        log_step(f"Query {i}/{len(queries)}", f'"{query}"')

        # Tìm kiếm
        results = pipeline.search(query)

        # Đánh giá
        eval_result = evaluator.evaluate_pipeline(results, gt_segments)

        all_ap_scores.append(eval_result.get("map_at_k", 0))
        all_rr_scores.append(eval_result.get("mrr", 0))
        all_iou_scores.append(eval_result.get("avg_iou", 0))

        _print_results_table(results, query=query)

    # Tổng hợp
    if all_ap_scores:
        avg_map = sum(all_ap_scores) / len(all_ap_scores)
        avg_mrr = sum(all_rr_scores) / len(all_rr_scores)
        avg_iou = sum(all_iou_scores) / len(all_iou_scores)

        print(f"\n{'═' * 60}")
        print(f"  {Colors.BOLD}📊 TỔNG KẾT ĐÁNH GIÁ{Colors.END}")
        print(f"{'═' * 60}")
        print(f"  Số queries         : {len(all_ap_scores)}")
        print(f"  Track              : {track}")
        print(f"  {Colors.GREEN}Mean MAP@K{Colors.END}         : {avg_map:.4f}")
        print(f"  {Colors.GREEN}Mean MRR{Colors.END}           : {avg_mrr:.4f}")
        print(f"  {Colors.GREEN}Mean IoU{Colors.END}           : {avg_iou:.4f}")

        # Grade
        if avg_map >= 0.7:
            grade = f"{Colors.GREEN}🟢 XUẤT SẮC{Colors.END}"
        elif avg_map >= 0.5:
            grade = f"{Colors.YELLOW}🟡 TỐT{Colors.END}"
        elif avg_map >= 0.3:
            grade = f"{Colors.YELLOW}🟠 TRUNG BÌNH{Colors.END}"
        else:
            grade = f"{Colors.RED}🔴 CẦN CẢI THIỆN{Colors.END}"

        print(f"  Đánh giá           : {grade}")
        print(f"{'═' * 60}\n")


def cmd_server(args):
    """Command: khởi chạy web server."""
    print(BANNER)

    host = args.host
    port = args.port

    log_step("WEB SERVER", f"Khởi chạy tại http://{host}:{port}")

    try:
        from web.app import run_server
        run_server(host=host, port=port)
    except ImportError as e:
        log_error(f"Không thể import web server: {e}")
        log_info("Cài đặt: pip install fastapi uvicorn")
    except Exception as e:
        log_error(f"Lỗi khởi chạy server: {e}")


def cmd_status(args):
    """Command: kiểm tra trạng thái hệ thống."""
    print(BANNER)
    print_system_info()


# ============================================================
# MAIN
# ============================================================

def main():
    """Entry point chính cho CLI."""
    parser = argparse.ArgumentParser(
        description="VER — Video Evidence Retrieval (AI Challenge 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess --dir data/raw
  python main.py search --query "tai nạn giao thông" --track semi
  python main.py interactive
  python main.py evaluate --ground-truth eval/ground_truth.json
  python main.py server --port 8000
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực thi")

    # --- preprocess ---
    p_preprocess = subparsers.add_parser(
        "preprocess",
        help="Tiền xử lý video offline (ASR, OCR, embedding, indexing)",
    )
    p_preprocess.add_argument(
        "--video", type=str, default=None,
        help="Đường dẫn đến 1 file video cụ thể",
    )
    p_preprocess.add_argument(
        "--dir", type=str, default=None,
        help="Thư mục chứa video (mặc định: data/raw)",
    )
    p_preprocess.set_defaults(func=cmd_preprocess)

    # --- search ---
    p_search = subparsers.add_parser(
        "search",
        help="Tìm kiếm bằng chứng video",
    )
    p_search.add_argument(
        "--query", "-q", type=str, required=True,
        help="Câu truy vấn tìm kiếm",
    )
    p_search.add_argument(
        "--track", "-t", type=str, default="auto",
        choices=["auto", "semi"],
        help="Track xử lý: auto (nhanh <2s) hoặc semi (chính xác <15s)",
    )
    p_search.add_argument(
        "--top-k", "-k", type=int, default=TOP_K_FINAL,
        help=f"Số kết quả trả về (mặc định: {TOP_K_FINAL})",
    )
    p_search.set_defaults(func=cmd_search)

    # --- interactive ---
    p_interactive = subparsers.add_parser(
        "interactive",
        help="Chế độ REPL tương tác (gõ query, nhận kết quả)",
    )
    p_interactive.add_argument(
        "--track", "-t", type=str, default="semi",
        choices=["auto", "semi"],
        help="Track mặc định (có thể đổi runtime bằng /track)",
    )
    p_interactive.set_defaults(func=cmd_interactive)

    # --- evaluate ---
    p_evaluate = subparsers.add_parser(
        "evaluate",
        help="Đánh giá hiệu năng pipeline với ground truth",
    )
    p_evaluate.add_argument(
        "--ground-truth", "-g", type=str, required=True,
        help="Đường dẫn file ground truth JSON",
    )
    p_evaluate.add_argument(
        "--track", "-t", type=str, default="auto",
        choices=["auto", "semi"],
        help="Track dùng để tìm kiếm",
    )
    p_evaluate.set_defaults(func=cmd_evaluate)

    # --- server ---
    p_server = subparsers.add_parser(
        "server",
        help="Khởi chạy web dashboard (FastAPI + Uvicorn)",
    )
    p_server.add_argument(
        "--host", type=str, default=WEB_HOST,
        help=f"Host bind (mặc định: {WEB_HOST})",
    )
    p_server.add_argument(
        "--port", type=int, default=WEB_PORT,
        help=f"Port (mặc định: {WEB_PORT})",
    )
    p_server.set_defaults(func=cmd_server)

    # --- status ---
    p_status = subparsers.add_parser(
        "status",
        help="Kiểm tra trạng thái hệ thống (GPU, models, indexes)",
    )
    p_status.set_defaults(func=cmd_status)

    # Parse
    args = parser.parse_args()

    if args.command is None:
        print(BANNER)
        parser.print_help()
        return

    try:
        args.func(args)
    except KeyboardInterrupt:
        print(f"\n\n  {Colors.YELLOW}Đã hủy bởi người dùng.{Colors.END}\n")
        sys.exit(0)
    except Exception as e:
        log_error(f"Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
