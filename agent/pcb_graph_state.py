from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class PCBGraphState(TypedDict, total=False):
    # input
    image_path: str
    conf: float
    imgsz: int
    max_det: int
    top_k: int
    prefer_yolo_conf: float
    rag_backend: str  # local or mcp

    # runtime output dirs
    output_dir: str
    crop_dir: str

    # graph intermediate states
    annotated_image_path: str
    detections: List[Dict[str, Any]]
    crops: List[Dict[str, Any]]
    vlm_results: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    rag_reports: Dict[str, str]

    # final
    report: str
    report_path: str

    # logs/errors
    messages: List[str]
    errors: List[str]
