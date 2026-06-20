from typing import Any, Dict, List
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

    # Qwen / VisualDiagnosisAgent call policy
    qwen_mode: str  # auto / always / never
    low_conf_threshold: float
    representative_per_type: int
    high_risk_types: List[str]

    # runtime output dirs
    output_dir: str
    crop_dir: str

    # graph intermediate states
    annotated_image_path: str
    detections: List[Dict[str, Any]]
    crops: List[Dict[str, Any]]
    visual_diagnoses: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    rag_reports: Dict[str, str]
    qwen_call_stats: Dict[str, Any]

    # backward compatibility with old UI/scripts if needed
    vlm_results: List[Dict[str, Any]]

    # final
    report: str
    report_path: str

    # logs/errors
    messages: List[str]
    errors: List[str]
