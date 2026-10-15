"""Download task models for PyGeoFetch."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


class RetryStrategy(str, Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"
    LINEAR = "linear"


class ChecksumAlgorithm(str, Enum):
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


class PostProcessAction(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_string(cls, s: str) -> "PostProcessAction":
        if ":" in s:
            parts = s.split(":", 1)
            action = parts[0].strip()
            param_str = parts[1].strip()
            try:
                import json

                params = json.loads(param_str)
            except Exception:
                params = {"value": param_str}
        else:
            action = s.strip()
            params = {}
        return cls(action=action, params=params)

    def __str__(self) -> str:
        if self.params:
            param_str = ":".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.action}:{param_str}"
        return self.action


class DownloadOptions(BaseModel):
    parallel: int = Field(default=4, ge=1, le=32)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.0)
    retry_strategy: str = "exponential_jitter"
    timeout_seconds: float = Field(default=300.0, gt=0)
    resume: bool = True
    verify_checksum: bool = False
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    on_failure: str = "skip"
    max_file_size_gb: Optional[float] = None
    chunk_size_mb: float = 16.0  # download chunk size in MB
    bandwidth_limit_mbps: Optional[float] = None  # speed cap (None = unlimited)
    priority: int = 0  # higher = higher priority
    notify_webhook: Optional[str] = None
    notify_email: Optional[str] = None
    bands: Optional[List[str]] = None
    post_process: List[PostProcessAction] = Field(default_factory=list)
    output_format: str = "original"
    overwrite: bool = False
    extract_archives: bool = True
    flatten_directory: bool = False


class DownloadResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    status: DownloadStatus = DownloadStatus.PENDING
    data_id: Optional[str] = None
    provider: Optional[str] = None
    output_path: Optional[Path] = None
    output_paths: List[Path] = Field(default_factory=list)  # always a list, never None
    bytes_downloaded: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    error_type: Optional[str] = None  # e.g. "network", "auth", "checksum"
    retries_used: int = 0
    checksum_verified: bool = False
    post_process_completed: List[str] = Field(default_factory=list)
    from_cache: bool = False
    attempt_number: int = 1
    # Set by AOI streaming (client.stream): share of the requested AOI the
    # streamed raster covers (1.0 = whole AOI; lower = AOI runs past scene edge).
    aoi_coverage: Optional[float] = None

    @property
    def success(self) -> bool:
        return self.status == DownloadStatus.COMPLETED

    @property
    def failed(self) -> bool:
        return self.status == DownloadStatus.FAILED

    @property
    def skipped(self) -> bool:
        return self.status == DownloadStatus.SKIPPED


class DownloadTask(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    data_ids: List[str] = Field(default_factory=list)
    provider: Optional[str] = None
    destination: Path = Path("./downloads")
    options: DownloadOptions = Field(default_factory=DownloadOptions)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[DownloadResult] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if r.failed)


class DownloadProgress(BaseModel):
    """Progress tracking model (separate from the logging display class)."""

    model_config = {"arbitrary_types_allowed": True}

    task_id: str = ""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    current_file: Optional[str] = None
    current_file_bytes: int = 0
    current_file_total: int = 0
    speed_bps: float = 0.0
    started_at: Optional[datetime] = None
    estimated_remaining_seconds: Optional[float] = None

    @property
    def percent_complete(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.completed_files / self.total_files * 100

    @property
    def is_complete(self) -> bool:
        return self.completed_files + self.failed_files >= self.total_files


DownloadResult.model_rebuild()
DownloadTask.model_rebuild()
