"""
ProcessingPipeline — chain preprocessing, indices, and post-processing steps.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

if TYPE_CHECKING:
    from pygeofetch.processing.base import ProcessingResult

logger = logging.getLogger(__name__)


class ProcessingStep:
    """A single step in a processing pipeline."""

    def __init__(self, step_type: str, config: dict[str, Any]) -> None:
        self.step_type = step_type
        self.config = config

    def __repr__(self) -> str:
        return f"ProcessingStep({self.step_type!r}, {self.config})"


class PipelineRunResult:
    """Result of a complete pipeline execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.steps: list[dict[str, Any]] = []
        self.success = False
        self.duration_seconds = 0.0
        self.outputs: list[Path] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 2),
            "steps": self.steps,
            "outputs": [str(p) for p in self.outputs],
        }


class ProcessingPipeline:
    """
    Chainable geospatial processing pipeline.

    Define steps in Python or from YAML, then run them sequentially.
    Each step's output is passed as input to the next step.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()

        result = (
            client.pipeline("my-ndvi")
            .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
            .reproject(crs="EPSG:4326")
            .ndvi(red="B04", nir="B08")
            .cog()
            .run(input="scene.tif")
        )

    Or from YAML::

        client.pipeline.from_yaml("ndvi_workflow.yaml").run()
    """

    def __init__(self, name: str, engine=None) -> None:
        self.name = name
        self._engine = engine
        self._steps: list[ProcessingStep] = []
        self._schedule: str | None = None
        self._description: str = ""
        self._output_dir: Path | None = None

    # ── Builder methods ───────────────────────────────────────────────────

    def clip(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("clip", kwargs))
        return self

    def reproject(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("reproject", kwargs))
        return self

    def resample(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("resample", kwargs))
        return self

    def cloud_mask(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("cloud_mask", kwargs))
        return self

    def cloud_fill(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("cloud_fill", kwargs))
        return self

    def atmos(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("atmos", kwargs))
        return self

    def composite(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("composite", kwargs))
        return self

    def mosaic(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("mosaic", kwargs))
        return self

    def ndvi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("ndvi", kwargs))
        return self

    def evi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("evi", kwargs))
        return self

    def ndwi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("ndwi", kwargs))
        return self

    def ndbi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("ndbi", kwargs))
        return self

    def tct(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("tct", kwargs))
        return self

    def pca(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("pca", kwargs))
        return self

    def lst(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("lst", kwargs))
        return self

    def vectorize(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("vectorize", kwargs))
        return self

    def smooth(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("smooth", kwargs))
        return self

    def zonal_stats(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("zonal_stats", kwargs))
        return self

    def compress(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("compress", kwargs))
        return self

    def cog(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("cog", kwargs))
        return self

    def topo_correct(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("topo_correct", kwargs))
        return self

    def pansharpen(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("pansharpen", kwargs))
        return self

    def dnbr(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("dnbr", kwargs))
        return self

    def savi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("savi", kwargs))
        return self

    def nbr(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("nbr", kwargs))
        return self

    def mndwi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("mndwi", kwargs))
        return self

    def ndsi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("ndsi", kwargs))
        return self

    def ndmi(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("ndmi", kwargs))
        return self

    def albedo(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("albedo", kwargs))
        return self

    def band_math(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("band_math", kwargs))
        return self

    def buffer(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("buffer", kwargs))
        return self

    def centroids(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("centroids", kwargs))
        return self

    def add_geometry_metrics(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("add_geometry_metrics", kwargs))
        return self

    def despeckle(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("despeckle", kwargs))
        return self

    def calibrate(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("calibrate", kwargs))
        return self

    def flood_map(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("flood_map", kwargs))
        return self

    def coherence(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("coherence", kwargs))
        return self

    def tile(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("tile", kwargs))
        return self

    def stack(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("stack", kwargs))
        return self

    def texture(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("texture", kwargs))
        return self

    def regularize(self, **kwargs) -> ProcessingPipeline:
        self._steps.append(ProcessingStep("regularize", kwargs))
        return self

    def custom(self, func: Callable, **kwargs) -> ProcessingPipeline:
        """Add a custom processing function as a step."""
        self._steps.append(ProcessingStep("custom", {"func": func, **kwargs}))
        return self

    def set_schedule(self, cron: str) -> ProcessingPipeline:
        self._schedule = cron
        return self

    def output_dir(self, path: str | Path) -> ProcessingPipeline:
        self._output_dir = Path(path)
        return self

    # ── Execution ─────────────────────────────────────────────────────────

    def run(
        self,
        input: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> PipelineRunResult:
        """
        Execute all pipeline steps sequentially.

        Args:
            input:      Starting input file (required for first step).
            output_dir: Override output directory for all steps.

        Returns:
            :class:`PipelineRunResult` with step details and output paths.
        """
        run_result = PipelineRunResult(self.name)
        t_pipeline_start = time.time()
        out_dir = Path(output_dir or self._output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)

        current_input = Path(input) if input else None

        logger.info(f"Pipeline '{self.name}': {len(self._steps)} steps")

        for i, step in enumerate(self._steps, 1):
            step_t0 = time.time()
            step_record: dict[str, Any] = {
                "step": step.step_type,
                "status": "pending",
                "index": i,
            }

            logger.info(f"  Step {i}/{len(self._steps)}: {step.step_type}")

            try:
                result = self._execute_step(step, current_input, out_dir, i)
                if result.success and result.output_path:
                    current_input = result.output_path
                    run_result.outputs.append(result.output_path)
                step_record["status"] = "ok" if result.success else "failed"
                step_record["output"] = (
                    str(result.output_path) if result.output_path else None
                )
                step_record["duration"] = round(time.time() - step_t0, 2)
                if result.error:
                    step_record["error"] = result.error
            except Exception as exc:
                step_record["status"] = "error"
                step_record["error"] = str(exc)
                step_record["duration"] = round(time.time() - step_t0, 2)
                logger.error(f"  Step {i} ({step.step_type}) failed: {exc}")

            run_result.steps.append(step_record)

        run_result.success = all(s["status"] == "ok" for s in run_result.steps)
        run_result.duration_seconds = time.time() - t_pipeline_start

        logger.info(
            f"Pipeline '{self.name}' complete in {run_result.duration_seconds:.1f}s "
            f"({'success' if run_result.success else 'FAILED'})"
        )
        return run_result

    def _execute_step(
        self,
        step: ProcessingStep,
        current_input: Path | None,
        out_dir: Path,
        step_num: int,
    ) -> ProcessingResult:
        """Dispatch a step to the appropriate processing method."""
        if self._engine is None:
            msg = "Pipeline has no engine — create via client.pipeline('name')"
            raise RuntimeError(msg)

        cfg = dict(step.config)
        # Auto-set output dir
        if "output" not in cfg and current_input:
            cfg["output"] = str(out_dir / f"step{step_num:02d}_{step.step_type}.tif")

        # Auto-set input from pipeline state
        if current_input and "input" not in cfg:
            cfg["input"] = str(current_input)

        # Dispatch
        st = step.step_type
        pre = self._engine.preprocess
        idx = self._engine.indices
        post = self._engine.post
        sar = self._engine.sar

        dispatch = {
            # Preprocessing
            "atmos": lambda: pre.atmos(**cfg),
            "cloud_mask": lambda: pre.cloud_mask(**cfg),
            "cloud_fill": lambda: pre.cloud_fill(**cfg),
            "clip": lambda: pre.clip(**cfg),
            "reproject": lambda: pre.reproject(**cfg),
            "resample": lambda: pre.resample(**cfg),
            "mosaic": lambda: pre.mosaic(**cfg),
            "composite": lambda: pre.composite(**cfg),
            "tile": lambda: pre.tile(**cfg),
            "pansharpen": lambda: pre.pansharpen(**cfg),
            "topo_correct": lambda: pre.topo_correct(**cfg),
            # Indices
            "ndvi": lambda: idx.ndvi(**cfg),
            "evi": lambda: idx.evi(**cfg),
            "savi": lambda: idx.savi(**cfg),
            "ndwi": lambda: idx.ndwi(**cfg),
            "mndwi": lambda: idx.mndwi(**cfg),
            "ndbi": lambda: idx.ndbi(**cfg),
            "ndsi": lambda: idx.ndsi(**cfg),
            "ndmi": lambda: idx.ndmi(**cfg),
            "nbr": lambda: idx.nbr(**cfg),
            "dnbr": lambda: idx.dnbr(**cfg),
            "tct": lambda: idx.tct(**cfg),
            "pca": lambda: idx.pca(**cfg),
            "texture": lambda: idx.texture(**cfg),
            "lst": lambda: idx.lst(**cfg),
            "albedo": lambda: idx.albedo(**cfg),
            "band_math": lambda: idx.band_math(**cfg),
            "stack": lambda: idx.stack(**cfg),
            # Post-processing
            "vectorize": lambda: post.vectorize(**cfg),
            "smooth": lambda: post.smooth(**cfg),
            "regularize": lambda: post.regularize(**cfg),
            "zonal_stats": lambda: post.zonal_stats(**cfg),
            "buffer": lambda: post.buffer(**cfg),
            "centroids": lambda: post.centroids(**cfg),
            "compress": lambda: post.compress(**cfg),
            "cog": lambda: post.cog(**cfg),
            # SAR
            "despeckle": lambda: sar.despeckle(**cfg),
            "calibrate": lambda: sar.calibrate(**cfg),
            "flood_map": lambda: sar.flood_map(**cfg),
            "coherence": lambda: sar.coherence(**cfg),
        }

        if st == "custom":
            func = cfg.pop("func")
            return func(current_input, **cfg)

        handler = dispatch.get(st)
        if handler is None:
            msg = f"Unknown pipeline step type: {st!r}"
            raise ValueError(msg)
        return handler()

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Export pipeline to YAML file."""
        pipeline_dict = {
            "name": self.name,
            "description": self._description,
            "schedule": self._schedule,
            "steps": [{s.step_type: s.config} for s in self._steps],
        }
        Path(path).write_text(yaml.dump(pipeline_dict, default_flow_style=False))
        logger.info(f"Pipeline saved → {path}")

    @classmethod
    def from_yaml(cls, path: str | Path, engine=None) -> ProcessingPipeline:
        """Load pipeline from a YAML definition file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        pl = cls(name=d.get("name", Path(path).stem), engine=engine)
        pl._description = d.get("description", "")
        pl._schedule = d.get("schedule")
        for step_dict in d.get("steps", []):
            for step_type, config in step_dict.items():
                pl._steps.append(ProcessingStep(step_type, config or {}))
        return pl

    def __repr__(self) -> str:
        return f"ProcessingPipeline(name={self.name!r}, steps={len(self._steps)})"
