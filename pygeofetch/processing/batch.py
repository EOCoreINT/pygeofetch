"""
BatchProcessor — apply processing chains to multiple files in parallel.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from pygeofetch.processing.base import ProcessingResult
from pygeofetch.processing.pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Apply a processing chain to multiple input files in parallel.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()

        results = client.batch.process(
            inputs=["scene1.tif", "scene2.tif", "scene3.tif"],
            chain=[
                ("clip",      {"bbox": (-74.1, 40.6, -73.7, 40.9)}),
                ("reproject", {"crs": "EPSG:4326"}),
                ("ndvi",      {}),
                ("cog",       {}),
            ],
            output_dir="./processed/",
            parallel=4,
        )
    """

    def __init__(self, engine) -> None:
        self._engine = engine

    def process(
        self,
        inputs: list[str | Path],
        chain: list[tuple[str, dict[str, Any]]],
        output_dir: str | Path = ".",
        parallel: int = 2,
        on_error: str = "skip",
    ) -> list[ProcessingResult]:
        """
        Apply a processing chain to a list of input files.

        Args:
            inputs:     List of input file paths.
            chain:      Ordered list of (step_type, kwargs) tuples.
            output_dir: Root output directory.
            parallel:   Number of parallel workers.
            on_error:   ``"skip"`` (continue with other files) or ``"abort"``.

        Returns:
            List of final :class:`ProcessingResult` — one per input file.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[ProcessingResult | None] = [None] * len(inputs)

        def process_one(idx: int, inp: str | Path) -> ProcessingResult:
            inp_path = Path(inp)
            file_out_dir = out_dir / inp_path.stem
            pipeline = ProcessingPipeline(
                name=f"batch_{inp_path.stem}",
                engine=self._engine,
            )
            for step_type, kwargs in chain:
                getattr(pipeline, step_type, lambda **kw: None)(**kwargs)
                # Fallback for steps not in builder
                if not pipeline._steps or pipeline._steps[-1].step_type != step_type:
                    from pygeofetch.processing.pipeline import ProcessingStep

                    pipeline._steps.append(ProcessingStep(step_type, kwargs))

            run_result = pipeline.run(input=inp_path, output_dir=file_out_dir)
            if run_result.outputs:
                last = run_result.outputs[-1]
                return ProcessingResult(
                    success=run_result.success,
                    output_path=last,
                    input_path=inp_path,
                    operation="batch",
                    metadata={"steps": len(run_result.steps)},
                )
            return ProcessingResult(
                success=False,
                input_path=inp_path,
                operation="batch",
                error="No output produced",
            )

        logger.info(f"Batch processing {len(inputs)} files with {parallel} workers")

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(process_one, i, inp): i for i, inp in enumerate(inputs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    results[idx]
                    logger.info(
                        f"  ✓ [{idx + 1}/{len(inputs)}] {Path(inputs[idx]).name}"
                    )
                except Exception as exc:
                    logger.error(
                        f"  ✗ [{idx + 1}/{len(inputs)}] {Path(inputs[idx]).name}: {exc}"
                    )
                    if on_error == "abort":
                        raise
                    results[idx] = ProcessingResult(
                        success=False,
                        input_path=Path(inputs[idx]),
                        operation="batch",
                        error=str(exc),
                    )

        succeeded = sum(1 for r in results if r and r.success)
        logger.info(f"Batch complete: {succeeded}/{len(inputs)} succeeded")
        return [r for r in results if r is not None]

    def apply(
        self,
        func: Callable,
        inputs: list[str | Path],
        output_dir: str | Path = ".",
        parallel: int = 2,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Apply an arbitrary function to each input file in parallel.

        Args:
            func:       Function accepting (input_path, output_dir, **kwargs).
            inputs:     List of input file paths.
            output_dir: Output directory.
            parallel:   Number of workers.
            **kwargs:   Passed to func.

        Example::

            def my_proc(inp, out_dir, threshold=0.3):
                # Custom processing...
                return result

            results = client.batch.apply(my_proc, inputs, threshold=0.5)
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(func, inp, out_dir, **kwargs): i
                for i, inp in enumerate(inputs)
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error(f"apply: {exc}")
                    results.append(None)

        return results
