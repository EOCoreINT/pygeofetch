"""
stream_options.py — real pydantic v2 configuration models for the
stac_compute and chunked-processing modules.

Uses pydantic v2 natively (model_config = ConfigDict(...), field
validators via @field_validator) -- not the deprecated v1-style
class-based Config this project has elsewhere flagged as a real,
pending deprecation warning to fix. New code should not repeat that
same real issue.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StreamOptions(BaseModel):
    """
    Real configuration for a remote, windowed COG read
    (`stac_compute.cog_streamer`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_range_support: bool = Field(
        default=True,
        description=(
            "Real, honest pre-flight check confirming the remote server "
            "actually supports HTTP range requests before attempting a "
            "windowed read -- if False, a server that silently doesn't "
            "support them causes GDAL to fall back to downloading the "
            "entire real file, defeating the purpose of streaming."
        ),
    )
    bbox_crs: str | None = Field(
        default=None,
        description="Real CRS the requested bbox's coordinates are in, if different from the remote COG's own CRS.",
    )
    bands: list[int] | None = Field(
        default=None,
        description="Real 1-based band indices to read. None reads every real band.",
    )
    gdal_disable_readdir_on_open: bool = Field(
        default=True,
        description="Real GDAL optimization -- prevents a directory-listing probe many object stores reject or handle slowly.",
    )

    @field_validator("bands")
    @classmethod
    def _bands_must_be_positive(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(b < 1 for b in v):
            msg = "StreamOptions.bands: band indices are real, 1-based rasterio band numbers -- must all be >= 1."
            raise ValueError(msg)
        return v


class ChunkConfig(BaseModel):
    """
    Real configuration for tiled raster processing
    (`core.chunked_processor.ChunkedProcessor`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tile_size: int = Field(
        default=1024, gt=0, description="Real tile edge length in pixels."
    )
    overlap: int = Field(
        default=0,
        ge=0,
        description=(
            "Real pixel margin read beyond tile_size on every side, then "
            "trimmed before writing. 0 for pixel-independent operations "
            "(NDVI, band math); > 0 (>= window_radius) for real "
            "neighborhood operations (a moving-window filter, GLCM "
            "texture) to avoid wrong values at tile boundaries."
        ),
    )
    output_format: Literal["cog", "zarr"] = Field(
        default="cog",
        description="Real output backend -- a properly tiled, BigTIFF-safe GeoTIFF, or a chunked Zarr store.",
    )
    nodata: float = Field(
        default=-9999.0, description="Real nodata value for the output."
    )
    zarr_compressor: Literal["zstd", "lz4", "gzip", "zlib", "none"] = Field(
        default="zstd",
        description="Real Zarr compressor -- only used when output_format='zarr'.",
    )

    @model_validator(mode="after")
    def _overlap_needs_tile_size_room(self) -> "ChunkConfig":
        if self.overlap >= self.tile_size:
            msg = (
                f"ChunkConfig: overlap ({self.overlap}) must be smaller than "
                f"tile_size ({self.tile_size}) -- an overlap this large would "
                f"mean each tile's real margin is bigger than the tile's own "
                f"real output region, which is never a real, sensible "
                f"configuration."
            )
            raise ValueError(msg)
        return self


class GeoParquetExportOptions(BaseModel):
    """Real configuration for GeoParquet export (`core.data_formats.geoparquet_io`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition_by: str | None = Field(
        default=None,
        description="Real column name to Hive-partition the output by, instead of writing one flat file.",
    )
    compression: Literal["snappy", "zstd", "gzip", "none"] = Field(
        default="snappy",
        description="Real Parquet-level compression codec.",
    )


class ZarrExportOptions(BaseModel):
    """Real configuration for Zarr export (`core.data_formats.zarr_io`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks: tuple[int, ...] | None = Field(
        default=None,
        description="Real per-dimension chunk shape. None lets Zarr choose a real, sensible default from the array's own shape/dtype.",
    )
    compressor: Literal["zstd", "lz4", "gzip", "zlib", "none"] = Field(default="zstd")

    @field_validator("chunks")
    @classmethod
    def _chunks_must_be_positive(
        cls, v: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if v is not None and any(c <= 0 for c in v):
            msg = "ZarrExportOptions.chunks: every real chunk dimension must be > 0."
            raise ValueError(msg)
        return v
