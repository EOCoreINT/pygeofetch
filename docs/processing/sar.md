# SAR Processing (SARProcessor)

```bash
pip install "pygeofetch[sar]"
```

Despeckling, calibration, flood mapping, and coherence, with pluggable
backends.

## Flood mapping example

```python
from pygeofetch.sar import SARProcessor

sar = SARProcessor()
cal_pre = sar.calibrate("pre_event_vv.tif", output_type="sigma0", in_db=True)
cal_post = sar.calibrate("post_event_vv.tif", output_type="sigma0", in_db=True)

flood_result = sar.flood_map(
    str(cal_post.output_path),
    reference=str(cal_pre.output_path),
    threshold=-15.0,
)
print(f"{flood_result.metadata['water_pct']:.1f}% flagged as flooded")
```

## Backends

| Backend | Requires | Best for |
|---|---|---|
| `"native"` | Nothing extra | Despeckle, calibrate, flood map, coherence |
| `"sarxarray"` | `pygeofetch[sar]` | xarray/Dask-native large-scale processing |
| `"ost"` | `pygeofetch[ost]` + SNAP | Production Range-Doppler terrain correction |

The default `"native"` backend needs nothing beyond the base install —
real despeckling, calibration, and flood mapping from raw SAR
intensity, with no separate SAR toolkit required to get started.

```python
sar = SARProcessor(backend="native")   # default
sar = SARProcessor(backend="sarxarray")
sar = SARProcessor(backend="ost")
```
