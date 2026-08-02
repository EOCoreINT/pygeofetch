**A full update on PyGeoFetch's InSAR work, start to finish.**

Since the first post here, I've been building out and stress-testing the InSAR side of PyGeoFetch against something adversarial: a real Sentinel-1 subsidence study over Mexico City, one of the hardest, best-documented InSAR problems there is. Here's the honest shape of that work.

**Data pipeline.** Real bugs in scene downloading (silent filename collisions across providers), GCP-based reprojection, and AOI extraction (an AOI that grows large enough to straddle two Sentinel-1 sub-swaths gets silently bounded by whichever one gets matched first). Found and fixed, each one confirmed against real data, not assumed.

**Core InSAR corrections.** SNAP's own documentation admits Sentinel-1 TOPS data needs burst-boundary correction and per-burst spectral diversity that most simplified pipelines skip. Built both natively: real per-burst-overlap ESD (verified against the original 2012 paper that introduced it) and real deburst (verified against ESA's own algorithm docs). Also found and fixed a real orbital phase ramp bug, unrelated to ground motion, that was quietly inflating every displacement estimate. One test told the story: a linear fit explained 95.5% of the "signal" before the fix, 0.7% after.

**Atmospheric correction.** The part I flagged as missing last time. Wired in real ECMWF ERA5 reanalysis data, which took three separate rounds of real, sequential failures against live infrastructure to get right, wrong constructor arguments, a missing download step, a coordinate-array bug, each only discoverable by actually running it, not by reading documentation. Also found a second, unrelated bug: an earlier correction method was fitting a straight line directly through wrapped phase, phase that jumps every time it crosses ±π. Fixed with the same circular-regression technique already proven elsewhere in the codebase.

**Validation.** Rather than trust one run, I rebuilt the entire notebook from scratch and reproduced the investigation independently. Same real bugs did not reappear. Same baseline-optimized network structure. Same pairs excluded for the same real reason, decorrelation, not a processing failure. Where the exact numbers didn't match precisely, the reason itself was real and traceable: reference pixel choice has a genuine, now-documented effect on how disconnected regions bridge together.

The honest result: displacement consistent in sign and order of magnitude with independently published rates for this same real location, a decade apart, which fits what's actually known about this specific process, longterm, ongoing aquifer compaction that doesn't reset on its own. Not a claim of matching a six-year, 300-scene published study. A claim that a much smaller, fully open dataset, processed with real, verified corrections, lands somewhere physically honest.

Every fix in this thread was found by running the pipeline against reality, not by writing code that looked right. That's the actual point of open source work like this: not a polished demo, a tool that gets more trustworthy each time someone, including me, tries to break it.

Open source, still evolving in the open.

#InSAR #RemoteSensing #Sentinel1 #OpenSource #GIS #EarthObservation #SAR #ERA5
