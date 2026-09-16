# Classification

Turn a multi-band raster into a map of discrete classes — land cover,
water vs. land, crop type, anything a spectral signature can
distinguish. Two real, different approaches, for two different real
situations: **K-means** when you have no labeled examples and want
pygeofetch to find natural groupings on its own, and **Random
Forest/SVM** when you have real, labeled training samples and want a
model that generalizes to the rest of the scene.

This fills a real gap that existed everywhere else in pygeofetch:
every spectral index and processing tool assumed you'd bring your own
classifier from a separate library. `client.classify` is that
classifier, built directly on the same raster I/O conventions as
`client.indices` and `client.preprocess` — its output is immediately
usable with `client.post.vectorize()` and `client.post.zonal_stats()`,
which already expected "a classification raster" as real input.

## Quick start

```python
from pygeofetch import PyGeoFetch
client = PyGeoFetch()

# No training data needed -- groups pixels by spectral similarity
clusters = client.classify.kmeans(
    inputs=["B02.tif", "B03.tif", "B04.tif", "B08.tif"], n_clusters=5,
)

# With real, labeled training samples
client.classify.train(
    inputs=["B02.tif", "B03.tif", "B04.tif", "B08.tif"],
    training_data="training_samples.geojson", class_field="land_cover",
    output_model="model.joblib",
)
land_cover = client.classify.predict(
    inputs=["B02.tif", "B03.tif", "B04.tif", "B08.tif"], model="model.joblib",
)
```

## Unsupervised: K-means

```python
result = client.classify.kmeans(
    inputs=["B02.tif", "B03.tif", "B04.tif", "B08.tif"],
    n_clusters=5, output="clusters.tif",
)
print(result.metadata)
# {'n_clusters': 5, 'valid_pixel_fraction': 0.94}
```

Every input band is read fully and stacked into one array; each pixel
becomes a point in N-dimensional spectral space (N = number of input
bands), and K-means groups them into `n_clusters` clusters by spectral
similarity alone.

**The cluster IDs are arbitrary.** Cluster `0` has no inherent meaning
— it isn't "water" just because it's the first cluster. After running
K-means, inspect each cluster's real spectral signature (e.g., its
mean NDVI, or overlay it on a true-color composite) to figure out
which cluster corresponds to which real class, before using the
result for anything downstream.

**Choosing `n_clusters`**: start with your best real guess at the
number of distinct land-cover types in the scene, then look at the
result — clusters that never form a coherent, spatially sensible
region usually mean you've asked for more clusters than the data
actually supports.

## Supervised: Random Forest / SVM

### Real training data format

`training_data` is any real vector file GeoPandas can read (GeoJSON,
Shapefile, GeoPackage) with one attribute column holding the class
label. Both point and polygon geometries work, and mean different
things for how many real pixels each label actually contributes:

- **Points**: exactly one pixel sampled per point — the pixel the
  point falls inside.
- **Polygons**: every real pixel the polygon's footprint covers gets
  labeled — a single polygon over a large, homogeneous field can
  contribute thousands of real training pixels from one drawn shape.

```python
result = client.classify.train(
    inputs=["B02.tif", "B03.tif", "B04.tif", "B08.tif"],
    training_data="training_samples.geojson",
    class_field="land_cover",       # the column holding real class labels
    output_model="model.joblib",
    model_type="random_forest",     # or "svm"
    n_estimators=100,               # Random Forest only
    test_size=0.2,                  # fraction held out for real, honest accuracy
)
print(result.metadata["test_accuracy"])
print(result.metadata["n_train_samples"], result.metadata["n_test_samples"])
```

**The reported accuracy is real and honest** — computed on a held-out
test split that never touches training, every single time, not an
optional flag you have to remember to turn on. `result.metadata` also
carries the full `classification_report` (per-class precision,
recall, F1) if you need to check whether the model struggles on one
specific class rather than trust one overall number.

**A real, deliberate constraint**: training raises a clear error if
fewer than 10 valid training pixels were actually extracted — usually
a sign the training geometries don't actually overlap the raster's
real extent or CRS, not a subtle statistical warning to ignore.

### Applying the trained model

```python
result = client.classify.predict(
    inputs=["B02.tif", "B03.tif", "B04.tif", "B08.tif"],
    model="model.joblib", output="land_cover_map.tif",
)
```

`inputs` must be the same real bands, in the same order, used for
training — `predict()` checks the band count against what the saved
model expects and fails with a clear message naming the mismatch,
rather than letting scikit-learn's own less specific shape error
surface instead.

## Clipping the result to an AOI

Both `kmeans` and `predict` accept `bbox`, `geometry`, and
`geometry_crs` — same real handling as every method in
[Spectral Indices](spectral-indices.md#clipping-the-result-to-an-aoi)
and `Preprocessor.clip()`:

```python
land_cover = client.classify.predict(
    inputs=[...], model="model.joblib",
    geometry="farm_boundary.geojson",
)
```

`train()` doesn't take a clip option — clip your training data's
extent before extracting samples if you need to restrict it, since
clipping *after* training a model doesn't do anything meaningful.

## CLI reference

```bash
# Unsupervised
pygeofetch classify kmeans B02.tif B03.tif B04.tif B08.tif -k 5 -o clusters.tif

# Train
pygeofetch classify train B02.tif B03.tif B04.tif B08.tif \
    --training-data samples.geojson --class-field land_cover \
    -o model.joblib --model-type random_forest --n-estimators 100

# Predict
pygeofetch classify predict B02.tif B03.tif B04.tif B08.tif \
    -m model.joblib -o land_cover_map.tif

# Clip the result
pygeofetch classify predict B02.tif B03.tif B04.tif B08.tif \
    -m model.joblib --geometry farm_boundary.geojson
```

Run `pygeofetch classify COMMAND --help` for the full option list of
any individual command.

## Common pitfalls

- **Mismatched band resolutions or extents.** All `inputs` must
  already share the same real grid — same shape, same transform.
  Reproject/resample mismatched bands first (see
  [Preprocessing](preprocessing.md)) rather than pass them directly.
- **Too few, or spatially clustered, training samples.** A model
  trained on 15 points all drawn from one corner of the scene will
  confidently mispredict everywhere else — spread real training
  samples across the actual range of conditions you expect the model
  to see.
- **Treating K-means cluster IDs as meaningful class labels without
  checking.** Always inspect what each cluster actually represents
  before reporting results from it.
- **Class imbalance in training data.** If 90% of your real training
  pixels are one class, the model can reach high overall accuracy by
  mostly ignoring the rare classes — check the per-class breakdown in
  `classification_report`, not just `test_accuracy`, before trusting
  a model on a genuinely imbalanced real problem.

## Full method reference

| Method | Real purpose | Needs training data? |
|---|---|---|
| `kmeans(inputs, n_clusters=5, ...)` | Unsupervised clustering | No |
| `train(inputs, training_data, class_field, output_model, ...)` | Fit a supervised model | Yes |
| `predict(inputs, model, ...)` | Apply a trained model | No (uses a saved model) |

**Not yet supported**: chunked/memory-safe processing for rasters
larger than available RAM — unlike the methods in
[Spectral Indices](spectral-indices.md), `kmeans` and `predict` both
currently read every input band fully into memory. `train()`'s own
real training-sample extraction is inherently limited to what fits in
memory regardless, since scikit-learn's fit step needs the full
training array at once.
