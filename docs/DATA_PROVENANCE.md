# Marmousi-II data provenance

The benchmark uses Marmousi-II from the Allied Geophysical Laboratories (AGL), University of Houston. The [official download page](https://www.agl.uh.edu/downloads/downloads.htm) lists the model files. The [AGL data portal](https://www.agl.uh.edu/resources-data.php) credits Gary S. Martin, Robert Wiley and Kurt J. Marfurt.

## Data terms

As checked on 2026-11-07, the AGL portal classifies Marmousi II as **Level 3 - Limited**. Its Level 3 description allows sharing among student researchers and publication sources. We have not identified an explicit general-purpose public redistribution license for these model files. The MIT licenses in this repository apply to code; they do not relicense the AGL data. Consult the linked source terms for your intended use.

The release tree excludes the downloaded SEG-Y sources and the prepared `model_input.npz` files. Preparation obtains the inputs directly from AGL. Recorded sensitivity kernels and performance measurements are research outputs with their provenance retained; the code license does not make a representation about rights in third-party input data.

## Official sources and integrity

`src/hybridadfwi/models.py` downloads the following files under `https://www.agl.uh.edu/downloads/`. Existing local downloads are also verified. A changed remote file causes an error rather than silently changing an experiment.

| Official source | SHA256 |
| --- | --- |
| [vp_marmousi-ii.segy.gz](https://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz) | `2c7a0c4170c95a15620c2146573711c61a4cef8dc27338ef6c211a81ffa68920` |
| [vs_marmousi-ii.segy.gz](https://www.agl.uh.edu/downloads/vs_marmousi-ii.segy.gz) | `c88bec68fc4bd7c6e624737a20ecd3241c90e10512103e28e61b344fb5baed7d` |
| [density_marmousi-ii.segy.gz](https://www.agl.uh.edu/downloads/density_marmousi-ii.segy.gz) | `9832e20ce1f79470fa8c100e79d66566cf8c1c162baa4289f37e7ee0b96f1e67` |

Source arrays use `(x,z)` axes, km/s velocities and g/cm³ density. Preparation produces `(z,x)` arrays in m/s and kg/m³, crops approximately x=5–14 km, uses bilinear resampling and preserves true water cells while smoothing the solid initial model. Both benchmark methods use the same prepared file. The default density is constant 2450 kg/m³ following the pinned ADFWI example.

## Regenerate the model inputs

From the installed source checkout:

```bash
python scripts/prepare_marmousi.py --preset smoke
python scripts/prepare_marmousi.py --preset paper
```

The outputs are `data/prepared/marmousi2_smoke.npz` and `data/prepared/marmousi2_paper.npz`, with JSON sidecars. [The archived smoke metadata](../evidence/marmousi2_smoke.json) and [paper metadata](../evidence/marmousi2_paper.json) record exact grids, source checksums and original prepared-file hashes. All three archived GPU evidence bundles used the `paper` prepared model; the no-checkpoint pair changes the shot count, not the model.

For optional local reconstruction of the omitted evidence input copies, after checking the applicable data terms:

```bash
cp data/prepared/marmousi2_paper.npz evidence/marmousi2-fd8-f64-20261106/model_input.npz
cp data/prepared/marmousi2_paper.npz evidence/no-checkpoint-1shot-20261106/model_input.npz
cp data/prepared/marmousi2_paper.npz evidence/no-checkpoint-capacity-9shot-20261106/model_input.npz
```

Historical manifests retain the hashes of the original local files; they are evidence records, not declarations that every recorded artifact is distributed. Archive byte hashes may differ across serialization/library versions, so check model arrays and metadata as well when reproducing with a different environment. New benchmark runs generate new local protocols and observations rather than reusing original host paths.
