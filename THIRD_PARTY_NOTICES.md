# Third-party notices

The root [MIT license](LICENSE) covers the original HybridADFWI code and its documentation. It does not replace third-party licenses or grant a license to Marmousi-II model data.

## ADFWI

- Source: <https://github.com/liufeng2317/ADFWI>
- Pinned commit: `559f581149343b1ed3dd2168caae595690bd601d`
- Location: `vendor/ADFWI`, a Git submodule. Source-file SHA256 values are recorded in [upstream.lock.json](upstream.lock.json).
- License: MIT, Copyright (c) 2024 Liu Feng.

HybridADFWI calls the unmodified native solver and constructs a runtime history-recording variant of its pinned loop. Retain the following notice with distributions that include ADFWI or substantial portions derived from it. The original notice is also retained in `vendor/ADFWI/LICENSE` after submodule initialization.

```text
MIT License

Copyright (c) 2024 Liu Feng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Marmousi-II

Marmousi-II is provided by the Allied Geophysical Laboratories, University of Houston, with acknowledgement to Gary S. Martin, Robert Wiley and Kurt J. Marfurt. Its data terms are separate from the ADFWI and HybridADFWI code licenses. See [data provenance and download instructions](docs/DATA_PROVENANCE.md). Source and prepared model files are obtained locally; they are excluded from the release tree.

## Installed dependencies

Python dependencies are installed separately using [pyproject.toml](pyproject.toml) and retain their own licenses. ADFWI's full optional example/deep-learning dependency set is outside this repository's supported benchmark path.

## Plot alignment tool

The unmodified plotting alignment tool from [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills) is distributed under Apache License 2.0. See [source revision and attribution](scripts/figure_qa/README.md) and [full license](scripts/figure_qa/LICENSE.Apache-2.0). Project and solver licensing is unchanged.
