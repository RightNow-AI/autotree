# AutoTree paper draft

Generate the fixture-marked figures and compile the two-column draft:

```console
cd figures
uv run python make_figures.py --out out/
cd ../paper
tectonic main.tex
```

The PDF intentionally reports no benchmark result. Every integrated chart is a
fixture-harness illustration pending released GPU-scale measurements.
