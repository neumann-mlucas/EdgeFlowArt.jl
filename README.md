# EdgeFlowArt.jl

Image → strokes that trace the image's edges. Particles are seeded on edge
pixels, then traced both ways along the image's tangent field until they run
into another stroke. Each trajectory becomes one stroke. SVG-first; PNG is an
anti-aliased preview; GIF shows the strokes appearing in drawing order.

Same vein as StringArt.jl / MonteCarloArt.jl / PrimitiveArt.md / StippleArt.md /
TriMosaic.md: constructive approximation via constrained primitive placement.
The primitive here is a short polyline, and the constraint is the flow field.

## Algorithm

- **Field**: tangent of the smoothed structure tensor $J = G_\sigma * (\nabla I \nabla I^T)$
  (minor eigenvector), magnitude $\sqrt{\lambda_1-\lambda_2}$ (coherence), soft-normalized
  $F/(|F|+\varepsilon)$ so strong coherent edges move fast and noise barely moves.
  Smoothing $J$ instead of $I$ keeps thin lines and gives long flowing strokes on
  texture (fur, swirls).
- **Orientation, not vector**: $F$ and $-F$ are the same direction. Bilinear sampling
  flips each corner to agree with the stroke's heading before blending, and each step
  is aligned with the previous one.
- **Seeds**: random pixels with Sobel magnitude above $\min(\tau,\ \text{top-10\% quantile})$,
  so low-contrast images still get edges.
- **Spacing**: 2 px occupancy grid. Strokes are traced one at a time (random order), each
  both ways from its seed, and stop on entering a cell another stroke owns. Seeds in a
  taken cell are skipped. No pile-up; `--points` is an upper bound.
- **Color**: mean source color along the stroke.

Gradient-descent fields (`sobel` = climb $|\nabla I|$, `dt` = descend edge
distance) were removed: their strokes end on the edge, so they draw short
dots / rays perpendicular to it instead of contours.

## Quick start

```bash
cd EdgeFlowArt.jl
just install
just test
julia --project=. -O3 -t 8 main.jl -i input.jpg -o out.svg --points 5000 --steps 300
```

CLI knobs:

```
--input INPUT
--output OUTPUT             .png / .svg / .gif (comma-separate for several)
--size S                    default 512
--points N                  default 3000
--steps K                   default 200            steps per direction from seed
--eta E                     default 1.0            step size (px per step)
--tau T                     default 0.15           seeding: edge threshold
--sigma S                   default 1.5            structure-tensor sigma
--stroke-width W            default 1.0
--opacity A                 default 0.9
--seed                      default 42
--verbose
```

## Tests

```bash
just test    # synthetic striped fixtures
just check   # JET lint + tests + shellcheck
```

Quality sweep (fixtures in `test/fixtures/`, not versioned):

```bash
PHASE=baseline ./test/scripts/baseline.sh            # fixtures, tangent
julia --project=. test/scripts/metrics.jl baseline p1 # → test/results/metrics.csv + sheet_tangent.png
```

Metrics (from the SVG polylines vs the source): `edge_prec`/`edge_rec`
(ink on edges / edges inked), `flat_ink` (ink on flat areas), `pile_p99`/`pile_max`
(paths per pixel), `dE` (stroke vs source color), `turn` (zigzag), `len`, `svg_kb`.

## Layout

```
EdgeFlowArt.jl/
├── Project.toml
├── main.jl                  # CLI
├── src/
│   ├── EdgeFlowArt.jl       # module: Config + render()
│   ├── fields.jl            # tangent field + edge mask
│   ├── particles.jl         # edge seeding, bilinear sample, integration
│   ├── render.jl            # SVG polylines + anti-aliased PNG / GIF
│   └── common.jl            # vendored from ArtScaffold.jl
├── test/
├── dev/
├── justfile
├── README.md
└── .gitignore
```

## Deferred

Ideas, not planned:

- **Fill / hatching**: seed new strokes at a fixed offset beside existing ones (Jobard–Lefer) to shade flat regions.
- **Harris-weighted seeding**: seed density ∝ corner response, concentrate budget on features.
- **Perlin blend**: `F_blend = (1-w)·F_Perlin + w·F_tangent` for Fidenza-style curves.
- **Palette lock** + variable stroke width. Reuse `kmeans_palette_lab` from `common.jl`.
