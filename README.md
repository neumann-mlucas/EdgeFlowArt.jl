# EdgeFlowArt.jl

Image → many small strokes that follow (or drift toward) the image's edges.
N particles are seeded uniformly across the canvas, then advected step-by-step
through a vector field derived from the source image. Each particle's
trajectory becomes one stroke in the output. SVG-first; PNG is a Bresenham
preview.

Same vein as StringArt.jl / MonteCarloArt.jl / PrimitiveArt.md / StippleArt.md /
TriMosaic.md: constructive approximation via constrained primitive placement.
The primitive here is a short polyline, and the constraint is the flow field.

## Fields

Pick one via `--field <mode>`:

| Mode      | Definition                                       | Look                                          |
|-----------|--------------------------------------------------|-----------------------------------------------|
| `sobel`   | $F = \nabla \| \nabla I \|$ (climb edge ridge)   | Points on edges lock in; flat regions stay put — visible jitter off subject. Sketchy. |
| `dt`      | $F = -\nabla D$, $D$ = distance to thresholded edge | Every particle pulled to nearest edge. Clean outline, no background jitter. |
| `tangent` | $F = (-I_y, I_x)$ (perpendicular to gradient)    | Particles glide *along* edges, tracing continuous curves. Painterly. |

For `tangent`, integration flips sign per step against previous displacement
so the 180° tangent ambiguity doesn't fold curves back on themselves.

## Quick start

```bash
cd EdgeFlowArt.jl
just install
just test
julia --project=. -O3 -t 8 main.jl -i input.jpg -o out.svg --field tangent --points 5000 --steps 300
```

CLI knobs:

```
--input INPUT
--output OUTPUT             .png / .svg (comma-separate for both; .gif errors)
--size S                    default 512
--field MODE                sobel | dt | tangent   default tangent
--points N                  default 3000
--steps K                   default 200            per-particle integration steps
--eta E                     default 1.0            step size (px per step)
--tau T                     default 0.15           DT: edge threshold
--sigma S                   default 1.5            tangent: pre-blur sigma
--stroke-width W            default 0.6
--opacity A                 default 0.6
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
PHASE=baseline ./test/scripts/baseline.sh            # fixtures × {tangent, dt, sobel}
julia --project=. test/scripts/metrics.jl baseline p1 # → test/results/metrics.csv + sheet_<field>.png
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
│   ├── fields.jl            # sobel / dt / tangent field construction
│   ├── particles.jl         # bilinear sample + integration
│   ├── render.jl            # SVG polylines + Bresenham PNG
│   └── common.jl            # vendored from ArtScaffold.jl
├── test/
├── dev/
├── justfile
├── README.md
└── .gitignore
```

## Deferred

Small on purpose. Room to grow (only GIF, Wu anti-aliasing and the
structure tensor carry `ponytail:` markers in code; the rest are ideas):

- **Repulsion** between nearby particles → Poisson-disk on edges (kills bunching on strong ridges). Add with a spatial hash grid.
- **Structure-tensor eigendecomp** for tangent field (with coherence weighting). Current tangent is `⊥∇I` — good enough for portraits, upgrade if noisy on textured images.
- **Coarse-to-fine sigma decay**: blur the field early, sharpen late. Gets particles seeded in flat regions to migrate toward the subject before pinning.
- **Harris-weighted seeding**: seed density ∝ corner response, concentrate budget on features.
- **Perlin blend**: `F_blend = (1-w)·F_Perlin + w·F_tangent` for Fidenza-style curves.
- **Palette lock** + variable stroke width. Reuse `kmeans_palette_lab` from `common.jl`.
- **GIF of the descent** (checkpoint trajectories per K frames). `-o *.gif` currently errors.
- **Anti-aliased PNG** (Xiaolin Wu) — PNG preview is plain Bresenham.
