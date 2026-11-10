# Rendering: SVG polylines + anti-aliased PNG / GIF rasterization.
# Stroke width follows tone (dark = wide) in all three outputs.

# Stroke color = mean source color along the trajectory (rounded pixels).
# Start-pixel color mislabels paths that cross color boundaries.
function path_color(img::AbstractMatrix, traj::Vector{Point2})::RGB{N0f8}
    h, w = size(img)
    acc = RGB{Float32}(0, 0, 0)
    for (x, y) in traj
        acc += RGB{Float32}(img[clamp(round(Int, y), 1, h), clamp(round(Int, x), 1, w)])
    end
    return RGB{N0f8}(acc / length(traj))
end

# Stroke width from tone: dark strokes wide, light strokes hairline, so the
# drawing keeps the source's light/dark contrast instead of uniform grey.
# `base` = width of a mid-grey stroke-ish; black ≈ 3.3×base, white ≈ 0.16×base.
const TONE_MIN = 0.3
const TONE_GAIN = 2.2
tone_width(base::Real, col::RGB{N0f8})::Float32 =
    Float32(base * (TONE_MIN + TONE_GAIN * (1 - Float64(Gray(col))))^1.5)

# Build SVG document from trajectories.
function svg_polylines(
    trajs::Vector{Vector{Point2}},
    colors::Vector{RGB{N0f8}},
    w::Int,
    h::Int;
    stroke_width::Float64=0.6,
    opacity::Float64=0.6,
    background::String="white",
)::String
    io = IOBuffer()
    print(io, svg_open(w, h), "\n")
    print(io, """<rect width="100%" height="100%" fill="$background"/>""", "\n")
    for (traj, col) in zip(trajs, colors)
        length(traj) < 2 && continue
        print(io, """<polyline points=\"""")
        for (j, (x, y)) in enumerate(traj)
            j > 1 && print(io, " ")
            print(io, round(x; digits=2), ",", round(y; digits=2))
        end
        print(
            io,
            """\" stroke="$(rgb_hex(col))" stroke-width="$(round(tone_width(stroke_width, col); digits=2))" """,
            """stroke-opacity="$opacity" fill="none" stroke-linecap="round"/>\n""",
        )
    end
    print(io, SVG_CLOSE)
    return String(take!(io))
end

# Alpha-blend `color` into canvas[y, x] with weight `a` (no-op off canvas).
@inline function blend!(canvas::Matrix{RGB{N0f8}}, x::Int, y::Int, color, a::Float32)
    h, w = size(canvas)
    (1 <= x <= w && 1 <= y <= h) || return nothing
    bg = RGB{Float32}(canvas[y, x])
    canvas[y, x] = RGB{N0f8}(
        clamp(a * color.r + (1 - a) * bg.r, 0.0f0, 1.0f0),
        clamp(a * color.g + (1 - a) * bg.g, 0.0f0, 1.0f0),
        clamp(a * color.b + (1 - a) * bg.b, 0.0f0, 1.0f0),
    )
    return nothing
end

# Paint one stroke piece (points `pts`, width `wd` px) as anti-aliased round
# stamps every ≤ 0.5 px. Coverage is max-combined in `cov` first and blended
# once per pixel, so a stroke never darkens itself where stamps overlap.
# `cov` is a canvas-sized scratch buffer, left zeroed on return.
function draw_stroke!(
    canvas::Matrix{RGB{N0f8}},
    cov::Matrix{Float32},
    pts::AbstractVector{Point2},
    color::RGB{N0f8},
    opacity::Float32,
    wd::Float32,
)
    h, w = size(canvas)
    r = max(wd / 2, 0.5f0)
    R = ceil(Int, r + 1)
    touched = Int[]
    stamp(cx, cy) =
        for y = (round(Int, cy)-R):(round(Int, cy)+R),
            x = (round(Int, cx)-R):(round(Int, cx)+R)

            (1 <= x <= w && 1 <= y <= h) || continue
            c = clamp(r + 0.5f0 - hypot(x - cx, y - cy), 0.0f0, 1.0f0)
            c > 0 || continue
            i = (x - 1) * h + y
            cov[i] == 0 && push!(touched, i)
            cov[i] = max(cov[i], c)
        end
    stamp(pts[1]...)
    for j = 2:length(pts)
        (x0, y0), (x1, y1) = pts[j-1], pts[j]
        n = max(1, ceil(Int, 2 * hypot(x1 - x0, y1 - y0)))
        for k = 1:n
            stamp(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
        end
    end
    cf = RGB{Float32}(color)
    for i in touched
        y, x = mod1(i, h), cld(i, h)
        blend!(canvas, x, y, cf, opacity * cov[i])
        cov[i] = 0
    end
end

# Rasterize all strokes onto a white canvas, width per stroke from tone.
function raster_polylines(
    trajs::Vector{Vector{Point2}},
    colors::Vector{RGB{N0f8}},
    w::Int,
    h::Int;
    opacity::Float32=0.6f0,
    stroke_width::Float64=1.0,
)::Matrix{RGB{N0f8}}
    canvas = fill(RGB{N0f8}(1, 1, 1), h, w)
    cov = zeros(Float32, h, w)
    for (traj, col) in zip(trajs, colors)
        draw_stroke!(canvas, cov, traj, col, opacity, tone_width(stroke_width, col))
    end
    return canvas
end

# GIF frames of all strokes growing at once: frame 1 = a dot at every seed
# (`heads[i]` = seed index in trajs[i]), then each frame extends every stroke
# both ways by 1/frames of its own length, so all finish on the last frame.
# Pieces are blended incrementally on one canvas.
# ponytail: the joint pixels between two frames' pieces get blended twice
# (slightly darker); per-stroke coverage memory if that ever shows.
function raster_growth(
    trajs::Vector{Vector{Point2}},
    heads::Vector{Int},
    colors::Vector{RGB{N0f8}},
    w::Int,
    h::Int;
    opacity::Float32=0.6f0,
    stroke_width::Float64=1.0,
    frames::Int=60,
)::Vector{Matrix{RGB{N0f8}}}
    canvas = fill(RGB{N0f8}(1, 1, 1), h, w)
    cov = zeros(Float32, h, w)
    wds = [tone_width(stroke_width, c) for c in colors]
    for (t, hd, c, wd) in zip(trajs, heads, colors, wds)
        draw_stroke!(canvas, cov, view(t, hd:hd), c, opacity, wd)
    end
    snaps = [copy(canvas)]
    for f = 1:frames
        for (t, hd, c, wd) in zip(trajs, heads, colors, wds)
            nf, nb = length(t) - hd, hd - 1          # segments ahead / behind the seed
            a, b = fld((f - 1) * nf, frames), fld(f * nf, frames)
            p, q = fld((f - 1) * nb, frames), fld(f * nb, frames)
            b > a && draw_stroke!(canvas, cov, view(t, (hd+a):(hd+b)), c, opacity, wd)
            q > p && draw_stroke!(canvas, cov, view(t, (hd-q):(hd-p)), c, opacity, wd)
        end
        push!(snaps, copy(canvas))
    end
    return snaps
end
