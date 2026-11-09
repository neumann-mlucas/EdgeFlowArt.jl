# Rendering: SVG polylines + anti-aliased PNG / GIF rasterization.

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
            """\" stroke="$(rgb_hex(col))" stroke-width="$stroke_width" """,
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

# Anti-aliased line (Xiaolin Wu): per major-axis step, split coverage between
# the two pixels straddling the line.
# ponytail: no endpoint coverage; shared polyline vertices get blended twice.
function draw_line!(
    canvas::Matrix{RGB{N0f8}},
    x0::Real,
    y0::Real,
    x1::Real,
    y1::Real,
    color::RGB{N0f8},
    alpha::Float32,
)
    cf = RGB{Float32}(color)
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep
        x0, y0, x1, y1 = y0, x0, y1, x1
    end
    if x0 > x1
        x0, x1, y0, y1 = x1, x0, y1, y0
    end
    g = x1 == x0 ? 0.0f0 : Float32((y1 - y0) / (x1 - x0))
    for x = round(Int, x0):round(Int, x1)
        y = Float32(y0 + g * (x - x0))
        yi = floor(Int, y)
        f = y - yi
        if steep
            blend!(canvas, yi, x, cf, alpha * (1 - f))
            blend!(canvas, yi + 1, x, cf, alpha * f)
        else
            blend!(canvas, x, yi, cf, alpha * (1 - f))
            blend!(canvas, x, yi + 1, cf, alpha * f)
        end
    end
    return nothing
end

function draw_polyline!(canvas, traj::Vector{Point2}, col::RGB{N0f8}, opacity::Float32)
    for j = 1:(length(traj)-1)
        (x0, y0), (x1, y1) = traj[j], traj[j+1]
        draw_line!(canvas, x0, y0, x1, y1, col, opacity)
    end
end

# Rasterize strokes in drawing order onto a white canvas. With `frames > 0`,
# also snapshot the canvas every length/frames strokes (GIF of the drawing).
function raster_polylines(
    trajs::Vector{Vector{Point2}},
    colors::Vector{RGB{N0f8}},
    w::Int,
    h::Int;
    opacity::Float32=0.6f0,
    frames::Int=0,
)::Tuple{Matrix{RGB{N0f8}},Vector{Matrix{RGB{N0f8}}}}
    canvas = fill(RGB{N0f8}(1, 1, 1), h, w)
    snaps = Matrix{RGB{N0f8}}[]
    every = frames > 0 ? max(1, cld(length(trajs), frames)) : 0
    for (i, (traj, col)) in enumerate(zip(trajs, colors))
        draw_polyline!(canvas, traj, col, opacity)
        every > 0 && (i % every == 0 || i == length(trajs)) && push!(snaps, copy(canvas))
    end
    return canvas, snaps
end
