# Rendering: SVG polylines + naive PNG line rasterization.

# Sample source color at trajectory start (rounded to pixel).
function sample_color(img::AbstractMatrix, x::Float32, y::Float32)::RGB{N0f8}
    h, w = size(img)
    xi = clamp(round(Int, x), 1, w)
    yi = clamp(round(Int, y), 1, h)
    return RGB{N0f8}(img[yi, xi])
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

# Bresenham line into an RGB canvas. Alpha-blend against existing pixel.
function draw_line!(
    canvas::Matrix{RGB{N0f8}},
    x0::Real,
    y0::Real,
    x1::Real,
    y1::Real,
    color::RGB{N0f8},
    alpha::Float32,
)
    h, w = size(canvas)
    ix0 = round(Int, x0)
    iy0 = round(Int, y0)
    ix1 = round(Int, x1)
    iy1 = round(Int, y1)
    dx = abs(ix1 - ix0)
    dy = -abs(iy1 - iy0)
    sx = ix0 < ix1 ? 1 : -1
    sy = iy0 < iy1 ? 1 : -1
    err = dx + dy
    cf = RGB{Float32}(color)
    while true
        if 1 <= ix0 <= w && 1 <= iy0 <= h
            bg = RGB{Float32}(canvas[iy0, ix0])
            r = alpha * cf.r + (1 - alpha) * bg.r
            g = alpha * cf.g + (1 - alpha) * bg.g
            b = alpha * cf.b + (1 - alpha) * bg.b
            canvas[iy0, ix0] = RGB{N0f8}(
                clamp(r, 0.0f0, 1.0f0),
                clamp(g, 0.0f0, 1.0f0),
                clamp(b, 0.0f0, 1.0f0),
            )
        end
        (ix0 == ix1 && iy0 == iy1) && break
        e2 = 2 * err
        if e2 >= dy
            err += dy
            ix0 += sx
        end
        if e2 <= dx
            err += dx
            iy0 += sy
        end
    end
    return nothing
end

# Rasterize all trajectories onto a white canvas via Bresenham. Cheap PNG
# preview. ponytail: no anti-aliasing; add Xiaolin Wu if lines look chunky.
function raster_polylines(
    trajs::Vector{Vector{Point2}},
    colors::Vector{RGB{N0f8}},
    w::Int,
    h::Int;
    opacity::Float32=0.6f0,
)::Matrix{RGB{N0f8}}
    canvas = fill(RGB{N0f8}(1, 1, 1), h, w)
    for (traj, col) in zip(trajs, colors)
        for j = 1:(length(traj)-1)
            (x0, y0) = traj[j]
            (x1, y1) = traj[j+1]
            draw_line!(canvas, x0, y0, x1, y1, col, opacity)
        end
    end
    return canvas
end
