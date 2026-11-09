module EdgeFlowArt

using Random
using Statistics: mean

include("common.jl")
include("fields.jl")
include("particles.jl")
include("render.jl")

# --- types ---------------------------------------------------------------

const Canvas = Matrix{RGB{N0f8}}

# --- config --------------------------------------------------------------

Base.@kwdef struct Config
    size::Int = 512
    steps::Int = 200               # integration steps per particle
    points::Int = 3000             # particle count
    eta::Float32 = 1.0f0           # step size (px per step)
    tau::Float32 = 0.15f0          # seeding: edge threshold on Sobel magnitude
    sigma::Float32 = 1.5f0         # Tangent: pre-blur sigma (px)
    stroke_width::Float64 = 0.6
    opacity::Float64 = 0.6
    seed::Int = 42
    formats::Set{Symbol} = Set{Symbol}()
    timer::LogTimer = LogTimer()
end

export Config, load_input, render

# --- I/O -----------------------------------------------------------------

""" Load, center-crop, resize to `size`. Returns RGB canvas. """
load_input(path::AbstractString, size::Int) = load_square(path, size)

# --- main entry ----------------------------------------------------------

""" Compute vector field, integrate `points` particles for `steps`, emit
    trajectories as SVG polylines and/or PNG raster. """
function render(input::AbstractMatrix, cfg::Config)::Tuple{Canvas,String,Vector{Canvas}}
    want_svg = :svg in cfg.formats
    want_png = :png in cfg.formats
    ts = cfg.timer

    h, w = size(input)
    @info ts() * "Building tangent field"
    gray = to_gray(input)
    Fx, Fy = field_tangent(gray, cfg.sigma)

    @info ts() * "Seeding $(cfg.points) particles"
    rng = MersenneTwister(cfg.seed)
    # start on edges so every stroke traces a contour, none wasted on flat areas
    seeds = seed_edges(cfg.points, edge_mask(gray, cfg.tau), rng)

    @info ts() * "Integrating $(cfg.steps) steps"
    trajs = filter!(t -> length(t) > 1, integrate(seeds, Fx, Fy, cfg.steps, cfg.eta))

    @info ts() * "Sampling colors"
    colors = [path_color(input, t) for t in trajs]

    png = if want_png
        @info ts() * "Rasterizing PNG"
        raster_polylines(trajs, colors, w, h; opacity=Float32(cfg.opacity))
    else
        fill(RGB{N0f8}(1, 1, 1), h, w)
    end

    svg = if want_svg
        @info ts() * "Building SVG"
        svg_polylines(
            trajs,
            colors,
            w,
            h;
            stroke_width=cfg.stroke_width,
            opacity=cfg.opacity,
        )
    else
        ""
    end

    # ponytail: no GIF (would need per-frame trajectory checkpointing).
    # Add if animation of the descent is worth it.
    return (png, svg, Canvas[])
end

# --- output helpers ------------------------------------------------------

function save_gif(path::AbstractString, frames::Vector{Canvas})
    isempty(frames) && return
    save(path, stack(frames; dims=3), fps=5)
end

end
