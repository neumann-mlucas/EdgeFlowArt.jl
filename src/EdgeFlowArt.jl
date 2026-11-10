module EdgeFlowArt

using Random
using Statistics: mean

include("common.jl")
include("fields.jl")
include("particles.jl")
include("render.jl")

# --- types ---------------------------------------------------------------

const Canvas = Matrix{RGB{N0f8}}
const GIF_FRAMES = 60             # growth frames: seed dots → full strokes
const GIF_FPS = 12
const GIF_HOLD = 18               # extra copies of the last frame (pause before loop)

# --- config --------------------------------------------------------------

Base.@kwdef struct Config
    size::Int = 512
    steps::Int = 400               # integration steps per direction
    points::Int = 20000            # max seeds (taken cells are skipped)
    eta::Float32 = 1.0f0           # step size (px per step)
    tau::Float32 = 0.05f0          # seeding: edge threshold on Sobel magnitude
    sigma::Float32 = 5.0f0         # structure-tensor smoothing sigma (px)
    stroke_width::Float64 = 1.2
    opacity::Float64 = 0.9
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
    want_gif = :gif in cfg.formats
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
    trajs = integrate(seeds, Fx, Fy, cfg.steps, cfg.eta)
    keep = length.(trajs) .> 1
    heads = [findfirst(==(s), t) for (s, t) in zip(seeds[keep], trajs[keep])]
    trajs = trajs[keep]

    @info ts() * "Sampling colors"
    colors = [path_color(input, t) for t in trajs]

    png = if want_png
        @info ts() * "Rasterizing PNG"
        raster_polylines(
            trajs,
            colors,
            w,
            h;
            opacity=Float32(cfg.opacity),
            stroke_width=cfg.stroke_width,
        )
    else
        fill(RGB{N0f8}(1, 1, 1), h, w)
    end

    gif = if want_gif
        @info ts() * "Rasterizing GIF ($GIF_FRAMES frames)"
        raster_growth(
            trajs,
            heads,
            colors,
            w,
            h;
            opacity=Float32(cfg.opacity),
            stroke_width=cfg.stroke_width,
            frames=GIF_FRAMES,
        )
    else
        Canvas[]
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

    return (png, svg, gif)
end

# --- output helpers ------------------------------------------------------

function save_gif(path::AbstractString, frames::Vector{Canvas})
    isempty(frames) && return
    held = [frames; fill(frames[end], GIF_HOLD)]
    save(path, stack(held; dims=3), fps=GIF_FPS)
end

end
