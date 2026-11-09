# Quality metrics over sweep outputs. Reads <phase>/<img>__<cfg>.svg (polylines)
# + <phase>/_inputs/<img>.png (source), writes test/results/metrics.csv and
# one contact sheet per config (rows = images, cols = phases).
#
# Usage: julia --project=. test/scripts/metrics.jl baseline p1 p2 ...
#
# Metrics (per run; ink = pixels touched by a path, counted once per path):
#   paths       polylines in the SVG
#   dots        paths whose bbox extent < 1 px (never moved)
#   edge_prec   share of ink pixels on (1-px dilated) edges     ↑
#   edge_rec    share of edge pixels touched by ink             ↑
#   flat_ink    share of ink mass on flat pixels                ↓
#   pile_p99    99th pct of paths-per-pixel over inked pixels   ↓
#   pile_max    max paths-per-pixel                             ↓
#   dE          mean Lab ΔE between stroke color and source under it ↓
#   turn        mean |turning angle| between segments (rad)     ↓ (smoother)
#   len         mean path length (px)
#   svg_kb      SVG size
# Edges = top 10% of blurred-Sobel magnitude, flat = bottom 50% (per image).

include(joinpath(@__DIR__, "..", "..", "src", "EdgeFlowArt.jl"))
using .EdgeFlowArt
using Images, ImageFiltering, ImageMorphology, Statistics, Printf

const RESULTS = joinpath(@__DIR__, "..", "results")
const SIZE = 512
const POLY_RE = r"<polyline points=\"([^\"]*)\" stroke=\"(#[0-9A-Fa-f]{6})\""

function parse_svg(path)
    paths = Vector{Vector{Tuple{Float64,Float64}}}()
    colors = RGB{N0f8}[]
    for m in eachmatch(POLY_RE, read(path, String))
        pts = [
            Tuple(parse.(Float64, split(p, ","))) for
            p in split(m.captures[1]) if !isempty(p)
        ]
        push!(paths, pts)
        push!(colors, parse(RGB{N0f8}, m.captures[2]))
    end
    paths, colors
end

# Unique linear indices of pixels a polyline passes through (0.5-px DDA).
function path_pixels(pts, h, w)
    idx = Int[]
    for k = 1:(length(pts)-1)
        (x0, y0), (x1, y1) = pts[k], pts[k+1]
        n = max(1, ceil(Int, 2 * hypot(x1 - x0, y1 - y0)))
        for t = 0:n
            x = round(Int, x0 + (x1 - x0) * t / n)
            y = round(Int, y0 + (y1 - y0) * t / n)
            (1 <= x <= w && 1 <= y <= h) && push!(idx, (x - 1) * h + y)
        end
    end
    length(pts) == 1 && let (x, y) = round.(Int, pts[1])
        (1 <= x <= w && 1 <= y <= h) && push!(idx, (x - 1) * h + y)
    end
    unique!(idx)
end

function turning(pts)
    angs = Float64[]
    prev = nothing
    for k = 1:(length(pts)-1)
        d = (pts[k+1][1] - pts[k][1], pts[k+1][2] - pts[k][2])
        hypot(d...) < 1e-3 && continue
        if prev !== nothing
            a = atan(prev[1] * d[2] - prev[2] * d[1], prev[1] * d[1] + prev[2] * d[2])
            push!(angs, abs(a))
        end
        prev = d
    end
    angs
end

function metrics(svg, src)
    lab = Lab{Float64}.(src)
    gray = Float64.(Gray.(src))
    gy, gx = imgradients(imfilter(gray, Kernel.gaussian(1.0)), KernelFactors.sobel)
    E = hypot.(gx, gy)
    edge = E .> quantile(vec(E), 0.90)
    flat = E .< quantile(vec(E), 0.50)
    edge_dil = dilate(edge)
    h, w = size(src)

    paths, colors = parse_svg(svg)
    H = zeros(Int, h, w)
    dots = 0
    dEs = Float64[]
    turns = Float64[]
    lens = Float64[]
    for (pts, c) in zip(paths, colors)
        xs, ys = first.(pts), last.(pts)
        (maximum(xs) - minimum(xs) < 1 && maximum(ys) - minimum(ys) < 1) && (dots += 1)
        px = path_pixels(pts, h, w)
        H[px] .+= 1
        cl = convert(Lab{Float64}, c)
        isempty(px) || push!(dEs, mean(colordiff(cl, lab[i]) for i in px))
        append!(turns, turning(pts))
        push!(
            lens,
            sum(
                hypot(pts[k+1][1] - pts[k][1], pts[k+1][2] - pts[k][2]) for
                k = 1:(length(pts)-1);
                init=0.0,
            ),
        )
    end
    ink = H .> 0
    n_ink = max(count(ink), 1)
    (
        paths=length(paths),
        dots=dots,
        edge_prec=count(ink .& edge_dil) / n_ink,
        edge_rec=count(ink .& edge) / max(count(edge), 1),
        flat_ink=sum(H[flat]) / max(sum(H), 1),
        pile_p99=any(ink) ? quantile(H[ink], 0.99) : 0.0,
        pile_max=maximum(H),
        dE=isempty(dEs) ? NaN : mean(dEs),
        turn=isempty(turns) ? NaN : mean(turns),
        len=isempty(lens) ? NaN : mean(lens),
        svg_kb=filesize(svg) / 1024,
    )
end

function main(phases)
    isempty(phases) && error("usage: metrics.jl <phase> [<phase> ...]")
    rows = String[]
    cols = "phase,image,config,paths,dots,edge_prec,edge_rec,flat_ink,pile_p99,pile_max,dE,turn,len,svg_kb"
    sheets = Dict{String,Dict{Tuple{String,String},Matrix{RGB{N0f8}}}}()
    images = Set{String}()
    for ph in phases
        dir = joinpath(RESULTS, ph)
        for f in sort(readdir(dir))
            endswith(f, ".svg") || continue
            img, cfg = split(f[1:(end-4)], "__")
            src = EdgeFlowArt.load_square(joinpath(dir, "_inputs", "$img.png"), SIZE)
            m = metrics(joinpath(dir, f), src)
            push!(
                rows,
                join(
                    [
                        ph,
                        img,
                        cfg,
                        (
                            v isa AbstractFloat ? @sprintf("%.4f", v) : string(v) for v in m
                        )...,
                    ],
                    ",",
                ),
            )
            push!(images, img)
            png = joinpath(dir, "$(img)__$(cfg).png")
            if isfile(png)
                tiles = get!(
                    () -> Dict{Tuple{String,String},Matrix{RGB{N0f8}}}(),
                    sheets,
                    String(cfg),
                )
                tiles[(String(img), ph)] = RGB{N0f8}.(imresize(load(png), (160, 160)))
            end
            println(stderr, "  $ph $img $cfg")
        end
    end
    open(joinpath(RESULTS, "metrics.csv"), "w") do io
        println(io, cols)
        foreach(r -> println(io, r), rows)
    end
    imgs = sort(collect(images))
    for (cfg, tiles) in sheets
        blank = fill(RGB{N0f8}(0.5, 0.5, 0.5), 160, 160)
        grid = [get(tiles, (img, ph), blank) for img in imgs, ph in phases]
        sheet = reduce(vcat, [reduce(hcat, grid[i, :]) for i in axes(grid, 1)])
        save(joinpath(RESULTS, "sheet_$cfg.png"), sheet)
    end
    println(
        "wrote $(length(rows)) rows → $(joinpath(RESULTS, "metrics.csv")); sheets: $(join(keys(sheets), ", "))",
    )
end

main(ARGS)
