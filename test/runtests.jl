using Test
using Images
using Random
using Statistics: mean
include(joinpath(@__DIR__, "..", "src", "EdgeFlowArt.jl"))
using .EdgeFlowArt

# Small synthetic image with a clear vertical edge in the middle.
function striped(size::Int)
    img = fill(RGB{N0f8}(1, 1, 1), size, size)
    img[:, div(size, 2):end] .= RGB{N0f8}(0, 0, 0)
    return img
end

@testset "EdgeFlowArt" begin
    Random.seed!(0)

    @testset "fields" begin
        img = striped(64)
        gray = EdgeFlowArt.to_gray(img)

        Fx, Fy = EdgeFlowArt.field_tangent(gray, 1.0f0)
        @test size(Fx) == (64, 64)
        # Tangent to a vertical edge is vertical → |Fy| dominates |Fx| on that column.
        col = div(64, 2)
        @test sum(abs.(Fy[:, col])) > sum(abs.(Fx[:, col]))
    end

    @testset "structure tensor: thin line keeps a field on its centre" begin
        line = ones(Float32, 64, 64)
        line[:, 32] .= 0                                  # 1-px dark vertical line
        Fx, Fy = EdgeFlowArt.field_tangent(line, 1.5f0)
        c = hypot.(Fx, Fy)[8:56, 32]
        @test minimum(c) > 0.3                            # gradient version: ≈ 0 here
        @test sum(abs.(Fy[8:56, 32])) > 10 * sum(abs.(Fx[8:56, 32]))
    end

    @testset "orientation sampling ignores sign flips" begin
        Fx = zeros(Float32, 8, 8)
        Fy = Float32[isodd(i + j) ? 1 : -1 for i = 1:8, j = 1:8]  # ±vertical checkerboard
        vx, vy = EdgeFlowArt.sample_field(Fx, Fy, 4.5f0, 4.5f0, 0.0f0, 1.0f0)
        @test vy ≈ 1.0f0 && vx == 0.0f0
    end

    @testset "particles glide along the edge" begin
        gray = EdgeFlowArt.to_gray(striped(64))
        Fx, Fy = EdgeFlowArt.field_tangent(gray, 1.0f0)
        mask = EdgeFlowArt.edge_mask(gray, 0.15f0)
        seeds = EdgeFlowArt.seed_edges(50, mask, MersenneTwister(0))
        trajs = EdgeFlowArt.integrate(seeds, Fx, Fy, 20, 1.0f0)
        @test length(trajs) == 50
        @test all(length(t) <= 41 for t in trajs)  # ≤ steps each way + seed
        # stays on the vertical edge, moves vertically
        @test all(abs(p[1] - 32) < 3 for t in trajs for p in t)
        @test maximum(abs(t[end][2] - t[1][2]) for t in trajs) > 5
    end

    @testset "render pipeline" begin
        cfg = EdgeFlowArt.Config(size=64, steps=10, points=20, formats=Set([:svg, :png]))
        img = striped(64)
        png, svg, gif = EdgeFlowArt.render(img, cfg)
        @test size(png) == (64, 64)
        @test occursin("<svg", svg) && occursin("</svg>", svg)
        @test occursin("<polyline", svg)
        @test isempty(gif)
    end

    @testset "gif frames + anti-aliased raster" begin
        cfg = EdgeFlowArt.Config(size=64, steps=10, points=50, formats=Set([:gif, :png]))
        png, _, gif = EdgeFlowArt.render(striped(64), cfg)
        @test 2 <= length(gif) <= EdgeFlowArt.GIF_FRAMES
        @test gif[end] == png                         # last frame = final drawing
        @test count(!=(RGB{N0f8}(1, 1, 1)), gif[1]) < count(!=(RGB{N0f8}(1, 1, 1)), png)
        canvas = fill(RGB{N0f8}(1, 1, 1), 16, 16)
        EdgeFlowArt.draw_line!(canvas, 1.0, 1.0, 15.0, 5.3, RGB{N0f8}(0, 0, 0), 1.0f0)
        @test length(unique(canvas)) > 3              # partial coverage, not 2-level
    end

    @testset "soft-normalized field: fast on edges, slow on noise" begin
        Random.seed!(1)
        img = striped(64)
        noisy = [
            RGB{N0f8}(clamp(Float32(Gray(c)) + 0.02f0 * randn(Float32), 0, 1)) for c in img
        ]
        gray = EdgeFlowArt.to_gray(noisy)
        Fx, Fy = EdgeFlowArt.field_tangent(gray, 1.0f0)
        mag = hypot.(Fx, Fy)
        @test maximum(mag) <= 1.0f0
        edge = mean(mag[8:56, 31:33])        # around the stripe boundary
        flat = mean(mag[8:56, 6:20])          # noise-only region
        @test edge > 0.4
        @test flat < 0.25 * edge
    end

    @testset "spacing: no two strokes share a cell" begin
        gray = EdgeFlowArt.to_gray(striped(64))
        Fx, Fy = EdgeFlowArt.field_tangent(gray, 1.0f0)
        # many seeds on one contour: without spacing they would all overlap
        seeds = [(32.0f0, Float32(y)) for y = 10:2:30]
        trajs = EdgeFlowArt.integrate(seeds, Fx, Fy, 40, 1.0f0)
        owners = Dict{Tuple{Int,Int},Set{Int}}()
        for (i, t) in enumerate(trajs), p in (length(t) > 1 ? t : ())
            c = (floor(Int, (p[2] - 1) / 2), floor(Int, (p[1] - 1) / 2))
            push!(get!(owners, c, Set{Int}()), i)
        end
        @test all(length(v) == 1 for v in values(owners))
        @test sum(length, trajs) < length(seeds) * 41 ÷ 2   # most got cut short
    end

    @testset "path color = mean along trajectory" begin
        img = striped(64)  # white cols < 32, black cols ≥ 32
        on_white = [(Float32(x), 10.0f0) for x = 5:15]
        @test EdgeFlowArt.path_color(img, on_white) == RGB{N0f8}(1, 1, 1)
        # starts on white, ends on black: half/half → mid gray, not start color
        across = [(Float32(x), 10.0f0) for x = 22:41]
        c = EdgeFlowArt.path_color(img, across)
        @test 0.4 < Float32(c.r) < 0.6
    end

    @testset "edge seeding" begin
        gray = EdgeFlowArt.to_gray(striped(64))
        mask = EdgeFlowArt.edge_mask(gray, 0.15f0)
        seeds = EdgeFlowArt.seed_edges(200, mask, MersenneTwister(0))
        @test length(seeds) == 200
        @test all(abs(s[1] - 32) < 3 for s in seeds)  # all on the stripe boundary
        # low contrast (0.45 | 0.55): below tau, still found via the quantile cap
        faint = fill(0.45f0, 64, 64)
        faint[:, 32:end] .= 0.55f0
        @test maximum(EdgeFlowArt.sobel_xy(faint)[1]) < 0.15
        m = EdgeFlowArt.edge_mask(faint, 0.15f0)
        @test any(m) && all(any(m[:, 30:34]; dims=2))
        flat = EdgeFlowArt.seed_edges(5, falses(8, 8), MersenneTwister(0))
        @test isempty(flat)                              # no edges → no seeds
    end

    @testset "flat image draws nothing" begin
        # no edges → no seeds → empty canvas
        cfg = EdgeFlowArt.Config(size=32, steps=10, points=20, formats=Set([:svg]))
        _, svg, _ = EdgeFlowArt.render(fill(RGB{N0f8}(0.5, 0.5, 0.5), 32, 32), cfg)
        @test !occursin("<polyline", svg)
    end
end
