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

    @testset "dead field ends strokes; live field moves eta per step" begin
        seeds = [(10.0f0, 10.0f0), (30.0f0, 30.0f0)]
        slow = EdgeFlowArt.integrate(
            seeds,
            fill(1.0f-4, 40, 40),
            zeros(Float32, 40, 40),
            50,
            1.0f0,
        )
        @test all(length(t) == 1 for t in slow)          # |F| < MIN_FIELD
        fast = EdgeFlowArt.integrate(
            seeds,
            fill(0.5f0, 40, 40),
            zeros(Float32, 40, 40),
            10,
            1.0f0,
        )
        @test all(length(t) > 1 for t in fast)
        @test all(t[k] != t[k+1] for t in fast for k = 1:(length(t)-1))
        # weak field (0.5) still steps a full eta: unit speed
        @test all(
            hypot(t[k+1][1] - t[k][1], t[k+1][2] - t[k][2]) ≈ 1.0f0 for t in fast for
            k = 1:(length(t)-1) if 1 < t[k+1][1] < 40
        )
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
        ink(c) = count(!=(RGB{N0f8}(1, 1, 1)), c)
        @test length(gif) == EdgeFlowArt.GIF_FRAMES + 1
        @test issorted(ink.(gif))                     # strokes only grow
        @test ink(gif[1]) < 0.5 * ink(png)            # frame 1 = seed dots
        @test abs(ink(gif[end]) - ink(png)) <= 0.05 * ink(png)  # fully traced
        # one stroke, seed in the middle: grows both ways from the seed
        t = [(Float32(x), 8.0f0) for x = 2:30]
        fr = EdgeFlowArt.raster_growth([t], [15], [RGB{N0f8}(0, 0, 0)], 32, 16; frames=4)
        cols(c) = findall(!=(RGB{N0f8}(1, 1, 1)), c[8, :])
        sym(c) = 16 - minimum(c) == maximum(c) - 16   # centred on t[15] = (16, 8)
        @test sym(cols(fr[1])) && length(cols(fr[1])) <= 5
        @test sym(cols(fr[3])) && length(cols(fr[3])) > length(cols(fr[1]))
        @test issubset(2:30, cols(fr[end]))
        # anti-aliased: partial coverage at the stroke rim, not 2-level
        canvas = fill(RGB{N0f8}(1, 1, 1), 16, 16)
        cov = zeros(Float32, 16, 16)
        line = [(1.0f0, 1.0f0), (15.0f0, 5.3f0)]
        EdgeFlowArt.draw_stroke!(canvas, cov, line, RGB{N0f8}(0, 0, 0), 1.0f0, 1.0f0)
        @test length(unique(canvas)) > 3
        @test all(iszero, cov)                        # scratch buffer reset
        # overlapping stamps don't darken the stroke itself (blend once per pixel)
        half = fill(RGB{N0f8}(1, 1, 1), 16, 16)
        EdgeFlowArt.draw_stroke!(half, cov, line, RGB{N0f8}(0, 0, 0), 0.5f0, 3.0f0)
        @test minimum(Float32(Gray(c)) for c in half) >= 0.49
    end

    @testset "tone width: dark strokes wider" begin
        tw(c) = EdgeFlowArt.tone_width(1.0, c)
        @test tw(RGB{N0f8}(0, 0, 0)) > 10 * tw(RGB{N0f8}(1, 1, 1))
        canvas(c) = EdgeFlowArt.raster_polylines(
            [[(2.0f0, 16.0f0), (30.0f0, 16.0f0)]],
            [c],
            32,
            32;
            opacity=1.0f0,
            stroke_width=1.2,
        )
        ink(c) = count(!=(RGB{N0f8}(1, 1, 1)), canvas(c))
        @test ink(RGB{N0f8}(0, 0, 0)) > 2 * ink(RGB{N0f8}(0.8, 0.8, 0.8))
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

    @testset "spacing: strokes never come within STOP_DIST of each other" begin
        # concentric swirl: many strokes run side by side
        Fx = Float32[-(i - 32) for i = 1:64, j = 1:64]
        Fy = Float32[j - 32 for i = 1:64, j = 1:64]
        rng = MersenneTwister(0)
        seeds = [(1 + 63 * rand(rng, Float32), 1 + 63 * rand(rng, Float32)) for _ = 1:300]
        trajs = EdgeFlowArt.integrate(seeds, Fx, Fy, 200, 1.0f0)
        live = [t for t in trajs if length(t) > 1]
        @test length(live) > 10
        gap = minimum(
            hypot(p[1] - q[1], p[2] - q[2]) for a = 1:length(live) for
            b = (a+1):length(live) for p in live[a] for q in live[b];
            init=Inf32,
        )
        @test gap >= EdgeFlowArt.STOP_DIST
        @test count(==(1) ∘ length, trajs) > 0   # seeds next to a stroke never start
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
