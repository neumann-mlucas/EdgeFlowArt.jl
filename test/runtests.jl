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

    @testset "particles glide along the edge" begin
        gray = EdgeFlowArt.to_gray(striped(64))
        Fx, Fy = EdgeFlowArt.field_tangent(gray, 1.0f0)
        mask = EdgeFlowArt.edge_mask(gray, 0.15f0)
        seeds = EdgeFlowArt.seed_edges(50, mask, MersenneTwister(0))
        trajs = EdgeFlowArt.integrate(seeds, Fx, Fy, 20, 1.0f0)
        @test length(trajs) == 50
        @test all(length(t) == 21 for t in trajs)
        # stays on the vertical edge, moves vertically
        @test all(abs(p[1] - 32) < 3 for t in trajs for p in t)
        @test mean(abs(t[end][2] - t[1][2]) for t in trajs) > 5
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
