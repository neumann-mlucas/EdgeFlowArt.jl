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

        Fx, Fy = EdgeFlowArt.field_sobel(gray)
        @test size(Fx) == (64, 64) && size(Fy) == (64, 64)
        # Field on/near the vertical edge column is non-zero somewhere.
        @test any(abs.(Fx[:, div(64, 2)]) .> 0)

        Fx, Fy = EdgeFlowArt.field_dt(gray, 0.15f0)
        @test size(Fx) == (64, 64)
        # DT is defined everywhere → non-trivial field off-edge.
        @test any(abs.(Fx) .+ abs.(Fy) .> 0)

        Fx, Fy = EdgeFlowArt.field_tangent(gray, 1.0f0)
        @test size(Fx) == (64, 64)
        # Tangent to a vertical edge is vertical → |Fy| dominates |Fx| on that column.
        col = div(64, 2)
        @test sum(abs.(Fy[:, col])) > sum(abs.(Fx[:, col]))
    end

    @testset "particles" begin
        img = striped(64)
        gray = EdgeFlowArt.to_gray(img)
        Fx, Fy = EdgeFlowArt.field_dt(gray, 0.15f0)
        rng = MersenneTwister(0)
        seeds = EdgeFlowArt.seed_uniform(50, 64, 64, rng)
        trajs = EdgeFlowArt.integrate(seeds, Fx, Fy, 20, 1.0f0)
        @test length(trajs) == 50
        @test all(length(t) == 21 for t in trajs)
        # Particles should end closer (on avg) to the vertical edge column.
        edge_col = div(64, 2)
        d_start = mean(abs(t[1][1] - edge_col) for t in trajs)
        d_end   = mean(abs(t[end][1] - edge_col) for t in trajs)
        @test d_end <= d_start
    end

    @testset "render pipeline" begin
        cfg = EdgeFlowArt.Config(
            size = 64, steps = 10, points = 20, field = :tangent,
            formats = Set([:svg, :png]),
        )
        img = striped(64)
        png, svg, gif = EdgeFlowArt.render(img, cfg)
        @test size(png) == (64, 64)
        @test occursin("<svg", svg) && occursin("</svg>", svg)
        @test occursin("<polyline", svg)
        @test isempty(gif)
    end
end
