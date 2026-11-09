# Particle advection through a (Fx, Fy) vector field.
# Coordinate convention: (x, y) with x along cols, y along rows. Both 1-based.

using Random

const Point2 = Tuple{Float32,Float32}

# Bilinear sample of Fx, Fy at fractional (x, y). Clamps to interior.
@inline function sample_field(
    Fx::Matrix{Float32},
    Fy::Matrix{Float32},
    x::Float32,
    y::Float32,
)
    h, w = size(Fx)
    xc = clamp(x, 1.0f0, Float32(w) - 1.0f-3)
    yc = clamp(y, 1.0f0, Float32(h) - 1.0f-3)
    x0 = floor(Int, xc)
    x1 = x0 + 1
    y0 = floor(Int, yc)
    y1 = y0 + 1
    fx = xc - x0
    fy = yc - y0
    a = (1 - fx) * (1 - fy)
    b = fx * (1 - fy)
    c = (1 - fx) * fy
    d = fx * fy
    @inbounds vx = a * Fx[y0, x0] + b * Fx[y0, x1] + c * Fx[y1, x0] + d * Fx[y1, x1]
    @inbounds vy = a * Fy[y0, x0] + b * Fy[y0, x1] + c * Fy[y1, x0] + d * Fy[y1, x1]
    return vx, vy
end

# Uniform random seeds in [1, W] × [1, H].
function seed_uniform(n::Int, w::Int, h::Int, rng::AbstractRNG)::Vector{Point2}
    [
        (rand(rng, Float32) * (w - 1) + 1.0f0, rand(rng, Float32) * (h - 1) + 1.0f0) for
        _ = 1:n
    ]
end

# Integrate n particles for K steps. Returns Vector of trajectories.
# align_tangent: for tangent field the 180° ambiguity would fold curves back on
# themselves; align each step with the previous displacement to keep advection
# monotone along the streamline.
# stop_on_reversal: for dt/sobel the field points *at* an edge/ridge from both
# sides, so a particle that reaches it oscillates across it forever (zigzag +
# pile-up). Freeze it the first step its direction flips (> 90°).
# Trailing points that no longer move are trimmed from every trajectory.
function integrate(
    seeds::Vector{Point2},
    Fx::Matrix{Float32},
    Fy::Matrix{Float32},
    steps::Int,
    eta::Float32;
    align_tangent::Bool=false,
    stop_on_reversal::Bool=false,
)::Vector{Vector{Point2}}
    h, w = size(Fx)
    n = length(seeds)
    trajs = [Vector{Point2}(undef, steps + 1) for _ = 1:n]
    prev = zeros(Float32, n, 2)
    stopped = falses(n)
    for i = 1:n
        trajs[i][1] = seeds[i]
    end
    @inbounds for k = 1:steps
        for i = 1:n
            x, y = trajs[i][k]
            if stopped[i]
                trajs[i][k+1] = (x, y)
                continue
            end
            vx, vy = sample_field(Fx, Fy, x, y)
            has_prev = prev[i, 1] != 0.0f0 || prev[i, 2] != 0.0f0
            if stop_on_reversal && has_prev && vx * prev[i, 1] + vy * prev[i, 2] < 0.0f0
                stopped[i] = true
                trajs[i][k+1] = (x, y)
                continue
            end
            if align_tangent && has_prev
                if vx * prev[i, 1] + vy * prev[i, 2] < 0.0f0
                    vx = -vx
                    vy = -vy
                end
            end
            xn = clamp(x + eta * vx, 1.0f0, Float32(w))
            yn = clamp(y + eta * vy, 1.0f0, Float32(h))
            prev[i, 1] = xn - x
            prev[i, 2] = yn - y
            trajs[i][k+1] = (xn, yn)
        end
    end
    foreach(trim_static_tail!, trajs)
    return trajs
end

# Drop trailing points within 1e-3 px of their predecessor (keeps ≥ 1 point).
function trim_static_tail!(traj::Vector{Point2})
    while length(traj) > 1 &&
          abs(traj[end][1] - traj[end-1][1]) < 1.0f-3 &&
          abs(traj[end][2] - traj[end-1][2]) < 1.0f-3
        pop!(traj)
    end
    traj
end

# Paths whose bbox is smaller than this (px) are dropped as static dots.
const MIN_EXTENT = 1.0f0

# Larger bbox side of a trajectory (px).
function extent(traj::Vector{Point2})::Float32
    xs, ys = first.(traj), last.(traj)
    max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))
end
