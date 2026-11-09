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

# Seeds on random edge pixels (± 0.5 px jitter); none if there are no edges.
function seed_edges(n::Int, mask::BitMatrix, rng::AbstractRNG)::Vector{Point2}
    h, w = size(mask)
    idx = findall(mask)
    isempty(idx) && return Point2[]
    map(rand(rng, idx, n)) do c
        (
            clamp(c[2] + rand(rng, Float32) - 0.5f0, 1.0f0, Float32(w)),
            clamp(c[1] + rand(rng, Float32) - 0.5f0, 1.0f0, Float32(h)),
        )
    end
end

# Integrate n particles for K steps through the tangent field. Returns Vector
# of trajectories. The field has a 180° ambiguity that would fold curves back on
# themselves, so each step is aligned with the previous displacement to keep
# advection monotone along the streamline.
# Occupancy cell size (px). A particle stops on entering a cell another particle
# already drew through, so strokes don't stack on the same contour (pile-up).
# Seeds landing in a taken cell never start (1-point trajectory).
# Strokes are drawn one after another, not in lockstep.
# ponytail: fixed-grid first-come ownership, Jobard–Lefer distance test if
# grid-aligned gaps show.
const SPACING = 2.0f0

function integrate(
    seeds::Vector{Point2},
    Fx::Matrix{Float32},
    Fy::Matrix{Float32},
    steps::Int,
    eta::Float32,
)::Vector{Vector{Point2}}
    h, w = size(Fx)
    n = length(seeds)
    owner = zeros(Int32, cld(h, Int(SPACING)) + 1, cld(w, Int(SPACING)) + 1)
    cell(x, y) =
        CartesianIndex(floor(Int, (y - 1) / SPACING) + 1, floor(Int, (x - 1) / SPACING) + 1)
    # Follow the streamline from (x, y) starting in direction (px, py) until
    # it enters a cell owned by another stroke or runs out of steps.
    function trace(x, y, px, py, i)
        pts = Point2[]
        for _ = 1:steps
            vx, vy = sample_field(Fx, Fy, x, y)
            if vx * px + vy * py < 0.0f0
                vx = -vx
                vy = -vy
            end
            xn = clamp(x + eta * vx, 1.0f0, Float32(w))
            yn = clamp(y + eta * vy, 1.0f0, Float32(h))
            c = cell(xn, yn)
            owner[c] == 0 || owner[c] == i || break
            owner[c] = i
            px, py = xn - x, yn - y
            x, y = xn, yn
            push!(pts, (x, y))
        end
        return pts
    end
    trajs = [Point2[s] for s in seeds]
    # one full stroke at a time (seed order = random): first strokes run long,
    # later ones fill the gaps; a seed in a taken cell is skipped. Each stroke
    # is traced both ways from its seed.
    @inbounds for i = 1:n
        x, y = seeds[i]
        c = cell(x, y)
        owner[c] == 0 || continue
        owner[c] = i
        vx, vy = sample_field(Fx, Fy, x, y)
        fwd = trace(x, y, vx, vy, i)
        bwd = trace(x, y, -vx, -vy, i)
        trajs[i] = [reverse!(bwd); seeds[i]; fwd]
    end
    return trajs
end
