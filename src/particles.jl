# Particle advection through a (Fx, Fy) vector field.
# Coordinate convention: (x, y) with x along cols, y along rows. Both 1-based.

using Random

const Point2 = Tuple{Float32,Float32}

# Bilinear sample of an orientation field at fractional (x, y). Clamps to
# interior. The field's sign is arbitrary (v and -v are the same direction),
# so each corner is flipped to agree with the reference (rx, ry) before
# blending; otherwise neighbours with opposite signs cancel to garbage.
# Zero reference → the top-left corner is the reference.
@inline function sample_field(
    Fx::Matrix{Float32},
    Fy::Matrix{Float32},
    x::Float32,
    y::Float32,
    rx::Float32=0.0f0,
    ry::Float32=0.0f0,
)
    h, w = size(Fx)
    xc = clamp(x, 1.0f0, Float32(w) - 1.0f-3)
    yc = clamp(y, 1.0f0, Float32(h) - 1.0f-3)
    x0 = floor(Int, xc)
    y0 = floor(Int, yc)
    fx = xc - x0
    fy = yc - y0
    if rx == 0.0f0 && ry == 0.0f0
        @inbounds rx, ry = Fx[y0, x0], Fy[y0, x0]
    end
    vx = vy = 0.0f0
    @inbounds for (dy, dx, wt) in (
        (0, 0, (1 - fx) * (1 - fy)),
        (0, 1, fx * (1 - fy)),
        (1, 0, (1 - fx) * fy),
        (1, 1, fx * fy),
    )
        cx, cy = Fx[y0+dy, x0+dx], Fy[y0+dy, x0+dx]
        s = cx * rx + cy * ry < 0.0f0 ? -wt : wt
        vx += s * cx
        vy += s * cy
    end
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

# Jobard–Lefer spacing (px): a stroke starts only where no other stroke is
# within SPACING, and stops once it comes within STOP_DIST of another one.
# Stopping closer than the seed spacing lets neighbours run side by side
# instead of cutting each other off (a cell test ended 97% of strokes early).
const SPACING = 3.0f0
const STOP_DIST = 1.5f0
# A step shorter than this (px) means the stroke is pinned at the border.
const MIN_STEP = 0.05f0
# Field magnitude below this = no direction left (flat image): end the stroke.
const MIN_FIELD = 1.0f-3

# Trace one stroke per seed through the tangent field (sign-ambiguous: each
# step is aligned with the previous one so curves don't fold back). Every step
# moves exactly `eta` px, so `steps * eta` bounds the length per direction.
# Strokes are traced one after another (seed order = random), both ways from
# the seed; seeds too close to an earlier stroke never start (1-point trajectory).
function integrate(
    seeds::Vector{Point2},
    Fx::Matrix{Float32},
    Fy::Matrix{Float32},
    steps::Int,
    eta::Float32,
)::Vector{Vector{Point2}}
    h, w = size(Fx)
    # points bucketed by SPACING-sized cells; distance tests scan the 3×3 block
    gh, gw = cld(h, Int(SPACING)) + 1, cld(w, Int(SPACING)) + 1
    grid = [Tuple{Float32,Float32,Int32}[] for _ = 1:gh, _ = 1:gw]
    cell(x, y) = (floor(Int, (y - 1) / SPACING) + 1, floor(Int, (x - 1) / SPACING) + 1)
    add!(x, y, i) = push!(grid[cell(x, y)...], (x, y, Int32(i)))
    function near(x, y, i, d)
        r, c = cell(x, y)
        for rr = max(1, r - 1):min(gh, r + 1), cc = max(1, c - 1):min(gw, c + 1)
            for (qx, qy, o) in grid[rr, cc]
                o != i && (qx - x)^2 + (qy - y)^2 < d^2 && return true
            end
        end
        return false
    end
    # Follow the streamline from (x, y) starting in direction (px, py) until it
    # nears another stroke, the field dies out, or it runs out of steps.
    function trace(x, y, px, py, i)
        pts = Point2[]
        for _ = 1:steps
            vx, vy = sample_field(Fx, Fy, x, y, px, py)
            m = hypot(vx, vy)
            m < MIN_FIELD && break
            s = (vx * px + vy * py < 0.0f0 ? -eta : eta) / m
            xn = clamp(x + s * vx, 1.0f0, Float32(w))
            yn = clamp(y + s * vy, 1.0f0, Float32(h))
            hypot(xn - x, yn - y) < MIN_STEP && break
            near(xn, yn, i, STOP_DIST) && break
            add!(xn, yn, i)
            px, py = xn - x, yn - y
            x, y = xn, yn
            push!(pts, (x, y))
        end
        return pts
    end
    trajs = [Point2[s] for s in seeds]
    # first strokes run long, later ones fill the gaps
    @inbounds for (i, (x, y)) in enumerate(seeds)
        near(x, y, i, SPACING) && continue
        add!(x, y, i)
        vx, vy = sample_field(Fx, Fy, x, y)
        fwd = trace(x, y, vx, vy, i)
        bwd = trace(x, y, -vx, -vy, i)
        trajs[i] = [reverse!(bwd); seeds[i]; fwd]
    end
    return trajs
end
