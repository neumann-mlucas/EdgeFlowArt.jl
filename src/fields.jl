# Image-derived tangent field for particle advection + edge mask for seeding.
# Field is (Fx, Fy), size (H, W), Float32, length < 1 scaled by gradient strength.

using ImageFiltering
using Images: Gray
using Statistics: quantile

to_gray(img::AbstractMatrix)::Matrix{Float32} = Float32.(Gray.(img))

# Soft-normalize in-place: F / (|F| + ε), ε = SOFT_EPS_Q99 × 99th-pct magnitude.
# Strong edges move at ≥ ~0.67 px/step, flat regions (noise-level gradient)
# barely move instead of amplifying noise into full-speed scribbles.
const SOFT_EPS_Q99 = 0.5f0
function normalize_field!(Fx::Matrix{Float32}, Fy::Matrix{Float32})
    M = @. sqrt(Fx^2 + Fy^2)
    eps = SOFT_EPS_Q99 * quantile(vec(M), 0.99)
    @inbounds for i in eachindex(Fx)
        m = M[i]
        if m > 1.0f-6
            Fx[i] /= (m + eps)
            Fy[i] /= (m + eps)
        else
            Fx[i] = 0.0f0
            Fy[i] = 0.0f0
        end
    end
    return nothing
end

# Sobel gradients. Returns (Gx, Gy): Gx = derivative along cols (x), Gy = along rows (y).
function sobel_xy(gray::Matrix{Float32})
    grads = imgradients(gray, KernelFactors.sobel)
    Gy = Float32.(grads[1])   # axis 1 = rows = y
    Gx = Float32.(grads[2])   # axis 2 = cols = x
    return Gx, Gy
end

# Edge pixels (seed sites): Sobel magnitude above min(tau, top-EDGE_TOP
# quantile), so low-contrast images still get edges while tau caps how much
# of a high-contrast image counts.
const EDGE_TOP = 0.10
function edge_mask(gray::Matrix{Float32}, tau::Float32)::BitMatrix
    Gx, Gy = sobel_xy(gray)
    E = @. sqrt(Gx^2 + Gy^2)
    return E .> min(tau, quantile(vec(E), 1 - EDGE_TOP))
end

# Tangent-flow field: perpendicular to gradient. Particles glide along edges.
# ponytail: skips full structure-tensor eigendecomp + coherence weighting;
# upgrade path is smoothed J = G_σ * (∇I)(∇I)ᵀ then eig per pixel.
function field_tangent(gray::Matrix{Float32}, sigma::Float32)
    smoothed = imfilter(gray, Kernel.gaussian(sigma))
    Gx, Gy = sobel_xy(Float32.(smoothed))
    # Tangent = 90° rotation of gradient
    Fx = -Gy
    Fy = Gx
    normalize_field!(Fx, Fy)
    return Fx, Fy
end
