# Image-derived vector fields for particle advection.
# All fields return (Fx, Fy) matrices of size (H, W), Float32, unit-length
# (zero where the field is degenerate).

using ImageFiltering
using ImageMorphology
using Images: Gray

to_gray(img::AbstractMatrix)::Matrix{Float32} = Float32.(Gray.(img))

# Normalize per-pixel to unit length in-place. Zeros stay zero.
function normalize_field!(Fx::Matrix{Float32}, Fy::Matrix{Float32})
    @inbounds for i in eachindex(Fx)
        m = sqrt(Fx[i]^2 + Fy[i]^2)
        if m > 1f-6
            Fx[i] /= m
            Fy[i] /= m
        else
            Fx[i] = 0f0
            Fy[i] = 0f0
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

# Sobel-climb field: F = ∇|∇I|. Points climb edge-magnitude ridge.
function field_sobel(gray::Matrix{Float32})
    Gx, Gy = sobel_xy(gray)
    E = @. sqrt(Gx^2 + Gy^2)
    Ex, Ey = sobel_xy(E)
    normalize_field!(Ex, Ey)
    return Ex, Ey
end

# DT-descent field: threshold Sobel, distance-transform, F = -∇D. Every
# pixel pulled to nearest edge.
function field_dt(gray::Matrix{Float32}, tau::Float32)
    Gx, Gy = sobel_xy(gray)
    E = @. sqrt(Gx^2 + Gy^2)
    B = E .> tau
    Fx = zeros(Float32, size(gray))
    Fy = zeros(Float32, size(gray))
    if !any(B)
        return Fx, Fy
    end
    # feature_transform on inverted mask: for each non-edge pixel, index of nearest edge.
    ft = feature_transform(B)
    D = Float32.(distance_transform(ft))
    Dx, Dy = sobel_xy(D)
    @. Fx = -Dx
    @. Fy = -Dy
    normalize_field!(Fx, Fy)
    return Fx, Fy
end

# Tangent-flow field: perpendicular to gradient. Particles glide along edges.
# ponytail: skips full structure-tensor eigendecomp + coherence weighting;
# upgrade path is smoothed J = G_σ * (∇I)(∇I)ᵀ then eig per pixel.
function field_tangent(gray::Matrix{Float32}, sigma::Float32)
    smoothed = imfilter(gray, Kernel.gaussian(sigma))
    Gx, Gy = sobel_xy(Float32.(smoothed))
    # Tangent = 90° rotation of gradient
    Fx = -Gy
    Fy =  Gx
    normalize_field!(Fx, Fy)
    return Fx, Fy
end

# Dispatch on mode symbol.
function build_field(gray::Matrix{Float32}, mode::Symbol, params::NamedTuple)
    if mode === :sobel
        return field_sobel(gray)
    elseif mode === :dt
        return field_dt(gray, Float32(params.tau))
    elseif mode === :tangent
        return field_tangent(gray, Float32(params.sigma))
    else
        error("unknown field mode: $mode (expected :sobel, :dt, :tangent)")
    end
end
