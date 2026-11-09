module EdgeFlowArtMain

include("src/EdgeFlowArt.jl")

using .EdgeFlowArt
using ArgParse
using FileIO
using Images
using Random

function main()
    args = parse_cmd()
    EdgeFlowArt.setup_logging(args)

    outputs = EdgeFlowArt.resolve_output(args["output"])
    formats = Set(fmt for (_, fmt) in outputs)

    # ponytail: no GIF yet (render returns no frames); fail loud instead of writing nothing
    :gif in formats && error("GIF output not supported yet; use .png or .svg")

    cfg = EdgeFlowArt.Config(
        size=args["size"],
        steps=args["steps"],
        points=args["points"],
        eta=Float32(args["eta"]),
        tau=Float32(args["tau"]),
        sigma=Float32(args["sigma"]),
        stroke_width=args["stroke-width"],
        opacity=args["opacity"],
        seed=args["seed"],
        formats=formats,
    )
    ts = cfg.timer
    Random.seed!(cfg.seed)

    input_path = args["input"]
    @info ts() * "Loading input '$input_path'"
    input = EdgeFlowArt.load_input(input_path, cfg.size)

    @info ts() * "Running edge-flow..."
    png, svg, gif = EdgeFlowArt.render(input, cfg)

    for (path, fmt) in outputs
        @info ts() * "Saving output '$path' ($fmt)"
        if fmt == :png
            save(path, png)
        elseif fmt == :svg
            EdgeFlowArt.write_svg(path, svg)
        elseif fmt == :gif
            EdgeFlowArt.save_gif(path, gif)
        end
    end

    @info ts() * "Done"
end

function parse_cmd()
    parser = ArgParseSettings(
        description="EdgeFlowArt - image → flow-line rendering",
        epilog="Example: julia main.jl -i input.jpg -o out.svg --points 5000",
    )
    EdgeFlowArt.add_common_args!(parser; steps_default=200)
    @add_arg_table! parser begin
        "--size", "-s"
        help = "output canvas size in pixels"
        arg_type = Int
        default = 512
        "--points"
        help = "particle count"
        arg_type = Int
        default = 3000
        "--eta"
        help = "step size (pixels per iteration)"
        arg_type = Float64
        default = 1.0
        "--tau"
        help = "seeding: edge-magnitude threshold (particles start on pixels above it)"
        arg_type = Float64
        default = 0.15
        "--sigma"
        help = "tangent field: Gaussian pre-blur sigma (px)"
        arg_type = Float64
        default = 1.5
        "--stroke-width"
        help = "SVG stroke width"
        arg_type = Float64
        default = 0.6
        "--opacity"
        help = "stroke opacity in [0, 1]"
        arg_type = Float64
        default = 0.6
        "--seed"
        help = "RNG seed"
        arg_type = Int
        default = 42
    end
    parse_args(parser)
end

end

EdgeFlowArtMain.main()
