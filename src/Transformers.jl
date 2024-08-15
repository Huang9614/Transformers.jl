module Transformers

using Flux

using NeuralAttentionlib

export Transformer

export todevice, enable_gpu
export Layers, TextEncoders, HuggingFace,
    Masks

const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

include("./device.jl")
include("./loss.jl")

include("./layers/Layers.jl")
include("./tokenizer/tokenizer.jl")
include("./textencoders/TextEncoders.jl")

include("./datasets/Datasets.jl")
include("./huggingface/HuggingFace.jl")

using .Layers
using .TextEncoders
using .Datasets

using .HuggingFace

function selfTry()
    println("Trying works!")
end

export selfTry
end # module
