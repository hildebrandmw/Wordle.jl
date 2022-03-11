module Wordle


#####
##### Exports
#####

export Green, Yellow, Gray
export PossibleStates

#####
##### deps
#####

using Statistics
import Random

import DataStructures
import JSON
import ProgressMeter
import SIMD
import StaticArrays: MVector

include("simd.jl")
include("utils.jl")
include("sorted.jl")
include("impl1.jl")
include("impl2.jl")
include("terminal.jl")

#####
##### Dictionary Loading
#####

function readwords(file::AbstractString)
    _, ext = splitext(file)
    if ext == ".txt"
        return open(io -> readwords(io), file; read = true)
    elseif ext == ".json"
        return open(JSON.parse, file; read = true)
    else
        error("Unknown file extension: $ext")
    end
end

readwords(io::IO) = collect(eachline(io))
function tolength(len, words::Vector{Any})
    if all(x -> isa(x, AbstractString), words)
        return tolength(len, identity.(words))
    end
    error("Parsed dictionary contains non-string elements!")
end

tolength(len, words::Vector{String}) = filter(x -> length(x) == len, words)


end # module
