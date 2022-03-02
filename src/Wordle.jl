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

import DataStructures
import JSON
import ProgressMeter
import SIMD
import StaticArrays: MVector
include("simd.jl")

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

#####
##### Helpers
#####

# ASCII Characters
struct ASCIIChar
    val::UInt8
end
Base.Char(char::ASCIIChar) = Char(char.val)
Base.show(io::IO, char::ASCIIChar) = show(io, Char(char))
Base.print(io::IO, char::ASCIIChar) = print(io, Char(char))

ASCIIChar(char::Char) = ASCIIChar(convert(UInt8, char))

const AA = ASCIIChar('a')
normalize(char::ASCIIChar) = (char - ASCIIChar('a'))
Base.:-(a::ASCIIChar, b::ASCIIChar) = (a.val - b.val)

Base.zero(::Type{ASCIIChar}) = ASCIIChar(zero(UInt8))
Base.zero(::ASCIIChar) = zero(ASCIIChar)

Base.iszero(char::ASCIIChar) = iszero(char.val)
Base.:(==)(a::ASCIIChar, b::ASCIIChar) = (a.val == b.val)
Base.isless(a::ASCIIChar, b::ASCIIChar) = isless(a.val, b.val)

# Masks
struct Mask{T}
    val::T
end
Mask{T}() where {T} = Mask{T}(zero(T))

Mask(char::ASCIIChar) = Mask{UInt32}(Base.shl_int(one(UInt32), normalize(char)))

Base.:|(a::Mask{T}, b::Mask{T}) where {T} = Mask{T}(a.val | b.val)
Base.:|(mask::Mask{UInt32}, char::ASCIIChar) = mask | Mask(char)
ismatch(a::Mask{T}, b::Mask{T}) where {T} = !iszero(a.val & b.val)
ismatch(a::Mask{UInt32}, char::ASCIIChar) = ismatch(a, Mask(char))

Base.zero(::Type{Mask{T}}) where {T} = Mask{T}()
Base.zero(::T) where {T <: Mask} = zero(T)

# SIMD Bridges
bitifelse(mask::Mask, args...) = bitifelse(mask.val, args...)
bitunpack(mask::Mask) = bitunpack(mask.val)

#####
##### Utilities
#####

# Set our vectors to their default state
function clear!(x::AbstractVector{T}) where {T}
    @inbounds for i in eachindex(x)
        x[i] = zero(T)
    end
    return x
end

@inline function unsafe_match(
    v::AbstractVector{ASCIIChar},
    (char, index)::Tuple{ASCIIChar,<:Integer},
)
    val = @inbounds(v[index])
    return iszero(val) || val == char
end

@inline function unsafe_strict_match(
    v::AbstractVector{ASCIIChar},
    (char, index)::Tuple{ASCIIChar,<:Integer},
)
    return @inbounds(v[index]) == char
end

@inline function unsafe_match(
    v::AbstractVector{Mask{UInt32}},
    (mask, index)::Tuple{Mask{UInt32},<:Integer},
)
    return ismatch(@inbounds(v[index]), mask)
end

@inline function unsafe_match(
    v::AbstractVector{Mask{UInt32}},
    (char, index)::Tuple{ASCIIChar,<:Integer},
)
    return unsafe_match(v, (Mask(char), index))
end

#####
##### Discovered Knowledge
#####

mutable struct Schema{N}
    exact::MVector{N,ASCIIChar}
    misses::MVector{N,Mask{UInt32}}
    lowerbound::SIMD.Vec{32,UInt8}
    upperbound::SIMD.Vec{32,UInt8}
end

function Schema{N}() where {N}
    exact = MVector{N,ASCIIChar}(undef)
    misses = MVector{N,Mask{UInt32}}(undef)
    lowerbound = SIMD.Vec{32,UInt8}(0x00)
    upperbound = SIMD.Vec{32,UInt8}(0xff)
    schema = Schema{N}(exact, misses, lowerbound, upperbound)
    empty!(schema)
    return schema
end

function Base.empty!(schema::Schema)
    clear!(schema.exact)
    clear!(schema.misses)
    return schema
end

# Filtering
compare(f::F, x::SIMD.Vec{32,UInt8}, y::SIMD.Vec{32,UInt8}) where {F} = sum(f(x, y)) == 32
function (f::Schema)(s::Union{AbstractString,Tuple})
    (; exact, misses, lowerbound, upperbound) = f
    scratch = SIMD.Vec{32,UInt8}(0x00)

    # Bit mask for characters processed for bounds checking.
    for (index, char) in enumerate(s)
        # If this isn't match a known hit - return false.
        # If we know this character does not belong at this index, also return false
        if !unsafe_match(exact, (char, index)) || unsafe_match(misses, (char, index))
            return false
        end

        # Increment the character count
        scratch += bitunpack(Mask(char))
    end
    # Perform bounds checks
    return compare(>=, scratch, lowerbound) & compare(<=, scratch, upperbound)
end

function merge!(a::Schema{N}, b::Schema{N}) where {N}
    # Merge exact matches and misses
    for i in Base.OneTo(N)
        b_exact = b.exact[i]
        if !iszero(b_exact)
            a.exact[i] = b_exact
        end
        a.misses[i] |= b.misses[i]
    end

    # Merge upper and lower bounds
    a.lowerbound = max(a.lowerbound, b.lowerbound)
    a.upperbound = min(a.upperbound, b.upperbound)
    return a
end

#####
##### Result Logic
#####

@enum States::UInt8 Gray = 1 Yellow = 2 Green = 4
function result_schema(guess, states::NTuple{N,States}) where {N}
    return result_schema!(Schema{N}(), guess, states)
end

function result_schema!(
    schema::Schema{N},
    guess::AbstractString,
    states::NTuple{N,States},
) where {N}
    @assert length(guess) == N
    return result_schema!(schema, ntuple(i -> ASCIIChar(guess[i]), Val(N)), states)
end

function result_schema!(schema::Schema{N}, guess, states::NTuple{N,States}) where {N}
    empty!(schema)
    (; exact, misses) = schema
    lowerbound = SIMD.Vec{32,UInt8}(0x00)
    upperbound = SIMD.Vec{32,UInt8}(0xff)

    # Process correct guesses first, then process incorrect guesses.
    # This makes it easier to deal with incorrect guesses that repeat a letter that
    # was correct.
    @inbounds for i in Base.OneTo(N)
        state = states[i]
        char = guess[i]
        if state == Green
            exact[i] = char
            lowerbound += bitunpack(Mask(char))
        elseif state == Yellow
            misses[i] |= char
            lowerbound += bitunpack(Mask(char))
        end
    end

    @inbounds for i in Base.OneTo(N)
        state = states[i]
        if state == Gray
            char = guess[i]
            # Add this to the "misses" list and clamp the upperbound to the exact number
            # of occurances.
            misses[i] |= char
            upperbound = bitifelse(Mask(char), lowerbound, upperbound)
        end
    end
    schema.lowerbound = lowerbound
    schema.upperbound = upperbound
    return schema
end

#####
##### Result Generator
#####

struct PossibleStates
    val::UInt8
end

function PossibleStates(states::States...)
    val = zero(UInt8)
    for state in states
        val |= UInt8(state)
    end
    return PossibleStates(val)
end

Base.show(io::IO, x::PossibleStates) = print(io, Tuple(collect(x)))
Base.length(x::PossibleStates) = count_ones(x.val)
Base.eltype(::Type{PossibleStates}) = States
function Base.iterate(x::PossibleStates, i = 0)
    (; val) = x
    while iszero(val & (Base.shl_int(1, i)))
        i += 1
        (i > 2) && return nothing
    end
    state = (i == 0) ? Gray : (i == 1 ? Yellow : Green)
    return state, i + 1
end

function generate(schema::Schema, guess::NTuple{N,ASCIIChar}) where {N}
    (; upperbound, exact, misses) = schema
    return ntuple(Val(N)) do i
        # If this is an exact match, then the only possibility is green.
        char = @inbounds(guess[i])
        if unsafe_strict_match(exact, (char, i))
            return PossibleStates(Green)
        end

        # Check if this character is a known non-existant character.
        # This means that there is a max of 0 entries for this character.
        # If so, it can only be Gray.
        if iszero(upperbound[char - AA + 1])
            return PossibleStates(Gray)
        end

        # Now check if this is forced to be either Gray or Yellow
        # Basically, if we know there is another exact match for this position, than this
        # index cannot be green.
        #
        # This can also be forced to be either Gray or Yellow if we know this character
        # does not belong in this possition.
        if !iszero(exact[i]) || unsafe_match(misses, (char, i))
            return PossibleStates(Yellow, Gray)
        end

        # Default case, entry can be anything.
        return PossibleStates(Green, Yellow, Gray)
    end
end

# Use this to filter out results generated by `generate_possibilities` that still validate
# the schema.
#
# This mostly used to handle cases where we have exactly one instance of a letter.
@inline function ispossible(
    schema::Schema,
    guess::NTuple{N,ASCIIChar},
    result::NTuple{N,States},
) where {N}
    # Discard repeat yellows
    #seen = SIMD.Vec{32,UInt8}(0x00)
    seen = Mask{UInt32}()
    scratch = SIMD.Vec{32,UInt8}(0x00)
    @inbounds for i in Base.OneTo(N)
        state, char = result[i], guess[i]
        setscratch = false
        if state == Gray
            seen |= char
            #seen += bitunpack(Base.shl_int(one(UInt32), (char - AA)))
        elseif state == Yellow
            ismatch(seen, char) && return false
            # If this is a Yellow following a Gray, than this should not be possible.
            #iszero(seen[char - AA + 1]) || return false
            setscratch = true
        else
            setscratch = true
        end
        setscratch && (scratch += bitunpack(Mask(char)))
    end
    (; lowerbound, upperbound) = schema
    return compare(>=, scratch, lowerbound) && compare(<=, scratch, upperbound)
end

struct ResultIter{N,I}
    schema::Schema{N}
    guess::NTuple{N,ASCIIChar}
    iter::I
end

function ResultIter(schema::Schema{N}, guess::NTuple{N}) where {N}
    iter = Iterators.product(generate(schema, guess)...)
    return ResultIter(schema, guess, iter)
end
Base.IteratorSize(::Type{<:ResultIter}) = Base.SizeUnknown()

function Base.iterate(ri::ResultIter, s...)
    (; schema, guess, iter) = ri
    y = iterate(iter, s...)
    y === nothing && return nothing
    (v, s) = y
    while !ispossible(schema, guess, v)
        y = iterate(iter, s)
        y === nothing && return nothing
        (v, s) = y
    end
    return v, s
end

#####
##### Count possibilies
#####

function countmatches(
    f::F,
    schema::Schema{N},
    guess::NTuple{N,ASCIIChar},
    dictionary::Union{AbstractVector,DataStructures.OrderedSet};
    tempschema::Schema = Schema{N}(),
    wordcallback::G = (_...) -> nothing,
    iter = ResultIter(schema, guess),
) where {F,N,G}
    for states in iter
        result_schema!(tempschema, guess, states)
        merge!(tempschema, schema)

        partitionsize = 0
        for (i, word) in enumerate(dictionary)
            if tempschema(word)
                wordcallback(i)
                partitionsize += 1
            end
        end

        if !iszero(partitionsize)
            proceed = f(partitionsize)
            proceed || return nothing
        end
    end
    return nothing
end

#####
##### Top level processing
#####

cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T<:Integer} = one(T) + div(a - one(T), b)

function process_dictionary(
    schema::Schema{N},
    dictionary;
    target = DataStructures.OrderedSet(dictionary),
    batchsize = 16,
) where {N}
    scores = Vector{Tuple{Int64,Float32}}(undef, length(dictionary))

    threads = Base.OneTo(Threads.nthreads())
    counts_tls = [Vector{Int}() for _ in threads]
    tempschema_tls = [Schema{N}() for _ in threads]
    seen_tls = [falses(length(target)) for _ in threads]

    meter = ProgressMeter.Progress(length(dictionary), 1)
    best_bound = Threads.Atomic{Int}(typemax(Int))
    workcount = Threads.Atomic{Int}(1)
    numbatches = cdiv(length(dictionary), batchsize)

    #for tid in Base.OneTo(Threads.nthreads())
    Threads.@threads for tid in Base.OneTo(Threads.nthreads())
        maxpartition = Ref(0)
        tempschema = tempschema_tls[tid]
        seen = seen_tls[tid]
        counts = counts_tls[tid]
        while true
            # Get this threads work load
            k = Threads.atomic_add!(workcount, 1)
            k > numbatches && break

            start = (k - 1) * batchsize + 1
            stop = min(k * batchsize, length(dictionary))


            # Process this batch
            for i = start:stop
                seen .= false
                maxpartition[] = 0
                empty!(counts)
                wordcallback(i) = (seen[i] = true)

                countmatches(
                    schema,
                    dictionary[i],
                    target;
                    tempschema,
                    wordcallback,
                ) do partitionsize
                    push!(counts, partitionsize)
                    maxpartition[] = max(maxpartition[], partitionsize)
                    # Only continue if this partition is within the best bound.
                    return partitionsize <= best_bound[]
                end

                # Handle any words that aren't covered by entering this guess
                missed = length(target) - count(seen)
                iszero(missed) || push!(counts, missed)
                maxpartition_unbox = max(maxpartition[], missed)
                if maxpartition_unbox < best_bound[]
                    Threads.atomic_min!(best_bound, maxpartition_unbox)
                end

                scores[i] = (maxpartition_unbox, mean(counts))
            end
            tid == 1 && ProgressMeter.update!(meter, stop)
        end
    end
    ProgressMeter.finish!(meter)
    return scores
end

# Initial filtering

function worklist_lt(a::NTuple{N}, b::NTuple{N}) where {N}
    gray_a = count(isequal(Gray), a)
    gray_b = count(isequal(Gray), b)
    yellow_a = count(isequal(Yellow), a)
    yellow_b = count(isequal(Yellow), b)
    if gray_a > gray_b
        return true
    elseif gray_a == gray_b && yellow_a > yellow_b
        return true
    end
    return false
end

function init_worklist(::Val{N}) where {N}
    states = (Gray, Yellow, Green)
    worklist = vec(collect(Iterators.product(ntuple(_ -> states, Val(N))...)))
    sort!(worklist; lt = worklist_lt)
    return worklist
end

function process_dictionary_init(
    dictionary::AbstractVector{<:NTuple{N}};
    batchsize = 16,
) where {N}
    schema = Schema{N}()
    scores = Vector{Tuple{Int64,Float32}}(undef, length(dictionary))

    threads = Base.OneTo(Threads.nthreads())
    tempschema_tls = [Schema{N}() for _ in threads]
    counts_tls = [Vector{Int}() for _ in threads]
    seen_tls = [falses(length(dictionary)) for _ in threads]

    meter = ProgressMeter.Progress(length(dictionary), 1)
    best_bound = Threads.Atomic{Int}(typemax(Int))
    workcount = Threads.Atomic{Int}(1)
    numbatches = cdiv(length(dictionary), batchsize)
    worklist = init_worklist(Val(N))

    #for tid in Base.OneTo(Threads.nthreads())
    Threads.@threads for tid in Base.OneTo(Threads.nthreads())
        maxpartition = Ref(0)
        tempschema = tempschema_tls[tid]
        seen = seen_tls[tid]
        counts = counts_tls[tid]
        while true
            # Get this threads work load
            k = Threads.atomic_add!(workcount, 1)
            k > numbatches && break

            start = (k - 1) * batchsize + 1
            stop = min(k * batchsize, length(dictionary))

            # Process this batch
            for i = start:stop
                seen .= false
                maxpartition[] = 0
                empty!(counts)

                wordcallback(i) = (seen[i] = true)
                guess = dictionary[i]
                countmatches(
                    schema,
                    guess,
                    dictionary;
                    tempschema,
                    wordcallback,
                    iter = Iterators.filter(x -> ispossible(schema, guess, x), worklist),
                ) do partitionsize
                    push!(counts, partitionsize)
                    maxpartition[] = max(maxpartition[], partitionsize)
                    # Only continue if we're stiller better than the best bound so far.
                    return maxpartition[] <= best_bound[]
                end

                # Handle any words that aren't covered by entering this guess
                missed = length(dictionary) - count(seen)
                iszero(missed) || push!(counts, missed)
                maxpartition_unbox = max(maxpartition[], missed)
                if maxpartition_unbox <= best_bound[]
                    Threads.atomic_min!(best_bound, maxpartition_unbox)
                end

                scores[i] = (maxpartition_unbox, mean(counts))
            end
            tid == 1 && ProgressMeter.update!(meter, stop)
        end
    end
    ProgressMeter.finish!(meter)
    return scores
end

#####
##### Allow for sorting an array based on scores held in another array.
#####

struct Glue{A,B} <: AbstractVector{Tuple{A,B}}
    a::Vector{A}
    b::Vector{B}
end
Base.size(g::Glue) = size(g.a)
Base.getindex(g::Glue, i::Int) = (g.a[i], g.b[i])
function Base.setindex!(g::Glue, (a, b), i::Int)
    return (g.a[i], g.b[i]) = (a, b)
end

include("terminal.jl")

end # module
