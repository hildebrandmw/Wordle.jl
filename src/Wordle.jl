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
##### Utilities
#####

# Set our vectors to their default state
function clear!(x::AbstractVector{T}) where {T}
    @inbounds for i in eachindex(x)
        x[i] = zero(T)
    end
    return x
end

function clear!(x::AbstractVector{Char})
    @inbounds for i in eachindex(x)
        x[i] = ' '
    end
end

# Bitmask Utilities
@inline function isset(v::I, c::Char) where {I <: Integer}
    return !iszero(v & (one(I) << (c - 'a')))
end
@inline set(v::I, c::Char) where {I <: Integer} = v | (one(I) << (c - 'a'))

# Matching
@inline function unsafe_match(v::AbstractVector{Char}, (char, index)::Tuple{Char,<:Integer})
    val = @inbounds(v[index])
    return val == ' ' || val == char
end

@inline function unsafe_strict_match(v::AbstractVector{Char}, (char, index)::Tuple{Char,<:Integer})
    return @inbounds(v[index]) == char
end

@inline function unsafe_match(v::AbstractVector{UInt32}, (char, index)::Tuple{Char,<:Integer})
    return isset(@inbounds(v[index]), char)
end

# # Counter Utilities
# isvalid(x::UInt8) = (x != typemax(UInt8))
# unsafe_getcount(v::Vector{UInt8}, c::Char) = @inbounds(v[(c - 'a') + 1])
# function unsafe_inccount!(v::Vector{UInt8}, c::Char)
#     i = (c - 'a') + 1
#     val = @inbounds(v[i])
#     return @inbounds(v[i] = isvalid(val) ? val + one(val) : one(val))
# end
#
# function unsafe_setcount!(v::Vector{UInt8}, count, c::Char)
#     i = (c - 'a') + 1
#     return @inbounds(v[i] = count)
# end

#####
##### Discovered Knowledge
#####

mutable struct Schema{N}
    exact::Vector{Char}
    misses::Vector{UInt32}
    lowerbound::SIMD.Vec{32,UInt8}
    upperbound::SIMD.Vec{32,UInt8}
    scratch::SIMD.Vec{32,UInt8}

    # Inner constructor to ensure everything is sorted.
    function Schema{N}(exact, misses, lowerbound, upperbound, scratch) where {N}
        schema = new{N}(exact, misses, lowerbound, upperbound, scratch)
        return empty!(schema)
    end
end

struct VecPointer
    ptr::Ptr{UInt8}
end

Base.getindex(x::VecPointer, c::Char) = Base.unsafe_load(x.ptr, c - 'a' + 1)
Base.setindex!(x::VecPointer, v, c::Char) = Base.unsafe_store!(x.ptr, v, c - 'a' + 1)

@inline function VecPointer(schema::Schema{N}, sym::Symbol) where {N}
    base = Ptr{UInt8}(Base.pointer_from_objref(schema))
    if sym == :lowerbound
        offset = Base.fieldoffset(Schema{N}, 3)
    elseif sym == :upperbound
        offset = Base.fieldoffset(Schema{N}, 4)
    elseif sym == :scratch
        offset = Base.fieldoffset(Schema{N}, 5)
    else
        msg = "Unknown field name $sym"
        throw(ArgumentError(msg))
    end
    return VecPointer(base + offset)
end

function Schema{N}() where {N}
    exact = Vector{Char}(undef, N)
    misses = Vector{UInt32}(undef, N)
    lowerbound = SIMD.Vec{32,UInt8}(0x00)
    upperbound = SIMD.Vec{32,UInt8}(0xff)
    scratch = SIMD.Vec{32,UInt8}(0x00)
    return Schema{N}(exact, misses, lowerbound, upperbound, scratch)
end

clearscratch!(schema::Schema) = schema.scratch = SIMD.Vec{32,UInt8}(0x00)
function Base.empty!(schema::Schema)
    clear!(schema.exact)
    clear!(schema.misses)
    schema.lowerbound = SIMD.Vec{32,UInt8}(0x00)
    schema.upperbound = SIMD.Vec{32,UInt8}(0xff)
    clearscratch!(schema)
    return schema
end

# Filtering
compare(f::F, x::SIMD.Vec{32,UInt8}, y::SIMD.Vec{32,UInt8}) where {F} = sum(f(x,y)) == 32
function (f::Schema)(s::Union{AbstractString,Tuple})
    (; exact, misses, lowerbound, upperbound) = f
    clearscratch!(f)
    scratch_pointer = VecPointer(f, :scratch)

    # Bit mask for characters processed for bounds checking.
    for (index, char) in enumerate(s)
        # If this isn't match a known hit - return false.
        # If we know this character does not belong at this index, also return false
        if !unsafe_match(exact, (char, index)) || unsafe_match(misses, (char, index))
            return false
        end

        # Increment the character count
        scratch_pointer[char] += 1
    end
    # Perform bounds checks
    (; scratch) = f
    if !compare(>=, scratch, lowerbound) || !compare(<=, scratch, upperbound)
        return false
    end
    return true
end

#__merge(f::F, a::UInt8, b::UInt8) where {F} = isvalid(a) ? (isvalid(b) ? f(a, b) : a) : b
function merge!(a::Schema{N}, b::Schema{N}) where {N}
    # Merge exact matches and misses
    for i in Base.OneTo(N)
        b_exact = b.exact[i]
        if b_exact != ' '
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

function result_schema!(schema::Schema{N}, guess, states::NTuple{N,States}) where {N}
    empty!(schema)
    lowerbound = VecPointer(schema, :lowerbound)
    upperbound = VecPointer(schema, :upperbound)

    # Process correct guesses first, then process incorrect guesses.
    # This makes it easier to deal with incorrect guesses that repeat a letter that
    # was correct.
    @inbounds for i in Base.OneTo(N)
        state = states[i]
        char = guess[i]
        if state == Green
            schema.exact[i] = char
            lowerbound[char] += one(UInt8)
        elseif state == Yellow
            schema.misses[i] = set(schema.misses[i], char)
            lowerbound[char] += one(UInt8)
        end
    end

    @inbounds for i in Base.OneTo(N)
        state = states[i]
        if state == Gray
            char = guess[i]
            # Add this to the "misses" list and clamp the upperbound to the exact number
            # of occurances.
            schema.misses[i] = set(schema.misses[i], char)
            upperbound[char] = lowerbound[char]
        end
    end
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
    while iszero(val & (1 << i))
        i += 1
        (i > 2) && return nothing
    end
    state = (i == 0) ? Gray : (i == 1 ? Yellow : Green)
    return state, i + 1
end

function generate(schema::Schema, guess::NTuple{N,Char}) where {N}
    return ntuple(Val(N)) do i
        # If this is an exact match, then the only possibility is green.
        char = @inbounds(guess[i])
        if unsafe_strict_match(schema.exact, (char, i))
            return PossibleStates(Green)
        end

        # Check if this character is a known non-existant character.
        # This means that there is a max of 0 entries for this character.
        # If so, it can only be Gray.
        upperbound = VecPointer(schema, :upperbound)
        if iszero(upperbound[char])
            return PossibleStates(Gray)
        end

        # Now check if this is forced to be either Gray or Yellow
        # Basically, if we know there is another exact match for this position, than this
        # index cannot be green.
        #
        # This can also be forced to be either Gray or Yellow if we know this character
        # does not belong in this possition.
        if schema.exact[i] != ' ' || unsafe_match(schema.misses, (char, i))
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
function ispossible(
    schema::Schema,
    guess::NTuple{N,Char},
    result::NTuple{N,States},
) where {N}
    # Bit mask of characters that have been process to avoid repeating work.
    processed = zero(UInt64)
    lowerbound = VecPointer(schema, :lowerbound)
    upperbound = VecPointer(schema, :upperbound)
    @inbounds for i in Base.OneTo(N)
        char = guess[i]
        isset(processed, char) && continue

        # Count how many times this character is a match.
        this_count = 0
        for j = i:N
            this_char = guess[j]
            this_state = result[j]
            if this_char == char && in(this_state, (Green, Yellow))
                this_count += 1
            end
        end

        # Make sure the number of occurances lies within our known bounds.
        this_count < lowerbound[char] && return false
        this_count > upperbound[char] && return false

        # # Ensure upper bound
        # ub = unsafe_getcount(schema.lacks, char)
        # if isvalid(ub) && this_count > ub
        #     return false
        # end

        processed = set(processed, char)
    end
    return true
end

#####
##### Count possibilies
#####

@generated function countmatches(
    f::F,
    schema::Schema{N},
    guess::NTuple{N,Char},
    dictionary::Union{AbstractVector,DataStructures.OrderedSet};
    tempschema::Schema = Schema{N}(),
    wordcallback::G = (_...) -> nothing,
) where {F,N,G}
    gather = [Symbol("state_$i") for i in Base.OneTo(N)]
    return quote
        possibilities = generate(schema, guess)
        Base.Cartesian.@nloops $N state (j -> possibilities[j]) begin
            states = ($(gather...),)
            ispossible(schema, guess, states) || continue

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
end

cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T<:Integer} = one(T) + div(a - one(T), b)

function process_dictionary(
    schema::Schema{N},
    dictionary;
    target = DataStructures.OrderedSet(dictionary),
    batchsize = 16,
) where {N}
    scores = Vector{Int64}(undef, length(dictionary))

    threads = Base.OneTo(Threads.nthreads())
    realschema_tls = [deepcopy(schema) for _ in threads]
    tempschema_tls = [Schema{N}() for _ in threads]
    seen_tls = [falses(length(target)) for _ in threads]

    meter = ProgressMeter.Progress(length(dictionary), 1)
    best_bound = Threads.Atomic{Int}(typemax(Int))
    workcount = Threads.Atomic{Int}(1)
    numbatches = cdiv(length(dictionary), batchsize)

    #@time for tid in Base.OneTo(Threads.nthreads())
    Threads.@threads for tid in Base.OneTo(Threads.nthreads())
        maxpartition = Ref(0)
        aborted = Ref(false)
        while true
            # Get this threads work load
            k = Threads.atomic_add!(workcount, 1)
            k > numbatches && break

            start = (k - 1) * batchsize + 1
            stop = min(k * batchsize, length(dictionary))

            realschema = realschema_tls[tid]
            tempschema = tempschema_tls[tid]
            seen = seen_tls[tid]

            # Process this batch
            for i in start:stop
                word = dictionary[i]
                seen .= false
                maxpartition[] = 0
                aborted[] = false

                wordcallback(i) = (seen[i] = true)
                currentbest = best_bound[]
                countmatches(
                    realschema,
                    word,
                    target;
                    tempschema,
                    wordcallback,
                ) do partitionsize
                    if partitionsize >= currentbest
                        aborted[] = true
                        return false
                    end
                    maxpartition[] = max(maxpartition[], partitionsize)
                    return true
                end

                # Handle any words that aren't covered by entering this guess
                missed = length(target) - count(seen)
                maxpartition_unbox = max(maxpartition[], missed)

                if maxpartition_unbox < currentbest && !aborted[]
                    Threads.atomic_min!(best_bound, maxpartition_unbox)
                end

                scores[i] = aborted[] ? typemax(eltype(scores)) : maxpartition_unbox
            end
            ProgressMeter.next!(meter; step = stop - start + 1)
        end
    end
    ProgressMeter.finish!(meter)
    return scores
end

#####
##### Merging two Schema
#####

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
