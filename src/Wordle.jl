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
##### Discovered Knowledge
#####

# exact: Exact caracter matches (character, index)
# misses: Known non-locations (character, index)
# partial: Partial matches including exact matches (character, count)
# lacks: Letters not present in the
struct Schema
    exact::Vector{Tuple{Char,Int}}
    misses::Vector{Tuple{Char,Int}}
    partial::Vector{Tuple{Char,Int}}
    lacks::Vector{Tuple{Char,Int}}

    # Inner constructor to ensure everything is sorted.
    function Schema(exact, misses, partial, lacks)
        schema = new(exact, misses, partial, lacks)
        return sort!(schema)
    end
end

const TupleCI = Tuple{Char,<:Integer}

_get(::Type{Char}, t::TupleCI) = t[1]
_get(::Type{<:Integer}, t::TupleCI) = t[2]
_get(::Type{<:TupleCI}, t::TupleCI) = t
_get(::T, t::TupleCI) where {T} = _get(T, t)

function Schema(;
    exact = Tuple{Char,UInt8}[],
    misses = Tuple{Char,UInt8}[],
    partial = Tuple{Char,UInt8}[],
    lacks = Tuple{Char,UInt8}[],
)
    return Schema(exact, misses, partial, lacks)
end

function match(v::Vector{<:Tuple{Char,<:Integer}}, x::T) where {T}
    for i in eachindex(v)
        _get(T, v[i]) == x && return i
    end
    return nothing
end

function Base.empty!(schema::Schema)
    empty!(schema.exact)
    empty!(schema.misses)
    empty!(schema.partial)
    empty!(schema.lacks)
    return schema
end

function Base.sort!(schema::Schema)
    sort!(schema.exact; by = last, alg = Base.InsertionSort)
    sort!(schema.misses; alg = Base.InsertionSort)
    sort!(schema.partial; alg = Base.InsertionSort)
    sort!(schema.lacks; alg = Base.InsertionSort)
    return schema
end

function (f::Schema)(s::Union{AbstractString,Tuple})
    (; exact, misses, partial, lacks) = f
    # First, check exact matches and exact misses
    for (char, index) in exact
        s[index] == char || return false
    end
    for (char, index) in misses
        s[index] == char && return false
    end

    # Next, check partial matches
    for (char, count) in partial
        Base.count(isequal(char), s) >= count || return false
    end

    # Finally, check the negative matches
    for (char, count) in lacks
        Base.count(isequal(char), s) > count && return false
    end
    return true
end

@enum States::UInt8 Green = 1 Yellow = 2 Gray = 4
function pushincrement!(v::AbstractVector{Tuple{Char,I}}, char) where {I<:Integer}
    j = match(v, char)
    if j === nothing
        push!(v, (char, one(I)))
    else
        current = v[j]
        v[j] = (char, current[2] + one(I))
    end
    return v
end

function result_schema(guess, states::NTuple{N,States}) where {N}
    return result_schema!(Schema(), guess, states)
end

function result_schema!(schema::Schema, guess, states::NTuple{N,States}) where {N}
    empty!(schema)

    # Process correct guesses first, then process incorrect guesses.
    # This makes it easier to deal with incorrect guesses that repeat a letter that
    # was correct.
    for i in eachindex(states)
        state = states[i]
        char = guess[i]
        if state == Green
            push!(schema.exact, (char, i))
            pushincrement!(schema.partial, char)
        elseif state == Yellow
            push!(schema.misses, (char, i))
            pushincrement!(schema.partial, char)
        end
    end

    for i in eachindex(states)
        state = states[i]
        if state == Gray
            char = guess[i]
            # Add this to the "misses" list.
            match(schema.misses, (char, i)) === nothing && push!(schema.misses, (char, i))

            # Keep track of upper bounds for character counts.
            j = match(schema.partial, char)
            maxcount = (j === nothing) ? 0 : _get(Int, schema.partial[j])
            match(schema.lacks, char) === nothing && push!(schema.lacks, (char, maxcount))
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
    state = (i == 0) ? Green : (i == 1 ? Yellow : Gray)
    return state, i + 1
end

function generate(schema::Schema, guess::NTuple{N,Char}) where {N}
    return ntuple(Val(N)) do i
        # If this is an exact match, then the only possibility is green.
        char = @inbounds(guess[i])
        if match(schema.exact, (char, i)) !== nothing
            return PossibleStates(Green)
        end

        # Check if this character is a known non-existant character.
        # This means that there is a max of 0 entries for this character.
        # If so, it can only be Gray.
        j = match(schema.lacks, char)
        if j !== nothing
            _, maxcount = @inbounds(schema.lacks[j])
            iszero(maxcount) && return PossibleStates(Gray)
        end

        # Now check if this is forced to be either Gray or Yellow
        # Basically, if we know there is another exact match for this position, than this
        # index cannot be green.
        if match(schema.exact, i) !== nothing
            return PossibleStates(Yellow, Gray)
        end

        # This can also be forced to be either Gray or Yellow if we know this character
        # does not belong in this possition.
        if match(schema.misses, (char, i)) !== nothing
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
    for i in Base.OneTo(N)
        char = guess[i]
        iszero(processed & (one(UInt64) << (Int(char - 'a')))) || continue

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
        # Ensure lower bound.
        k = match(schema.partial, char)
        if k !== nothing
            this_count < _get(Int, schema.partial[k]) && return false
        end

        # Ensure upper Bound.
        k = match(schema.lacks, char)
        if k !== nothing
            this_count > _get(Int, schema.lacks[k]) && return false
        end
        processed |= (one(UInt64) << (Int(char - 'a')))
    end
    return true
end

#####
##### Result Iterator
#####

struct ResultIterator{N,I}
    schema::Schema
    guess::NTuple{N,Char}
    iter::I

end

function ResultIterator(schema::Schema, guess::NTuple{N,Char}) where {N}
    iter = Iterators.product(generate(schema, guess)...)
    return ResultIterator(schema, guess, iter)
end

Base.IteratorSize(::Type{<:ResultIterator}) = Base.SizeUnknown()
function Base.iterate(r::ResultIterator{T}, s...) where {T}
    (; schema, guess, iter) = r
    y = iterate(iter, s...)
    y === nothing && return nothing
    (v, t) = y
    while !ispossible(schema, guess, v)
        y = iterate(iter, t)
        y === nothing && return nothing
        (v, t) = y
    end
    return (v, t)
end

#####
##### Count possibilies
#####

@generated function countmatches(
    f::F,
    schema::Schema,
    guess::NTuple{N,Char},
    dictionary::Union{AbstractVector,DataStructures.OrderedSet};
    tempschema::Schema = Schema(),
    wordcallback::G = (_...) -> nothing,
) where {F,N,G}
    gather = [Symbol("state_$i") for i in Base.OneTo(N)]
    return quote
        possibilities = generate(schema, guess)
        Base.Cartesian.@nloops $N state (j -> possibilities[j]) begin
            states = ($(gather...),)
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
    schema::Schema,
    dictionary;
    target = DataStructures.OrderedSet(dictionary),
    batchsize = 16,
)
    scores = Vector{Int64}(undef, length(dictionary))

    threads = Base.OneTo(Threads.nthreads())
    tempschema_tls = [Schema() for _ in threads]
    seen_tls = [falses(length(target)) for _ in threads]

    meter = ProgressMeter.Progress(length(dictionary), 1)
    best_bound = Threads.Atomic{Int}(typemax(Int))
    workcount = Threads.Atomic{Int}(1)
    numbatches = cdiv(length(dictionary), batchsize)

    Threads.@threads for tid in Base.OneTo(Threads.nthreads())
        while true
            # Get this threads work load
            k = Threads.atomic_add!(workcount, 1)
            k > numbatches && break

            start = (k - 1) * batchsize + 1
            stop = min(k * batchsize, length(dictionary))

            tempschema = tempschema_tls[tid]
            seen = seen_tls[tid]

            # Process this batch
            for i = start:stop
                word = dictionary[i]
                seen .= false

                wordcallback(i) = (seen[i] = true)
                maxpartition = 0
                currentbest = best_bound[]
                aborted = Ref(false)
                countmatches(
                    schema,
                    word,
                    target;
                    tempschema,
                    wordcallback,
                ) do partitionsize
                    if partitionsize >= currentbest
                        aborted[] = true
                        return false
                    end
                    maxpartition = max(maxpartition, partitionsize)
                    return true
                end

                # Handle any words that aren't covered by entering this guess
                missed = length(target) - count(seen)
                maxpartition = max(maxpartition, missed)

                if maxpartition < currentbest && !aborted[]
                    Threads.atomic_min!(best_bound, maxpartition)
                end

                scores[i] = aborted[] ? typemax(eltype(scores)) : maxpartition
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

counting_merge(a, b) = counting_merge!(copy(a), b)
function counting_merge!(a, b)
    for i in eachindex(b)
        c = b[i]
        j = match(a, c[1])
        if j === nothing
            push!(a, c)
        elseif c[2] > a[j][2]
            a[j] = c
        end
    end
    sort!(a; alg = Base.InsertionSort)
    return a
end

function merge!(a::Schema, b::Schema)
    # Add any new exact matches
    for i in eachindex(b.exact)
        c = b.exact[i]
        in(c, a.exact) || push!(a.exact, c)
    end
    sort!(a.exact; by = last, alg = Base.InsertionSort)

    # Merge any partial matches as well.
    # Need to treat this slightly differently because increase in partial match counts
    # shouldn't get pushed.
    counting_merge!(a.partial, b.partial)
    counting_merge!(a.lacks, b.lacks)
    return a
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
