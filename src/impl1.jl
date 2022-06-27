#####
##### Pre-encode Wordes
#####

@inline choose_simd_width(N) = N > 8 ? 16 : 8
struct Fingerprint{N,M}
    masks::SIMD.Vec{M,UInt32}
    counts::SIMD.Vec{32,UInt8}
end

function Fingerprint(word::NTuple{N,ASCIIChar}) where {N}
    M = choose_simd_width(N)
    masks_tuple = ntuple(Val(M)) do i
        i > N && return zero(UInt32)
        return unwrap(Mask(word[i]))
    end

    return Fingerprint{N,M}(SIMD.Vec{M,UInt32}(masks_tuple), charcounts(word))
end

function getword(f::Fingerprint{N}) where {N}
    # Reverse the finterprint mask
    word = ntuple(Val(N)) do i
        mask = f.masks[i]
        j = Base.trailing_zeros(mask)
        return ASCIIChar(UInt('a') + j)
    end
    return join(word)
end

#####
##### Discovered Knowledge
#####

abstract type AbstractSchema{N} end
struct EmptySchema{N} <: AbstractSchema{N} end
struct Schema{N,M} <: AbstractSchema{N}
    misses::SIMD.Vec{M,UInt32}
    lowerbound::SIMD.Vec{32,UInt8}
    upperbound::SIMD.Vec{32,UInt8}
end

Schema{N,M}() where {N,M} = Schema{N,M}(schema_template(Val(N))...)
function Schema{N}(misses::SIMD.Vec{M,UInt32}, lowerbound, upperbound) where {N,M}
    return Schema{N,M}(misses, lowerbound, upperbound)
end

function schema_template(::Val{N}) where {N}
    M = choose_simd_width(N)
    misses = SIMD.Vec{M,UInt32}(0x00)
    lowerbound = SIMD.Vec{32,UInt8}(0x00)
    upperbound = SIMD.Vec{32,UInt8}(0xff)
    return (; misses, lowerbound, upperbound)
end

# Filtering
compare(f::F, x::SIMD.Vec{N,T}, y::SIMD.Vec{N,T}) where {F,N,T} = sum(f(x, y)) == N
allof(x::SIMD.Vec{N,Bool}) where {N} = iszero(sum(!x))
anyof(x::SIMD.Vec{N,Bool}) where {N} = !iszero(sum(x))

function (f::Schema{N,M})(x::Fingerprint{N,M}) where {N,M}
    (; misses, lowerbound, upperbound) = f
    (; masks, counts) = x

    green_match = allof(iszero(misses & masks))
    lowerbound_match = compare(>=, counts, lowerbound)
    upperbound_match = compare(<=, counts, upperbound)
    return green_match & lowerbound_match & upperbound_match
end

merge(::EmptySchema{N}, b::Schema{N}) where {N} = b
function merge(a::Schema{N}, b::Schema{N}) where {N}
    return Schema{N}(
        a.misses | b.misses,
        max(a.lowerbound, b.lowerbound),
        min(a.upperbound, b.upperbound),
    )
end

iscompatible(::EmptySchema{N}, b::Schema{N}) where {N} = true
function iscompatible(a::T, b::T) where {T<:Schema}
    # "b" is not compatible with "a" if
    # (1) "b" has an exact match (inverted mask) that contradicts a known non-match in "a"
    # (2) "b" has a non-match that contradicts a known match (inverted mask) in "a"
    #
    # We can check this by "or"-ing together the two mask vectors and checking if any
    # result in all-ones masks.
    #
    # If there is a mismatch, than "green_mismatch" will be true.
    green_mismatch = anyof((a.misses | b.misses) == typemax(UInt32))

    # Next, we have to check whether "b" has incompatible bounds compared to "a". This is
    # the case if
    # (1) "b.upperbound" is less than "a.lowerbound"
    # (2) "b.lowerbound" is greater than "b.upperbond"
    lowerbound_mismatch = anyof(b.upperbound < a.lowerbound)
    upperbound_mismatch = anyof(b.lowerbound > a.upperbound)
    return !(green_mismatch | lowerbound_mismatch | upperbound_mismatch)
end

const Gray = UInt8(1)
const Yellow = UInt8(2)
const Green = UInt8(3)

isgray(x::Union{UInt8,SIMD.Vec{<:Any,UInt8}}) = (x == Gray)
isyellow(x::Union{UInt8,SIMD.Vec{<:Any,UInt8}}) = (x == Yellow)
isgreen(x::Union{UInt8,SIMD.Vec{<:Any,UInt8}}) = (x == Green)

function tovec(states::NTuple{N,UInt8}) where {N}
    M = choose_simd_width(N)
    states_tuple = ntuple(Val(M)) do i
        i > N && return zero(UInt8)
        return UInt8(states[i])
    end
    return SIMD.Vec{M,UInt8}(states_tuple)
end

convert_guess(x::Fingerprint) = x
convert_guess(x::NTuple{N,ASCIIChar}) where {N} = convert_guess(Fingerprint(x))

convert_states(x::SIMD.Vec{N,UInt8}) where {N} = x
convert_states(x::NTuple{N,UInt8}) where {N} = convert_states(tovec(x))

function result_schema(guess, states)
    return _result_schema(convert_guess(guess), convert_states(states))
end

function _result_schema(guess::Fingerprint{N,M}, states::SIMD.Vec{M,UInt8}) where {N,M}
    (; masks) = guess
    misses = SIMD.vifelse(isgreen(states), ~masks, masks)
    lowerbound = SIMD.Vec{32,UInt8}(0x00)
    graymask = zero(UInt32)
    @inbounds for i in Base.OneTo(N)
        mask = masks[i]
        if isgray(states[i])
            graymask |= mask
        else
            lowerbound = bitifelse(mask, lowerbound + one(lowerbound), lowerbound)
        end
    end
    upperbound = bitifelse(graymask, lowerbound, SIMD.Vec{32,UInt8}(0xff))
    return Schema{N}(misses, lowerbound, upperbound)
end

function state_sort(a::NTuple{N,UInt8}, b::NTuple{N,UInt8}) where {N}
    # Prioritize number of zeros, than number of ones.
    zeros_a = count(iszero, a)
    zeros_b = count(iszero, b)
    ones_a = count(isone, a)
    ones_b = count(isone, b)

    if zeros_a > zeros_b
        return true
    elseif (zeros_a + ones_a) > (zeros_b + ones_b)
        return true
    end
    return false
end

function generate_schemas(dictionary::AbstractVector{NTuple{N,ASCIIChar}}) where {N}
    # Parameters
    M = choose_simd_width(N)

    # Setup iteration space and Schema results
    possible_states = (Gray, Yellow, Green)
    iterator = ntuple(Returns(possible_states), Val(N))
    allstates = vec(collect(Iterators.product(iterator...)))
    sort!(allstates; lt = state_sort)

    schemas = Matrix{Schema{N,M}}(undef, length(possible_states)^N, length(dictionary))
    for (j, word) in enumerate(dictionary)
        for (i, states) in enumerate(allstates)
            schemas[i, j] = result_schema(word, states)
        end
    end
    println("Done!")
    return schemas
end

#####
##### Count possibilies
#####

struct CountMatches{N} end

function reverse_state(::Val{N}, i) where {N}
    possible_states = (Gray, Yellow, Green)
    iterator = ntuple(Returns(possible_states), Val(N))
    count = 0
    for states in Iterators.product(iterator...)
        count += 1
        count == i && return states
    end
    return nothing
end

# Bottom of recursion
const Dictionary{T} = Union{AbstractVector{T},DataStructures.OrderedSet{T}}
function (::CountMatches{0})(
    f::F,
    schema::AbstractSchema{N},
    guess,
    schemas::AbstractMatrix,
    target::Dictionary{Fingerprint{N,M}},
    ::Tuple{} = ();
    current_best = length(target)
) where {F,N,M}
    maxsize = zero(Int64)
    for nextschema in view(schemas, :, guess)
        iscompatible(schema, nextschema) || continue
        mergedschema = merge(schema, nextschema)
        partitionsize = 0
        for _ in Iterators.filter(mergedschema, target)
            partitionsize += 1
            partitionsize >= current_best && return (current_best, 1)
        end

        # If adding this guess results in no remaining words, it's invalid
        # and we should just continue
        iszero(partitionsize) && continue
        maxsize = max(maxsize, partitionsize)

        # Check for early exit
        f(maxsize) && break
    end
    return maxsize, 1
end

function (::CountMatches{K})(
    f::F,
    schema::AbstractSchema{N},
    guess,
    schemas::AbstractMatrix,
    target::Dictionary{Fingerprint{N,M}},
    buffers::NTuple{K,AbstractVector{Fingerprint{N,M}}} = ntuple(_->PushVector{Fingerprint{N,M}}, Val(K));
    current_best = length(target)
) where {K,F,N,M}
    maxsize = zero(Int64)
    maxheight = zero(Int64)
    targetbuffer = buffers[1]
    subbuffers = Base.tail(buffers)

    for nextschema in view(schemas, :, guess)
        iscompatible(schema, nextschema) || continue

        mergedschema = merge(schema, nextschema)
        empty!(targetbuffer)
        for fingerprint in Iterators.filter(mergedschema, target)
            push!(targetbuffer, fingerprint)
        end
        this_partition = length(targetbuffer)

        iszero(this_partition) && continue
        if isone(this_partition)
            min_subpartition = 1
            this_treeheight = 1
        else
            min_subpartition = this_partition
            this_treeheight = typemax(Int64)
            for next_guess in Base.OneTo(size(schemas, 2))
                partitionsize, height = (CountMatches{K - 1}())(
                    f,
                    mergedschema,
                    next_guess,
                    schemas,
                    targetbuffer,
                    subbuffers;
                    current_best = min_subpartition,
                )
                iszero(partitionsize) && continue

                if partitionsize < min_subpartition
                    min_subpartition = partitionsize
                    # Increment the height to indicate that this partition came
                    # from deeper in the search tree.
                    this_treeheight = height + 1
                end
            end
            min_subpartition == this_partition && return (this_partition, maxheight)
        end

        maxsize = max(maxsize, min_subpartition)
        maxheight = max(maxheight, this_treeheight)
        (min_subpartition >= current_best) || f(maxsize) && break
    end
    return maxsize, maxheight
end

#####
##### Top level processing
#####

cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T<:Integer} = one(T) + div(a - one(T), b)

_kickstart_bounds(::Val{N}, dictionary) where {N} = length(dictionary)
_kickstart_bounds(::Val{4}, _) = 101
#_kickstart_bounds(::Val{5}, _) = 46
#_kickstart_bounds(::Val{6}, _) = 29

struct Abort{T}
    best::Threads.Atomic{T}
end

(f::Abort)(x) = (x > f.best[])

function process_dictionary(
    schema::AbstractSchema{N},
    dictionary::AbstractVector{Fingerprint{N,M}},
    schemas::AbstractMatrix{Schema{N,M}};
    batchsize = 16,
    target = dictionary,
) where {N,M}
    scores = Vector{Tuple{Int64,Int64}}(undef, length(dictionary))

    meter = ProgressMeter.Progress(length(dictionary), 1)
    best_bound = Threads.Atomic{Int}(_kickstart_bounds(Val(N), target))
    abort = Abort(best_bound)
    workcount = Threads.Atomic{Int}(1)
    numbatches = cdiv(length(dictionary), batchsize)

    buffers = map(Base.OneTo(Threads.nthreads())) do _
        return (
            PushVector{Fingerprint{N,M}}(),
            #PushVector{Fingerprint{N,M}}(),
        )
    end

    @time Threads.@threads for tid in Base.OneTo(Threads.nthreads())
    #@time for tid in 1:1
        maxpartition = Ref(0)
        while true
            # Get this threads work load
            k = Threads.atomic_add!(workcount, 1)
            k > numbatches && break

            start = (k - 1) * batchsize + 1
            stop = min(k * batchsize, length(dictionary))

            # Process this batch
            for i = start:stop
                maxpartition[] = 0
                maxsize, maxheight = (CountMatches{1}())(
                    abort,
                    schema,
                    i,
                    schemas,
                    target,
                    buffers[tid],
                )

                # Handle edge-case zeros by clamping to the worst case.
                maxsize = iszero(maxsize) ? length(dictionary) : maxsize
                if maxsize <= best_bound[]
                    Threads.atomic_min!(best_bound, maxsize)
                end

                scores[i] = (maxsize, maxheight)
            end

            ProgressMeter.next!(meter; step = length(start:stop))
        end
    end
    ProgressMeter.finish!(meter)
    return scores
end
