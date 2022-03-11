function make_masks(
    schemas::Matrix{Schema{N,M}},
    fingerprints::AbstractVector{Fingerprint{N,M}},
) where {N,M}
    pv_threadlocal = [PushVector{UInt16}() for _ in Base.OneTo(Threads.nthreads())]
    A = [CompressedVectors{UInt16}() for _ in Base.OneTo(length(fingerprints))]
    Threads.@threads for col in Base.OneTo(size(schemas, 2))
        pv = pv_threadlocal[Threads.threadid()]
        cv = A[col]
        for row in Base.OneTo(size(schemas, 1))
            schema = @inbounds(schemas[row, col])
            empty!(pv)
            for (i, fingerprint) in enumerate(fingerprints)
                schema(fingerprint) && push!(pv, i)
            end
            append!(cv, pv)
        end
    end
    return A
end

const Runs{T} = Vector{CompressedVectors{T}}
function (::CountMatches{1})(
    guess::Integer,
    runs::Runs{T},
    ::AbstractRange,
) where {T <: Integer}
    maxsize = zero(Int64)
    for run in runs[guess]
        isempty(run) && continue

        # Intersect the current
        min_subpartition = length(run)
        for nextguess in Base.OneTo(length(runs))
            _maxsize = zero(Int64)
            for nextrun in runs[nextguess]
                isempty(nextrun) && continue
                _maxsize = max(_maxsize, sorted_intersect_size(run, nextrun))
                _maxsize >= min_subpartition && break
            end
            min_subpartition = min(min_subpartition, _maxsize)
        end
        maxsize = max(maxsize, min_subpartition)
    end
    return maxsize
end

