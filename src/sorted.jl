function sorted_intersect_size(
    a::AbstractVector,
    b::AbstractVector,
)
    count = 0
    (isempty(a) | isempty(b)) && return count
    ai = firstindex(a)
    bi = firstindex(b)

    # Last indices
    la = lastindex(a)
    lb = lastindex(b)
    while true
        va = @inbounds(a[ai])
        vb = @inbounds(b[bi])

        # Bools for equality and less than
        eq = va == vb
        lt = va < vb

        count += Int(eq)
        ai += (eq | lt)
        bi += (eq | ~lt)

        # If either index is now out of bounds, than we're done
        ((ai > la) | (bi > lb)) && break
    end
    return count
end

function sorted_intersect!(
    dst::AbstractVector,
    a::AbstractVector,
    b::AbstractVector,
)
    empty!(dst)
    (isempty(a) | isempty(b)) && return dst
    ai = firstindex(a)
    bi = firstindex(b)

    # Last indices
    la = lastindex(a)
    lb = lastindex(b)
    while true
        va = @inbounds(a[ai])
        vb = @inbounds(b[bi])

        # Bools for equality and less than
        eq = va == vb
        lt = va < vb

        ai += (eq | lt)
        bi += (eq | ~lt)
        eq && push!(dst, va)

        # If either index is now out of bounds, than we're done
        ((ai > la) | (bi > lb)) && break
    end
    return dst
end
