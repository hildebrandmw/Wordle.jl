#####
##### PushVector
#####

# Ever so slightly faster for pushing a bunch of items to because it doesn't
# require the unconditional `ccall` required by Julia's standard vector.
mutable struct PushVector{T} <: AbstractVector{T}
    data::Vector{T}
    current_length::Int
    max_length::Int
end

PushVector{T}() where {T} = PushVector{T}(T[], 0, 0)
Base.IndexStyle(::Type{<:PushVector}) = Base.IndexLinear()
Base.size(A::PushVector) = (A.current_length,)

function Base.getindex(A::PushVector, i::Int)
    @boundscheck checkbounds(A, i)
    return @inbounds(A.data[i])
end

function Base.setindex!(A::PushVector, v, i::Int)
    @boundscheck checkbounds(A, i)
    return @inbounds(A.data[i] = v)
end
Base.empty!(A::PushVector) = (A.current_length = 0)

function Base.push!(A::PushVector, v)
    (; current_length, max_length) = A
    current_length += 1
    if (current_length <= max_length)
        @inbounds A[current_length] = v
    else
        push!(A.data, v)
        A.max_length = current_length
    end
    A.current_length = current_length
    return A
end

#####
##### CompressedVector
#####

# Compress a vector of vectors.
struct CompressedVectors{T} <:
        AbstractVector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}
    data::Vector{T}
    offsets::Vector{Int}
end

CompressedVectors{T}() where {T} = CompressedVectors{T}(T[], [1])
function _check_eltype(x::AbstractArray{T}) where {T}
    @assert isa(getindex(x, firstindex(x)), T)
end

Base.IndexStyle(::Type{<:CompressedVectors}) = Base.IndexLinear()
Base.size(A::CompressedVectors) = ((length(A.offsets) - 1),)

function Base.getindex(A::CompressedVectors, i::Int)
    @boundscheck checkbounds(A, i)
    (; data, offsets) = A
    start = @inbounds(offsets[i])
    stop = @inbounds(offsets[i+1]) - 1
    return view(data, start:stop)
end

function Base.append!(A::CompressedVectors, v::AbstractVector)
    (; data, offsets) = A
    append!(data, v)
    push!(offsets, length(data) + 1)
    return A
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
Base.convert(::Type{UInt8}, char::ASCIIChar) = char.val

ASCIIChar(char::Char) = ASCIIChar(convert(UInt8, char))

const AA = ASCIIChar('a')
normalize(char::ASCIIChar) = (char - ASCIIChar('a'))
Base.:-(a::ASCIIChar, b::ASCIIChar) = (a.val - b.val)

Base.zero(::Type{ASCIIChar}) = ASCIIChar(zero(UInt8))
Base.zero(::ASCIIChar) = zero(ASCIIChar)

Base.iszero(char::ASCIIChar) = iszero(char.val)
Base.:(==)(a::ASCIIChar, b::ASCIIChar) = (a.val == b.val)
Base.isless(a::ASCIIChar, b::ASCIIChar) = isless(a.val, b.val)
Base.getindex(simd::SIMD.Vec{N,UInt8}, char::ASCIIChar) where {N} =
    simd[normalize(char) + 1]

# Masks
struct Mask{T}
    val::T
end
Mask{T}() where {T} = Mask{T}(zero(T))

Mask(char::ASCIIChar) = Mask{UInt32}(Base.shl_int(one(UInt32), normalize(char)))
unwrap(mask::Mask) = mask.val

Base.:|(a::Mask{T}, b::Mask{T}) where {T} = Mask{T}(a.val | b.val)
Base.:|(mask::Mask{UInt32}, char::ASCIIChar) = mask | Mask(char)
ismatch(a::Mask{T}, b::Mask{T}) where {T} = !iszero(a.val & b.val)
ismatch(a::Mask{UInt32}, char::ASCIIChar) = ismatch(a, Mask(char))
flip(a::Mask) = Mask(~unwrap(a))

Base.zero(::Type{Mask{T}}) where {T} = Mask{T}()
Base.zero(::T) where {T<:Mask} = zero(T)

# SIMD Bridges
bitifelse(mask::Mask, args...) = bitifelse(mask.val, args...)
bitunpack(mask::Mask) = bitunpack(mask.val)

# Generate counts
function charcounts(word::NTuple{N,ASCIIChar}) where {N}
    counts = SIMD.Vec{32,UInt8}(0x00)
    for i in Base.OneTo(N)
        char = word[i]
        counts += bitunpack(Mask(char))
    end
    return counts
end
