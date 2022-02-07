function bitunpack(x::UInt32)
    str = raw"""
    define <32 x i8> @entry(<32 x i8>,i32) #0 {
    top:
        %2 = bitcast i32 %1 to <32 x i1>
        %3 = select <32 x i1> %2, <32 x i8> %0, <32 x i8> zeroinitializer
        ret <32 x i8> %3
    }

    attributes #0 = { alwaysinline }
    """
    i = SIMD.Vec{32,UInt8}(0x01)
    z = Base.llvmcall(
        (str, "entry"),
        SIMD.LVec{32,UInt8},
        Tuple{SIMD.LVec{32,UInt8},UInt32},
        i.data,
        x,
    )
    return SIMD.Vec(z)
end

function bitifelse(m::UInt32, x::SIMD.Vec{32,UInt8}, y::SIMD.Vec{32,UInt8})
    str = raw"""
    define <32 x i8> @entry(i32,<32 x i8>,<32 x i8>) #0 {
    top:
        %3 = bitcast i32 %0 to <32 x i1>
        %4 = select <32 x i1> %3, <32 x i8> %1, <32 x i8> %2
        ret <32 x i8> %4
    }

    attributes #0 = { alwaysinline }
    """
    z = Base.llvmcall(
        (str, "entry"),
        SIMD.LVec{32,UInt8},
        Tuple{UInt32,SIMD.LVec{32,UInt8},SIMD.LVec{32,UInt8}},
        m,
        x.data,
        y.data,
    )
    return SIMD.Vec(z)
end
