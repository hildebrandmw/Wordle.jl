using Test, Wordle
using Random

@testset "Testing Utilities" begin
    @testset "Testing `clear!`" begin
        x = rand(UInt32, 100)
        Wordle.clear!(x)
        @test all(iszero, x)

        x = rand(Char, 100)
        Wordle.clear!(x)
        @test all(isequal(' '), x)
    end

    @testset "Testing Bitmask Utilities" begin
        ntrials = 100
        for T in (UInt32, UInt64)
            set = Set{Char}()
            bitset = zero(T)
            for _ in Base.OneTo(ntrials)
                for char in 'a':'z'
                    @test Wordle.isset(bitset, char) == in(char, set)
                end
                thischar = rand('a':'z')
                push!(set, thischar)
                bitset = Wordle.set(bitset, thischar)
            end
        end
    end

    @testset "Testing Matching" begin
        ntrials = 10
        v = Vector{Char}(undef, 5)
        for _ in Base.OneTo(ntrials)
            insertion_order = shuffle(1:length(v))
            groundtruth = Dict{Int,Char}()
            Wordle.clear!(v)
            for i in insertion_order
                char = rand('a':'z')
                v[i] = char
                groundtruth[i] = char

                for j in 1:length(v), c in 'a':'z'
                    m = Wordle.unsafe_match(v, (c, j))
                    gt = !haskey(groundtruth, j) || groundtruth[j] == c
                    @test m == gt
                end
            end
        end
    end

    # @testset "Testing Counter Utilities" begin
    #     @test all(Wordle.isvalid, UInt8(0):(typemax(UInt8) - one(UInt8)))
    #     @test Wordle.isvalid(typemax(UInt8)) == false
    #     v = Wordle.clear!(Vector{UInt8}(undef, length('a':'z')))
    #     for c in 'a':'z'
    #         @test !Wordle.isvalid(Wordle.unsafe_getcount(v, c))
    #     end

    #     @test Wordle.unsafe_getcount(v, 'a') == 0xff
    #     Wordle.unsafe_inccount!(v, 'a')
    #     @test Wordle.unsafe_getcount(v, 'a') == 1
    #     Wordle.unsafe_inccount!(v, 'a')
    #     @test Wordle.unsafe_getcount(v, 'a') == 2

    #     Wordle.unsafe_setcount!(v, 0, 'b')
    #     @test Wordle.unsafe_getcount(v, 'b') == 0
    # end
end

#####
##### Schema
#####

@testset "Testing Schema" begin
    @testset "Constructor" begin
        for N in (4,5,6)
            schema = Wordle.Schema{N}()
            @test length(schema.exact) == N
            @test all(isequal(' '), schema.exact)

            @test length(schema.misses) == N
            @test all(iszero, schema.misses)

            @test all(isequal(0x00), Tuple(schema.lowerbound))
            @test all(isequal(0xff), Tuple(schema.upperbound))
            @test all(isequal(0x00), Tuple(schema.scratch))
        end
    end

    @testset "Filtering" begin
        # Empty schema should match all words
        ntrials = 100
        schema = Wordle.Schema{5}()
        for _ in Base.OneTo(ntrials)
            s = ntuple(_ -> rand('a':'z'), Val{5}())
            @test schema(s)
        end

        # Now, we get more exact.
        guess = "hello"
        schema = Wordle.Schema{5}()
        @test schema(guess) == true

        # Exact matching
        schema.exact[1] = 'h'
        @test schema(guess) == true
        schema.exact[1] = 'g'
        @test schema(guess) == false
        empty!(schema)

        schema.exact[4] = 'l'
        @test schema(guess) == true
        schema.exact[4] = 'u'
        @test schema(guess) == false
        empty!(schema)

        # Exact Misses
        schema.misses[1] = Wordle.set(schema.misses[1], 'c')
        @test schema(guess) == true
        schema.misses[1] = Wordle.set(schema.misses[1], 'h')
        @test schema(guess) == false
        empty!(schema)

        schema.misses[3] = Wordle.set(schema.misses[1], 'h')
        @test schema(guess) == true
        schema.misses[3] = Wordle.set(schema.misses[1], 'l')
        @test schema(guess) == false
        empty!(schema)

        # Lower bounds checks
        lowerbound = Wordle.VecPointer(schema, :lowerbound)
        lowerbound['e'] += 1
        @test schema(guess) == true
        lowerbound['e'] += 1
        @test schema(guess) == false
        empty!(schema)

        lowerbound['l'] += 1
        @test schema(guess) == true
        lowerbound['l'] += 1
        @test schema(guess) == true
        lowerbound['l'] += 1
        @test schema(guess) == false
        empty!(schema)

        # Upper bounds checks
        upperbound = Wordle.VecPointer(schema, :upperbound)
        upperbound['h'] = 1
        @test schema(guess) == true
        upperbound['h'] = 0
        @test schema(guess) == false
        empty!(schema)

        upperbound['l'] = 2
        @test schema(guess) == true
        upperbound['l'] = 1
        @test schema(guess) == false
        upperbound['l'] = 0
        @test schema(guess) == false
        empty!(schema)
    end
end

