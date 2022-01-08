using Test, Wordle

@testset "Testing Schema" begin
    @testset "Testing Matching" begin
    end

    @testset "Testing Counting Merge" begin
        a = [('a', 1), ('b', 2), ('e', 5)]
        b = [('a', 1), ('b', 1), ('f', 5)]
        @test Wordle.counting_merge(a, b) == [('a', 1), ('b', 2), ('e', 5), ('f', 5)]
        @test Wordle.counting_merge(b, a) == [('a', 1), ('b', 2), ('e', 5), ('f', 5)]
    end
end

@testset "Testing Possibilities" begin
    @testset "Testing Iterator" begin
        iter = ((Wordle.Green,), (Wordle.Yellow,), (Wordle.Gray,), ())
        for (a, b, c) in Iterators.product(iter, iter, iter)
            possibilities = Wordle.PossibleStates(a..., b..., c...)
            reference = sort(unique([a..., b..., c...]))
            @test collect(possibilities) == reference
        end
    end
end
