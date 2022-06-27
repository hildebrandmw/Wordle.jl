function newgame(wordlength::Integer, file; hardmode = false, original_guess = nothing, target = nothing)
    # Read in the dictionary and filter the the appropriate length.
    d = Tuple.(map.(ASCIIChar, collect.((tolength(wordlength, readwords(file))))))
    @show typeof(d)
    @assert isa(d, Vector{NTuple{wordlength,ASCIIChar}})
    if original_guess !== nothing
        @assert length(original_guess) == wordlength
        original_guess = Fingerprint(Tuple(ASCIIChar.(collect(original_guess))))
    end

    if target !== nothing
        target = Tuple.(map.(ASCIIChar, collect.((tolength(wordlength, readwords(target))))))
        target = DataStructures.OrderedSet(target)
    else
        target = DataStructures.OrderedSet(d)
    end
    Random.shuffle!(d)
    return newgame(d; hardmode, original_guess, _target = target,)
end

function newgame(
    _dictionary::Vector{NTuple{N,ASCIIChar}};
    hardmode = false,
    original_guess = nothing,
    _target = DataStructures.OrderedSet(_dictionary),
) where {N}
    dictionary = Fingerprint.(_dictionary)
    target = Fingerprint.(_target)

    schema = EmptySchema{N}()
    schemas = @time generate_schemas(_dictionary)
    first_iteration = true
    while length(target) > 1
        @show length(target)
        if first_iteration && original_guess !== nothing
            guess = original_guess
        else
            # This is SOO gross ...
            _scores = process_dictionary(schema, dictionary, schemas; target)

            function lt((score1, word1), (score2, word2))
                if score1 < score2
                    return true
                elseif score1 == score2
                    if in(word1, target) && !in(word2, target)
                        return true
                    end
                end
                return false
            end
            temp = sort(Glue(_scores, dictionary); lt = lt)
            scores = first.(temp)
            dictionary_sorted = last.(temp)

            # Display guesses and scores
            strings = getword.(view(dictionary_sorted, 1:min(length(dictionary_sorted), 10)))
            first_scores = view(scores, 1:min(length(scores), 10))
            display(collect(zip(strings, first_scores)))
            guess = first(dictionary_sorted)
        end
        println("Current Guess: $(getword(guess))")

        # Read in how well this guess did.
        local status
        retry = false
        while true
            print("Status: ")
            status = readline(stdin)

            # Handle special cases
            if status == "show schema"
                println(stdout, repr(schema))
                continue
            elseif status == "show target"
                foreach(x -> println(getword(x)), target)
                retry = true
                break
            elseif startswith(status, "override ")
                guess = lowercase(last(split(status)))
                println("Overriding guess: $guess")
                continue
            # elseif status == "delete"
            #     popfirst!(dictionary)
            #     retry = true
            #     break
            end

            validate(N, uppercase(status)) && break
        end
        retry && continue

        states = parse_status(Val(N), uppercase(status))
        nextschema = result_schema(guess, states)
        schema = merge(schema, nextschema)
        target = DataStructures.OrderedSet(Iterators.filter(schema, target))
        if hardmode
            filter!(schema, dictionary)
        end
        first_iteration = false
    end
    return getword(only(target))
end

function validate(N, status)
    if length(status) != N
        println("Result must be $N characters long!")
        println()
        return false
    end

    for (i, char) in enumerate(status)
        if !in(char, ('B', 'Y', 'G'))
            println("Unknown character $char at index $(i)!")
            println("Status characters must be one of (B, Y, G)")
            println()
            return false
        end
    end
    return true
end

function parse_status(::Val{N}, status::AbstractString) where {N}
    return ntuple(Val(N)) do i
        char = status[i]
        return (char == 'G') ? Green : (char == 'Y' ? Yellow : Gray)
    end
end
