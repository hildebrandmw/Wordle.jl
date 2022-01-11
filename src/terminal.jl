function newgame(wordlength::Integer, file; hardmode = false)
    # Read in the dictionary and filter the the appropriate length.
    d = Tuple.(collect.(tolength(wordlength, readwords(file))))
    @assert isa(d, Vector{NTuple{wordlength,Char}})
    return newgame(d; hardmode)
end

function newgame(dictionary::Vector{NTuple{N,Char}}; hardmode = false) where {N}
    schema = Schema{N}()
    target = DataStructures.OrderedSet(dictionary)

    while length(target) > 1
        @show length(target)
        scores = process_dictionary(schema, dictionary; target)
        sort!(Glue(scores, dictionary))

        # Display guesses and scores
        strings = join.(view(dictionary, 1:min(length(dictionary), 10)))
        first_scores = view(scores, 1:min(length(scores), 10))
        display(collect(zip(strings, first_scores)))
        guess = first(dictionary)
        println("Current Guess: $(join(guess))")

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
                foreach(x -> println(join(x)), target)
                retry = true
                break
            elseif startswith(status, "override ")
                guess = lowercase(last(split(status)))
                println("Overriding guess: $guess")
                continue
            elseif status == "delete"
                popfirst!(dictionary)
                retry = true
                break
            end

            validate(N, uppercase(status)) && break
        end
        retry && continue

        states = parse_status(Val(N), uppercase(status))
        nextschema = result_schema(guess, states)
        schema = merge!(nextschema, schema)
        target = DataStructures.OrderedSet(Iterators.filter(schema, target))
        if hardmode
            filter!(schema, dictionary)
        end
    end
    return join(only(target))
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
