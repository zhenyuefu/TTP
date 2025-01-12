module Utils
    using Dates

    function start_timing()
        return time_ns()
    end

    function stop_timing(start_time)
        stop_time = time_ns()
        return (stop_time - start_time) ÷ 1_000_000  # Convert to milliseconds
    end

    function stop_timing_print(start_time)
        elapsed = stop_timing(start_time)
        println("[time used: $(elapsed)ms]")
        return elapsed
    end

    function get_file_list(dir_path::String, pattern::String)
        files = String[]
        for file in readdir(dir_path, join=true)
            if endswith(file, ".ttp") && contains(file, pattern)
                push!(files, file)
            end
        end
        return files
    end

    function permutation(n::Int)
        return shuffle(1:n)
    end
end
