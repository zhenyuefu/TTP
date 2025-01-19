using JuMP
using Distances
import Plots
import Gurobi

include("TTP.jl")
using .TTP

function build_tsp_model(d::Matrix, n::Int)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, x[1:n, 1:n], Bin, Symmetric)

    @objective(model, Min, sum(d .* x) / 2)

    # TSP constraints
    @constraint(model, [i in 1:n], sum(x[i, :]) == 2)
    @constraint(model, [i in 1:n], x[i, i] == 0)

    return model
end

function subtour(edges::Vector{Tuple{Int,Int}}, n)
    shortest_subtour, unvisited = collect(1:n), Set(collect(1:n))
    while !isempty(unvisited)
        this_cycle, neighbors = Int[], unvisited
        while !isempty(neighbors)
            current = pop!(neighbors)
            push!(this_cycle, current)
            if length(this_cycle) > 1
                pop!(unvisited, current)
            end
            neighbors =
                [j for (i, j) in edges if i == current && j in unvisited]
        end
        if length(this_cycle) < length(shortest_subtour)
            shortest_subtour = this_cycle
        end
    end
    return shortest_subtour
end

subtour(x::Matrix{Float64}) = subtour(selected_edges(x, size(x, 1)), size(x, 1))
subtour(x::AbstractMatrix{VariableRef}) = subtour(value.(x))

function selected_edges(x::Matrix{Float64}, n)
    return Tuple{Int,Int}[(i, j) for i in 1:n, j in 1:n if x[i, j] > 0.5]
end



function subtour_elimination_callback(cb_data)
    status = callback_node_status(cb_data, tsp_model)
    if status != MOI.CALLBACK_NODE_STATUS_INTEGER
        return  # Only run at integer solutions
    end
    cycle = subtour(callback_value.(cb_data, tsp_model[:x]))
    if !(1 < length(cycle) < n)
        return  # Only add a constraint if there is a cycle
    end
    S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
    con = @build_constraint(
        sum(tsp_model[:x][i, j] for (i, j) in S) <= length(cycle) - 1,
    )
    MOI.submit(tsp_model, MOI.LazyConstraint(cb_data), con)
    return
end



filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
instance = TTPInstance(filename)
n = instance.numberOfNodes
m = instance.numberOfItems
nodes = instance.nodes
d = pairwise(Euclidean(), nodes', dims=2)
itemsPerCity = m ÷ (n - 1)

tsp_model = build_tsp_model(d, n)
set_attribute(
    tsp_model,
    MOI.LazyConstraintCallback(),
    subtour_elimination_callback,
)
optimize!(tsp_model)


time_tsp = solve_time(tsp_model)



function final_tsp_tour(x::Matrix{Float64})
    n = size(x, 1)
    edges = selected_edges(x, n)
    path = [1]
    current = 1
    unvisited = Set(collect(2:n))
    neighbors = [j for (i, j) in edges if i == current && j in unvisited]
    while !isempty(neighbors)
        current = pop!(neighbors)
        push!(path, current)

        pop!(unvisited, current)

        neighbors =
            [j for (i, j) in edges if i == current && j in unvisited]
    end

    push!(path, 1)
    return path
end

tspTour = final_tsp_tour(value.(tsp_model[:x]))


function build_kp_model(instance::TTP.TTPInstance, tspTour::Vector{Int})

    final_city = tspTour[n]

    items = instance.items
    W = instance.capacityOfKnapsack
    R = instance.rentingRatio
    vmax = instance.maxSpeed
    K = R / vmax / W
    w = zeros(n, itemsPerCity)
    p = zeros(n, itemsPerCity)
    idx = 1
    for j in 1:itemsPerCity
        for i in 2:n
            w[i, j] = items[idx, 2]
            p[i, j] = items[idx, 1]
            idx += 1
        end
    end



    model = Model(Gurobi.Optimizer)
    set_silent(model)
    @variable(model, y[2:n, 1:itemsPerCity], Bin)
    # W_i[i] 表示第路线上第 i 个城市的背包中的物品总重量
    @variable(model, W_i[1:n] >= 0)

    @objective(model, Max,
        sum(p[i, k] * y[i, k] for i = 2:n for k = 1:itemsPerCity)
        -
        K * (d[final_city, 1] * W_i[n] + sum(d[i+1, i] * W_i[i] for i = 2:n-1)))

    # Knapsack capacity
    @constraint(model,
        sum(w[i, k] * y[i, k] for i = 2:n for k = 1:itemsPerCity) <= W
    )

    # W_i
    @constraint(model, W_i[1] == 0)
    @constraint(model, [i in 2:n],
        W_i[i] == W_i[i-1] + sum(w[tspTour[i], k] * y[tspTour[i], k] for k = 1:itemsPerCity)
    )

    return model
end


kp_model = build_kp_model(instance, tspTour)
optimize!(kp_model)

time_kp = solve_time(kp_model)

packingPlan = value.(kp_model[:y])
packingPlan = [round(Int, packingPlan[i, k]) for k = 1:itemsPerCity for i = 2:n]

sol = TTPSolution(tspTour, packingPlan)
sol.computationTime = round(Int, (time_tsp + time_kp) * 1000)
TTP.evaluate(instance, sol)
obj = objective_value(kp_model)
sol.ob = obj
TTP.printFullSolution(sol)