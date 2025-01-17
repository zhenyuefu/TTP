using JuMP
using Distances
import Plots
import Gurobi

include("TTP.jl")
using .TTP

function build_kctsp_model(instance::TTP.TTPInstance)
    n = instance.numberOfNodes
    m = instance.numberOfItems

    itemsPerCity = m ÷ (n - 1)

    # matrix(X,Y)
    nodes = instance.nodes

    # matrix(profit,weight,city)
    items = instance.items

    # w is the 2nd col of items matrix
    w = [items[i, 2] for i = 1:m]
    p = [items[i, 1] for i = 1:m]

    W = instance.capacityOfKnapsack
    K = instance.rentingRatio

    d = pairwise(Euclidean(), nodes', dims=2)

    model = Model(Gurobi.Optimizer)

    @variable(model, x[1:n, 1:n], Bin, Symmetric)
    @variable(model, y[i=2:n, k=1:itemsPerCity], Bin)
    @variable(model, W_i[2:n] >= 0) # 离开每个城市时的重量

    @objective(model, Max,
        sum(p[(i-2)*itemsPerCity+k] * y[i, k] for i = 2:n for k = 1:itemsPerCity) -
        K * (d[1, n] * W_i[n] + sum(d[i, i+1] * W_i[i] for i = 2:n-1))
    )

    # TSP constraints
    @constraint(model, [i in 1:n], sum(x[i, :]) == 2)
    @constraint(model, [i in 1:n], x[i, i] == 0)

    # 背包容量约束
    @constraint(model,
        sum(w[(i-2)*itemsPerCity+k] * y[i, k] for i = 2:n for k = 1:itemsPerCity) <= W
    )

    # 累计重量 W_i == sum_k=1:i sum_j = 1:m_i w_kj * y_kj
    @constraint(
        model, [i in 2:n], W_i[i] == sum(w[(j-2)*itemsPerCity+k] * y[i, k] for j = 2:i for k = 1:itemsPerCity)
    )


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
    status = callback_node_status(cb_data, lazy_model)
    if status != MOI.CALLBACK_NODE_STATUS_INTEGER
        return  # Only run at integer solutions
    end
    cycle = subtour(callback_value.(cb_data, lazy_model[:x]))
    if !(1 < length(cycle) < n)
        return  # Only add a constraint if there is a cycle
    end
    S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
    con = @build_constraint(
        sum(lazy_model[:x][i, j] for (i, j) in S) <= length(cycle) - 1,
    )
    MOI.submit(lazy_model, MOI.LazyConstraint(cb_data), con)
    return
end



filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
instance = TTPInstance(filename)
n = instance.numberOfNodes
lazy_model = build_kctsp_model(instance)
set_attribute(
    lazy_model,
    MOI.LazyConstraintCallback(),
    subtour_elimination_callback,
)
optimize!(lazy_model)


obj = objective_value(lazy_model)
println("Objective value: $obj")
time_lazy = solve_time(lazy_model)
println("Time to solve: $time_lazy")


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

tspTour = final_tsp_tour(value.(lazy_model[:x]))
println("Tour: $tspTour")

