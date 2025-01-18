using JuMP
using Distances
import Plots
import Gurobi

include("TTP.jl")
using .TTP

function build_kctsp_model(instance::TTP.TTPInstance)
    # matrix(X,Y)
    nodes = instance.nodes

    # matrix(profit,weight,city)
    items = instance.items

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

    W = instance.capacityOfKnapsack
    K = instance.rentingRatio

    d = pairwise(Euclidean(), nodes', dims=2)

    model = Model(Gurobi.Optimizer)

    # Decision variables:
    # x[i,j] = 1 if the path goes directly from city i to j
    @variable(model, x[1:n, 1:n], Bin)
    # y[i,k] = 1 if item k in city i is taken
    @variable(model, y[i=2:n, k=1:itemsPerCity], Bin)
    # W_i = current knapsack weight upon *departing* city i
    @variable(model, W_i[1:n] >= 0)

    # Objective: maximize total profit minus transport cost
    # cost = K * Σ_{(i->j)} d[i,j]*W_i * x[i,j]
    # profit = Σ p_{i,k} * y[i,k]
    @objective(model, Max,
        sum(p[i, k] * y[i, k] for i = 2:n for k = 1:itemsPerCity)
        -
        K * sum(d[i, j] * W_i[i] * x[i, j] for i in 1:n, j in 1:n if i != j))

    # Knapsack capacity
    @constraint(model,
        sum(w[i, k] * y[i, k] for i = 2:n for k = 1:itemsPerCity) <= W
    )

    # Basic TSP constraints: each city has in-degree 1 and out-degree 1
    @constraint(model, [i in 1:n], sum(x[i, :]) == 1)
    @constraint(model, [j in 1:n], sum(x[:, j]) == 1)
    @constraint(model, [i in 1:n], x[i, i] == 0)

    # Weight propagation (the "two-sided" big-M trick):
    # If x[i,j]==1, then W_j must be exactly W_i + sum-of-items-chosen-in-j.
    # Using M = W (the max capacity) as big-M.
    @constraint(model, [i in 1:n, j in 2:n; i != j],
        W_i[j] >= W_i[i] + sum(w[j, k] * y[j, k] for k = 1:itemsPerCity)
                  -
                  W * (1 - x[i, j])
    )
    @constraint(model, [i in 1:n, j in 2:n; i != j],
        W_i[j] <= W_i[i] + sum(w[j, k] * y[j, k] for k = 1:itemsPerCity)
                  +
                  W * (1 - x[i, j])
    )

    # W_i cannot exceed capacity
    @constraint(model, [i in 1:n], W_i[i] <= W)


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
m = instance.numberOfItems

itemsPerCity = m ÷ (n - 1)
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

packingPlan = value.(lazy_model[:y])

# packingPlan[i, k] 表示第 i 个城市的第 k 个物品是否被拾取
# 转换为一个一维数组，packingPlan[i] 表示第 i 个物品是否被拾取
newpackingPlan = [round(Int, packingPlan[i, k]) for k = 1:itemsPerCity for i = 2:n]

sol = TTPSolution(tspTour, newpackingPlan)
# convert time to ms
sol.computationTime = round(Int, time_lazy * 1000)
TTP.evaluate(instance, sol)
TTP.printFullSolution(sol)