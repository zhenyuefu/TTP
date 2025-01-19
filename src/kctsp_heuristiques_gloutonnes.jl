module KCTSPGreedy

include("TTP.jl")  

using .TTP
using Printf, Random, Plots

"""
    greedy_kctsp_route(instance::TTPInstance)

使用简单“最近邻”来获取 KCTSP 所用的路线（与普通TSP一样）。
返回: 一个 1-based 的城市序列 (如 [1, 5, 10, …, 1])。
"""
function greedy_kctsp_route(instance::TTPInstance)
    n = instance.numberOfNodes
    dist = [distances(instance, i, j) for i in 1:n, j in 1:n]

    visited = falses(n)
    visited[1] = true
    route = [1]
    current_city = 1

    for _ in 1:(n-1)
        next_city = 0
        min_dist = Inf
        for city in 1:n
            if !visited[city] && city != current_city
                if dist[current_city, city] < min_dist
                    min_dist = dist[current_city, city]
                    next_city = city
                end
            end
        end
        push!(route, next_city)
        visited[next_city] = true
        current_city = next_city
    end

    push!(route, 1)  # 回到城市1
    return route
end

"""
    compute_dist_carried(distMatrix, route)

给定路线 route 和 预构造的 distMatrix (size=nNodes×nNodes),
计算每个城市 c 到路线结束的总距离 dist_after[c]。
即：若 route 的顺序是 [1, ..., c, ..., 1],
    dist_after[c] = 总距离(从 c 开始直到最后一个城市).
"""
function compute_dist_carried(distMatrix::Matrix{Float64}, route::Vector{Int}, nNodes::Int)
    n = length(route)
    dist_after = fill(0.0, nNodes)

    # 先计算 routeDistance[i] = distMatrix[ route[i], route[i+1] ]
    routeDistance = Float64[]
    for i in 1:(n-1)
        push!(routeDistance, distMatrix[ route[i], route[i+1] ])
    end

    # position_in_route[c] = c在route中的位置
    position_in_route = fill(-1, nNodes)
    for i in 1:n
        c = route[i]
        position_in_route[c] = i
    end

    # totalDistFrom[i] = sum( routeDistance[i] .. routeDistance[n-1] )
    totalDistFrom = fill(0.0, n)
    for i in (n-1):-1:1
        totalDistFrom[i] = totalDistFrom[i+1] + routeDistance[i]
    end

    # 对每个城市 c => dist_after[c] = totalDistFrom[ position_in_route[c] ]
    for c in 1:nNodes
        idx = position_in_route[c]
        dist_after[c] = (idx < n) ? totalDistFrom[idx] : 0.0
    end

    return dist_after
end

"""
    greedy_kctsp_knapsack(instance, route, K, distMatrix)

给定路线，使用 "netGain = profit - K * weight * distCarried" 来对物品贪心排序。
distCarried 即在 route 下，该物品所在城市到路线终点的距离。
只在 netGain>0 且容量允许的情况下选择该物品。

返回: 0/1 向量 (长度=numberOfItems)
"""
function greedy_kctsp_knapsack(instance::TTPInstance, route::Vector{Int}, K::Float64, distMatrix::Matrix{Float64})
    m = instance.numberOfItems
    W = instance.capacityOfKnapsack
    itemsMatrix = instance.items
    nNodes = instance.numberOfNodes

    # 先预计算 dist_after, 只有一次
    dist_after = compute_dist_carried(distMatrix, route, nNodes)

    # 构造 netGain
    item_net_gain = Vector{Tuple{Int,Float64}}(undef, m)
    for i in 1:m
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]
        city   = itemsMatrix[i, 3]
        carriedDist = dist_after[city]
        netGain = profit - K*weight*carriedDist
        item_net_gain[i] = (i, netGain)
    end

    # 按 netGain 降序
    sort!(item_net_gain, by = x -> x[2], rev=true)

    packingPlan = fill(0, m)
    remainCap = W

    for (idx, gain) in item_net_gain
        if gain <= 0
            continue
        end
        weight = itemsMatrix[idx, 2]
        if weight <= remainCap
            packingPlan[idx] = 1
            remainCap -= weight
        end
    end

    return packingPlan
end

"""
    evaluate_kctsp(instance, route, packingPlan, K, distMatrix)

KCTSP 目标：
  objective = ∑(profit_of_chosen_items) - K * ∑( d_{ route[i], route[i+1] } * W_{ route[i] } )

其中离开城市 i 时的背包重量 W_i 需在遍历路径时逐步加上城市 i 的物品。

返回: (objective, rawDistance, totalProfit, finalWeight, transportCost)
"""
function evaluate_kctsp(
    instance::TTPInstance,
    route::Vector{Int},
    packingPlan::Vector{Int},
    K::Float64,
    distMatrix::Matrix{Float64}
)
    m = instance.numberOfItems
    itemsMatrix = instance.items
    n = length(route)

    # 1) totalProfit
    totalProfit = 0.0
    for i in 1:m
        if packingPlan[i] == 1
            totalProfit += itemsMatrix[i, 1]
        end
    end

    # 2) transportCost
    # 遍历 route，累加 localDist * currentWeight * K
    currentWeight = 0.0
    transportCost = 0.0
    rawDistance   = 0.0

    for i in 1:(n-1)
        c  = route[i]
        nc = route[i+1]

        # 若城市 c 有被选物品 => 加入背包
        for itemIdx in 1:m
            if packingPlan[itemIdx] == 1 && (itemsMatrix[itemIdx, 3] == c)
                currentWeight += itemsMatrix[itemIdx, 2]
            end
        end

        localDist = distMatrix[c, nc]
        rawDistance += localDist
        transportCost += (localDist * currentWeight * K)
    end

    localObjective = totalProfit - transportCost
    return (localObjective, rawDistance, totalProfit, currentWeight, transportCost)
end


"""
    solve_kctsp_greedy(instance::TTPInstance, K::Float64)

主函数：
1) 贪心路线 (nearest neighbor)
2) 只构造一次 distMatrix
3) 用 netGain = p - K*w*distCarried 选背包
4) evaluate
返回 TTPSolution
"""
function solve_kctsp_greedy(instance::TTPInstance, K::Float64)

    start_time = time_ns()
    # 1) 路线
    route = greedy_kctsp_route(instance)

    # 2) 一次性构造 distMatrix
    nNodes = instance.numberOfNodes
    distMatrix = [distances(instance, i, j) for i in 1:nNodes, j in 1:nNodes]

    # 3) knapsack
    packingPlan = greedy_kctsp_knapsack(instance, route, K, distMatrix)

    # 4) evaluate
    (obj, rawDist, tp, finalWeight, transCost) = evaluate_kctsp(
        instance, route, packingPlan, K, distMatrix
    )

    # 封装成 TTPSolution (借用 TTP 的数据结构)
    sol = TTPSolution(
        route,
        packingPlan;
        fp      = tp,
        ft      = 0.0,
        ftraw   = Int(round(rawDist)),
        ob      = obj,
        wend    = instance.capacityOfKnapsack - finalWeight,
        wendUsed= finalWeight,
        computationTime=0
    )

    end_time = time_ns()
    sol.computationTime = (end_time - start_time)
    return sol
end

"""
    plot_kctsp_solution(instance, sol)

绘制 KCTSP 解的示意图：城市位置、路线，以及每个城市带走物品的数量。
"""
function plot_kctsp_solution(instance::TTPInstance, sol::TTPSolution)
    # 若需要基于坐标绘图
    cities = instance.nodes
    route = sol.tspTour

    # 收集路线坐标
    xs = [cities[route[i], 1] for i in 1:length(route)]
    ys = [cities[route[i], 2] for i in 1:length(route)]

    # 统计城市物品数
    items = instance.items
    packing = sol.packingPlan
    city_item_count = zeros(Int, instance.numberOfNodes)
    for i in 1:length(packing)
        if packing[i] == 1
            c = items[i, 3]
            city_item_count[c] += 1
        end
    end

    # 颜色：根据城市物品数做简单灰度区分
    colors = Vector{RGB}(undef, instance.numberOfNodes)
    max_items = 10
    for c in 1:instance.numberOfNodes
        k = min(city_item_count[c], max_items)
        g = 1.0 - (k/max_items)
        colors[c] = RGB(g, g, g)
    end
    cityX = [cities[i,1] for i in 1:instance.numberOfNodes]
    cityY = [cities[i,2] for i in 1:instance.numberOfNodes]

    # 绘制散点 + 路线
    title_str = string(instance.problemName, "_",
                       "items=", instance.numberOfItems, "_",
                       "obj=", Int64(round(sol.ob)), "_",
                       "time=", Int64(round(sol.computationTime/1000000)))

    plt = scatter(
        cityX, cityY, 
        marker=:circle, 
        color=colors, 
        ms=5, 
        title=title_str,
        label="Cities"
    )
    # 连线
    plot!(plt, xs, ys, seriestype=:path, linecolor=:blue, label="Route")
    savefig(plt, "results/greedy/KCTSP_$title_str.png")
    return plt
end

"""
    test_kctsp_greedy()

示例：读取某KCTSP-like文件(与TTP同格式)，贪心求解并打印解
"""
function test_kctsp_greedy()
    # filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/a280_n2790_uncorr_10.ttp.txt"
    # filename = "data/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/fnl4461_n22300_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/fnl4461_n44600_uncorr_10.ttp.txt"
    # filename = "data/pla33810_n33809_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/pla33810_n169045_uncorr-similar-weights_05.ttp.txt"
    filename = "data/pla33810_n338090_uncorr_10.ttp.txt"
    instance = TTPInstance(filename)
    K = instance.rentingRatio / instance.maxSpeed / instance.capacityOfKnapsack

    @info "Running KCTSP greedy on $(instance.problemName), K=$K"

    sol = solve_kctsp_greedy(instance, K)
    println("\n--- KCTSP Greedy Solution ---")
    TTP.printlnSolution(sol)
    plt = plot_kctsp_solution(instance, sol)
end

test_kctsp_greedy()

end # module KCTSPGreedy
