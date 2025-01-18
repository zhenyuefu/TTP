module KCTSPGreedy

include("TTP.jl")  

using .TTP
using Printf, Random

"""
    greedy_kctsp_route(instance::TTPInstance)

使用简单“最近邻”来获取KCTSP所用的路线（与普通TSP一样）。
返回: 一个 1-based 的城市序列 (如 [1, 5, 10, ..., 1])。
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
    compute_dist_carried(instance::TTPInstance, route::Vector{Int})

对给定路线，每个城市 c (1-based) 到路线结束（或回到起点）的距离。
返回: dist_after[c]，表示从城市 c 在该route里开始，直到 route 终点（route[end]）的总距离。
"""
function compute_dist_carried(instance::TTPInstance, route::Vector{Int})
    n = length(route)
    dist = [distances(instance, i, j) for i in 1:instance.numberOfNodes,
                                         j in 1:instance.numberOfNodes]

    # route[i] 表示第 i 个访问城市(1-based城市号)
    # dist_after[c] = sum of distances from c to the end of route
    dist_after = fill(0.0, instance.numberOfNodes)  # 1..instance.numberOfNodes

    # 先把 route 每段距离预计算
    # routeDistance[i] = d( route[i], route[i+1] )
    routeDistance = Float64[]
    for i in 1:(n-1)
        push!(routeDistance, dist[route[i], route[i+1]])
    end

    # 从后往前累加
    # totalDist[i] = routeDistance[i] + routeDistance[i+1] + ...
    #               = sum of all edges from step i to the last step
    # 但是要和城市号对齐
    # 先做一个 "index_of_city[c]" -> 在route中的位置
    position_in_route = fill(-1, instance.numberOfNodes)
    for i in 1:n
        c = route[i]
        position_in_route[c] = i
    end

    # precompute prefix from the end:
    # totalDistFrom[i] = sum(routeDistance[i], ..., routeDistance[n-1])
    totalDistFrom = fill(0.0, n)
    for i in (n-1):-1:1
        totalDistFrom[i] = totalDistFrom[i+1] + routeDistance[i]
    end

    # 对每个城市 c，dist_after[c] = totalDistFrom[pos(c)]
    # 如果是 route 的最后一个点(就是回到1了)，则 cost=0
    for c in 1:instance.numberOfNodes
        idx = position_in_route[c]
        if idx < n
            dist_after[c] = totalDistFrom[idx]
        else
            dist_after[c] = 0.0
        end
    end

    return dist_after
end


"""
    greedy_kctsp_knapsack(instance::TTPInstance, route::Vector{Int}, K::Float64)

在给定路线下，对每个物品计算其"净收益" = profit - K*weight*distCarried。
其中 distCarried 是从物品所在城市到路线终点所经过的距离。

最后使用"净收益"从大到小做贪心，直到背包容量满。
若净收益 <= 0，则不选它(除非你想试试把它当成0收益处理)。

返回: 一个长度=numberOfItems的0/1选择向量
"""
function greedy_kctsp_knapsack(instance::TTPInstance, route::Vector{Int}, K::Float64)
    m = instance.numberOfItems
    W = instance.capacityOfKnapsack
    itemsMatrix = instance.items  # each row: (profit, weight, city)

    # 先计算 "distCarried[c]"：城市 c 出发到route结束(或回到起点)的距离
    dist_after = compute_dist_carried(instance, route)

    # 构造"有效净收益"
    # netGain[i] = p_i - K * w_i * dist_after[ city_i ]
    item_net_gain = Vector{Tuple{Int,Float64}}(undef, m)

    for i in 1:m
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]
        city   = itemsMatrix[i, 3]
        # 如果 city=1 或 city与最后相同 也可以跳过
        # 这里不强制，因为它在后面可自行比较 netGain
        carriedDist = dist_after[city]
        netGain     = profit - K*weight*carriedDist
        item_net_gain[i] = (i, netGain)
    end

    # 按 netGain 降序排序
    sort!(item_net_gain, by = x -> x[2], rev=true)

    packingPlan = fill(0, m)
    remainingCap = W

    for (idx, gain) in item_net_gain
        profit = itemsMatrix[idx, 1]
        weight = itemsMatrix[idx, 2]

        # 如果 netGain <= 0，选它只会降低目标函数
        if gain <= 0
            continue
        end

        if weight <= remainingCap
            packingPlan[idx] = 1
            remainingCap -= weight
        end
    end

    return packingPlan
end


"""
    evaluate_kctsp(instance::TTPInstance, route, packingPlan, K)

线性目标:
    sum( profit_of_chosen_items ) - K * sum_{(i->i+1)} [ d_{i,i+1} * W_i ],
其中 W_i = 离开城市 i 时背包内的重量。

返回:
    (objective, rawDistance, totalProfit, totalWeight, transportCost)
"""
function evaluate_kctsp(instance::TTPInstance, route::Vector{Int}, packingPlan::Vector{Int}, K::Float64)
    m = instance.numberOfItems
    itemsMatrix = instance.items
    n = length(route)
    distMatrix = [distances(instance, i, j) for i in 1:instance.numberOfNodes, j in 1:instance.numberOfNodes]

    # 1) 先统计 totalProfit
    totalProfit = 0.0
    for i in 1:m
        if packingPlan[i] == 1
            totalProfit += itemsMatrix[i, 1]  # profit
        end
    end

    # 2) 计算运输成本 linear: cost = K * sum(d_{x_i, x_{i+1}} * W_{x_i})
    #    这里 x_i = route[i]; W_{x_i} 是从城市 route[i] 出发时的重量
    #    => 所以我们需要随着路线的推进，计算当前背包重量
    currentWeight = 0.0
    transportCost = 0.0
    rawDistance   = 0.0

    # 我们需知道在“离开某城市 c”时背包里是多少重量。
    # => 当到城市 c 时，把 c 所在城市物品全部加入(若 packingPlan选了)
    # 但与 TTP 不同，这里不需要做速度衰减计算，只需线性 cost
    # route[i], route[i+1] 都是1-based城市号
    # i=1: 第一个城市(往往是1)
    for i in 1:(n-1)
        c  = route[i]
        nc = route[i+1]   # next city

        # 把城市 c 的物品加入背包(若我们此时才拿)
        # 这里有两种模型: 
        #   - 只在到达城市 c 的那一刻拿上它(一般KCTSP也这么想)
        #   - 可能 c=1(起点)没有物品
        # 如果 city c 有被选中的物品，就把它们加进 currentWeight
        for itemIdx in 1:m
            if packingPlan[itemIdx] == 1
                if itemsMatrix[itemIdx, 3] == c
                    # 选中的物品在城市 c
                    currentWeight += itemsMatrix[itemIdx, 2]
                end
            end
        end

        localDist = distMatrix[c, nc]
        rawDistance += localDist

        # cost for this edge = localDist * currentWeight
        transportCost += localDist * currentWeight * K
    end

    # 最终objective
    localObjective = totalProfit - transportCost

    return (localObjective, rawDistance, totalProfit, currentWeight, transportCost)
end


"""
    solve_kctsp_greedy(instance::TTPInstance, K::Float64)

主函数: 
1) 用最近邻生成路线 
2) 按( profit - K * weight * distCarried )贪心选背包
3) 用 evaluate_kctsp 求出最终目标值
4) 返回 TTPSolution (或其他结构), 记录关键信息
"""
function solve_kctsp_greedy(instance::TTPInstance, K::Float64)
    # 1) 路线
    route = greedy_kctsp_route(instance)

    # 2) knapsack
    packingPlan = greedy_kctsp_knapsack(instance, route, K)

    # 3) 计算目标值
    (obj, rawDist, tp, finalWeight, transCost) = evaluate_kctsp(instance, route, packingPlan, K)

    # 将结果存入 TTPSolution (可以重用 TTP 的 TTPSolution 或自己定义)
    sol = TTPSolution(
        route,
        packingPlan;
        fp      = tp,          # total profit
        ft      = 0.0,         # 这里KCTSP没用ft(时间),随便置0
        ftraw   = Int64(round(rawDist)),
        ob      = obj,
        wend    = instance.capacityOfKnapsack - finalWeight,
        wendUsed= finalWeight,
        computationTime=0
    )

    return sol
end


"""
    test_kctsp_greedy()

示例：读取某个KCTSP文件(同一格式, 只是目标函数不同),
运行贪心, 并打印解
"""
function test_kctsp_greedy()
    filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
    instance = TTPInstance(filename)
    K = 0.001   # 每公里每公斤的运输成本(根据问题设定)

    @info "Running KCTSP greedy on $(instance.problemName), K=$K"

    sol = solve_kctsp_greedy(instance, K)
    println("\n--- KCTSP Greedy Solution ---")
    TTP.printlnSolution(sol)
end

test_kctsp_greedy()

end # module KCTSPGreedy
