include("TTP.jl")  

using .TTP  # 假设 TTP.jl 与本文件同在 src/ 目录下，需要在主文件中: include("TTP.jl"), 然后 using .TTP
using Printf, Random

# =========== 1) 贪心TSP：最近邻策略 ===========
"""
    greedy_tsp(instance::TTPInstance)

基于最近邻（Nearest Neighbor）策略构造一个 TSP 路线。
- 从城市 1 (index=1) 开始，反复选取与当前城市最近的、未访问过的城市。
- 最后回到城市 1 构成一个封闭环。

返回：一个 1-based 的城市访问序列（如 [1, 5, 10, ..., 1]）
"""
function greedy_tsp(instance::TTPInstance)
    n = instance.numberOfNodes

    # 1. 准备所有城市的距离，存到一个 2D 数组 dist[i,j]
    dist = Array{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = distances(instance, i, j)
    end

    # 2. 标记未访问城市, 从城市1出发
    visited = falses(n) 
    visited[1] = true
    route = [1]

    current_city = 1
    for _ in 1:(n-1)  # 还需访问 n-1 个城市
        # 寻找距离 current_city 最近的未访问城市
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
        # 更新
        push!(route, next_city)
        visited[next_city] = true
        current_city = next_city
    end

    # 3. 回到城市1
    push!(route, 1)

    return route
end


# =========== 2) 贪心选物品：价值/重量比 ===========
"""
    greedy_knapsack(instance::TTPInstance)

基于价值/重量（profit/weight）比值的贪心策略，对背包中的物品进行挑选。
- 按照 (profit/weight) 从大到小排序后，依次尝试放入背包
- 只要剩余容量足够，就选该物品，否则跳过

返回：一个长度为 instance.numberOfItems 的 0/1 向量
"""
function greedy_knapsack(instance::TTPInstance)
    m = instance.numberOfItems
    capacity = instance.capacityOfKnapsack
    itemsMatrix = instance.items
    n = instance.numberOfNodes

    # 过滤掉分配在city=1或city=n的物品:
    valid_items_indices = []
    for i in 1:m
        city = itemsMatrix[i, 3]
        if city == 1 || city == n
            continue
        end
        push!(valid_items_indices, i)
    end

    # 构造 (idx, p, w, ratio)
    item_list = []
    for idx in valid_items_indices
        p = itemsMatrix[idx, 1]
        w = itemsMatrix[idx, 2]
        ratio = p / w
        push!(item_list, (idx, p, w, ratio))
    end
    sort!(item_list, by=x->x[4], rev=true)  # ratio降序
    
    packingPlan = fill(0, m)
    remainingCap = capacity
    for (idx, p, w, ratio) in item_list
        if w <= remainingCap
            packingPlan[idx] = 1
            remainingCap -= w
        end
        # println("idx=$idx, p=$p, w=$w, ratio=$ratio, remainingCap=$remainingCap")
    end
    return packingPlan
end


# =========== 3) 整合：solve_ttp_greedy ===========
"""
    solve_ttp_greedy(instance::TTPInstance)

对给定的 TTP 实例使用“贪心TSP + 贪心选物品”的方法，返回一个 TTPSolution。
步骤：
1. 使用 `greedy_tsp` 得到城市访问顺序
2. 使用 `greedy_knapsack` 得到背包选取方案
3. 构造 TTPSolution，调用 evaluate 计算目标函数值
"""
function solve_ttp_greedy(instance::TTPInstance)
    # 1) 得到贪心TSP路线 (1-based 序列)
    route = greedy_tsp(instance)

    # 2) 得到贪心背包方案
    packingPlan = greedy_knapsack(instance)

    # 3) 构造TTPSolution并评估
    sol = TTPSolution(
        route,
        packingPlan;
        fp=-Inf,
        ft=Inf,
        ftraw=typemax(Int),
        ob=-Inf,
        wend=Inf,
        wendUsed=Inf,
        computationTime=0
    )

    start_time = time_ns()
    evaluate(instance, sol)
    end_time = time_ns()
    sol.computationTime = (end_time - start_time)

    return sol
end


# =========== 4) 简单测试示例 ===========

"""
    test_greedy_heuristic()

示例测试函数，读取 data/a280_n279_bounded-strongly-corr_01.ttp.txt 文件，
用贪心法生成解并打印评估结果。
"""
function test_greedy_heuristic()
    # 替换为实际路径
    filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
    
    # 构造TTP实例
    instance = TTPInstance(filename)
    @printf("\n[TEST] Loaded instance: %s\n", instance.problemName)

    # 贪心求解
    sol = solve_ttp_greedy(instance)

    # 打印结果
    @printf("\n--- Greedy Solution ---\n")
    TTP.printlnSolution(sol)

    # 如果想查看详细路线、背包，可以：
    # println("Route = ", sol.tspTour)
    # println("PackingPlan = ", sol.packingPlan)
end

test_greedy_heuristic()  # 测试