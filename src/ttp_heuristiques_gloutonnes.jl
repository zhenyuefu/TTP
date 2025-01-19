include("TTP.jl")  # 假设 TTP.jl 与本文件在同一目录下

using .TTP         # 导入 TTPInstance, TTPSolution, distances, evaluate 等
using Printf, Random, Plots

# =========== 1) 贪心TSP：最近邻策略 ===========

"""
    greedy_tsp(instance::TTPInstance)

基于最近邻策略构造 TSP 路线：
- 从城市1开始，每次选择当前城市最近的未访问城市；
- 最后回到起点，构成闭合路线。

返回：1-based 的城市访问序列 (如 [1, 5, 10, …, 1])
"""
function greedy_tsp(instance::TTPInstance)
    n = instance.numberOfNodes

    # 预计算距离矩阵，避免重复计算
    dist = Array{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = distances(instance, i, j)
    end

    visited = falses(n)
    visited[1] = true
    route = [1]
    current_city = 1

    for _ in 1:(n-1)
        next_city = 0
        min_dist = Inf
        for city in 1:n
            if !visited[city] && city != current_city && dist[current_city, city] < min_dist
                min_dist = dist[current_city, city]
                next_city = city
            end
        end
        push!(route, next_city)
        visited[next_city] = true
        current_city = next_city
    end

    push!(route, 1)  # 回到起点
    return route
end


# =========== 2) 近似额外时间 + 贪心选物品 ===========

"""
    approximate_item_time(
        instance, route, city, itemWeight;
        wc=0.0,
        distmat::Matrix{Float64}
    )

估算在城市 city 拿取物品（重量 itemWeight）后，对后续路程增加的额外时间。
TTP 速度公式:
  time = sum(distance / (vmax - currentWeight * (vmax-vmin)/W))

此处做简化：假设背包当前已有 wc≈0，主要关注 itemWeight 对速度的影响。
需要外部提前构造好 distmat[i,j] = 距离(i,j)，并将其作为参数传入。
"""
function approximate_item_time(
    instance::TTPInstance,
    route::Vector{Int},
    city::Int,
    itemWeight::Int;
    wc::Float64=0.0,
    distmat::Matrix{Float64}
)
    W   = instance.capacityOfKnapsack
    fac = (instance.maxSpeed - instance.minSpeed) / W

    pos = findfirst(==(city), route)
    if pos === nothing
        return 0.0
    end

    extra_time = 0.0
    for i in pos:(length(route)-1)
        c1, c2 = route[i], route[i+1]
        d  = distmat[c1, c2]
        old_speed = max(instance.maxSpeed - wc*fac, 1e-6)
        new_speed = max(instance.maxSpeed - (wc + itemWeight)*fac, 1e-6)
        extra_time += d/new_speed - d/old_speed
    end
    return extra_time
end


"""
    greedy_knapsack_with_time(instance, route, distmat)

基于“单位时间收益” = profit / extra_time 的贪心法选物品：
1. 过滤掉城市1和城市n的物品；
2. 对每件物品 i，计算 approximate_item_time(...) 获得其额外时间 dtime；
3. ratio_i = profit_i / dtime；
4. 按 ratio_i 降序排序后依次选择，直到容量耗尽或 ratio≤0。

参数 distmat 必须是事先预构造的 n×n 距离矩阵。

返回：packingPlan (0/1 向量，长度 = numberOfItems)。
"""
function greedy_knapsack_with_time(instance::TTPInstance, route::Vector{Int}, distmat::Matrix{Float64})
    m = instance.numberOfItems
    capacity = instance.capacityOfKnapsack
    itemsMatrix = instance.items

    item_list = []
    for i in 1:m
        city = itemsMatrix[i, 3]
        # 过滤起点 / 终点
        if city == 1 || city == instance.numberOfNodes
            continue
        end
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]

        dtime = approximate_item_time(instance, route, city, weight; wc=0.0, distmat=distmat)
        dtime = max(dtime, 1e-9)
        ratio = profit / dtime
        push!(item_list, (i, profit, weight, ratio))
    end

    sort!(item_list, by = x -> x[4], rev=true)
    packingPlan = fill(0, m)
    remainingCap = capacity

    for (idx, p, w, ratio) in item_list
        if w <= remainingCap && ratio > 0
            packingPlan[idx] = 1
            remainingCap -= w
        end
    end
    return packingPlan
end


# =========== 3) 整合：solve_ttp_greedy ===========

"""
    solve_ttp_greedy(instance::TTPInstance)

使用“基础贪心”方法求解 TTP：
1. 用 greedy_tsp 得到城市访问顺序；
2. 一次性构造距离矩阵 distmat；
3. 用 greedy_knapsack_with_time 选物品；
4. 构造 TTPSolution 并 evaluate；
5. 记录总时间并返回。

这是一个单轮的贪心，用于与更复杂的算法做对比参考。
"""
function solve_ttp_greedy(instance::TTPInstance)
    start_time = time_ns()

    # (1) 贪心TSP 路线
    route = greedy_tsp(instance)

    # (2) 预先构造一次距离矩阵 distmat (避免在 approximate_item_time 中重复构造)
    n = instance.numberOfNodes
    distmat = [distances(instance, i, j) for i in 1:n, j in 1:n]

    # (3) 基于单位时间收益选物品
    packingPlan = greedy_knapsack_with_time(instance, route, distmat)

    # (4) 构建 TTPSolution 并评估
    sol = TTPSolution(route, packingPlan;
                      fp=-Inf, ft=Inf, ftraw=typemax(Int),
                      ob=-Inf, wend=Inf, wendUsed=Inf,
                      computationTime=0)

    evaluate(instance, sol)

    end_time = time_ns()
    sol.computationTime = end_time - start_time
    return sol
end


# =========== 4) 结果可视化 ===========

"""
    plot_ttp_solution(instance, sol)

绘制TTP解的示意图：
- 绘制所有城市散点，并用灰度表示该城市带出的物品数量；
- 用线连接路线；
- 标题包含 problemName、items数量、obj值、时间（ms）。

保存图片到 results/greedy/ 目录下。
"""
function plot_ttp_solution(instance::TTPInstance, sol::TTPSolution)
    n = instance.numberOfNodes
    X = [instance.nodes[i,1] for i in 1:n]
    Y = [instance.nodes[i,2] for i in 1:n]

    route = sol.tspTour
    routeX = [X[c] for c in route]
    routeY = [Y[c] for c in route]

    items = instance.items
    packingPlan = sol.packingPlan
    city_item_count = zeros(Int, n)
    for i in 1:length(packingPlan)
        if packingPlan[i] == 1
            c = items[i, 3]
            city_item_count[c] += 1
        end
    end

    colors = Vector{RGB}(undef, n)
    max_items = 10
    for c in 1:n
        k = min(city_item_count[c], max_items)
        gray_level = 1.0 - (k / max_items)
        colors[c] = RGB(gray_level, gray_level, gray_level)
    end

    # 标题：problemName, items=m, obj=..., time=...
    # 时间以毫秒显示
    title_str = string(
        instance.problemName, "_",
        "items=", instance.numberOfItems, "_",
        "obj=", Int(round(sol.ob)), "_",
        "time=", Int(round(sol.computationTime / 1e6))
    )

    plt = scatter(
        X, Y,
        marker=:circle,
        color=colors,
        ms=6,
        label="Cities",
        title=title_str,
        legend=:topright
    )
    plot!(
        plt,
        routeX, routeY,
        seriestype=:path,
        lw=2,
        linecolor=:skyblue,
        label="Route"
    )

    # 自动保存
    mkpath("results/greedy")  # 确保目录存在
    savefig(plt, "results/greedy/TTP_$title_str.png")
    return plt
end


# =========== 5) 测试函数 ===========

"""
    test_greedy_heuristic()

示例测试函数：
- 从指定文件构造 TTPInstance；
- 用 solve_ttp_greedy 做基础贪心求解；
- 打印结果并绘图展示。
"""
function test_greedy_heuristic()
    # 根据需要选择合适文件
    filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/a280_n2790_uncorr_10.ttp.txt"
    # filename = "data/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/fnl4461_n22300_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/fnl4461_n44600_uncorr_10.ttp.txt"
    # filename = "data/pla33810_n33809_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/pla33810_n169045_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/pla33810_n338090_uncorr_10.ttp.txt"
    # 也可换其他文件进行测试
    instance = TTPInstance(filename)
    @printf("[TEST] Loaded instance: %s\n", instance.problemName)

    sol = solve_ttp_greedy(instance)

    @printf("\n--- Greedy Solution ---\n")
    TTP.printlnSolution(sol)

    plt = plot_ttp_solution(instance, sol)
    display(plt)
end

test_greedy_heuristic()
