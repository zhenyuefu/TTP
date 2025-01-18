include("TTP.jl")  

using .TTP  # 假设 TTP.jl 与本文件在同一目录下
using Printf, Random, Plots

# =========== 1) 贪心TSP：最近邻策略 ===========
"""
    greedy_tsp(instance::TTPInstance)

基于最近邻策略构造 TSP 路线：
- 从城市1开始，每次选择当前城市最近的未访问城市；
- 最后回到起点，构成闭合路线。

返回：1-based 的城市访问序列 (例如 [1, 5, 10, …, 1])
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


# =========== 2) 贪心选物品：价值/重量比、以及考虑"单位时间收益" ===========
"""
    approximate_item_time(instance, route, city, itemWeight; wc=0.0)

估算在城市 city 拿取物品（重量 itemWeight）后，对后续路程增加的额外时间。
该近似计算方法：
  对于 route 中从城市 city 开始的每条边，
  计算： d/ (vmax - (wc+itemWeight)*fac) - d/(vmax - wc*fac)
其中 fac=(vmax-vmin)/W。

返回近似的额外时间。
"""
function approximate_item_time(instance::TTPInstance, route::Vector{Int}, city::Int, itemWeight::Int; wc::Float64=0.0)
    W   = instance.capacityOfKnapsack
    fac = (instance.maxSpeed - instance.minSpeed) / W

    pos = findfirst(==(city), route)
    if pos === nothing
        return 0.0
    end

    # 预计算距离矩阵（若实例中未预先保存，可考虑全局保存以提升效率）
    n = instance.numberOfNodes
    distmat = Array{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        distmat[i, j] = distances(instance, i, j)
    end

    extra_time = 0.0
    for i in pos:(length(route)-1)
        c1, c2 = route[i], route[i+1]
        d = distmat[c1, c2]
        old_speed = max(instance.maxSpeed - wc*fac, 1e-6)
        new_speed = max(instance.maxSpeed - (wc+itemWeight)*fac, 1e-6)
        extra_time += d/new_speed - d/old_speed
    end
    return extra_time
end

"""
    greedy_knapsack_with_time(instance, route)

基于"单位时间收益"（profit / extra_time）的贪心策略进行物品选取：
1. 对于每个物品（过滤掉城市1和城市n的物品），计算其 profit、weight 与近似额外时间；
2. 计算收益比 ratio = profit / max(extra_time, ε)；
3. 按 ratio 降序依次选择物品，保证背包容量不超限。

返回：0/1 向量，长度为 instance.numberOfItems。
"""
function greedy_knapsack_with_time(instance::TTPInstance, route::Vector{Int})
    itemsMatrix = instance.items
    m = instance.numberOfItems
    capacity = instance.capacityOfKnapsack

    item_list = []
    for i in 1:m
        city = itemsMatrix[i, 3]
        # 过滤掉起点和终点
        if city == 1 || city == instance.numberOfNodes
            continue
        end
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]
        dtime = approximate_item_time(instance, route, city, weight; wc=0.0)
        # 避免除0
        dtime = max(dtime, 1e-9)
        ratio = profit / dtime
        push!(item_list, (i, profit, weight, ratio))
    end

    sort!(item_list, by=x->x[4], rev=true)
    packingPlan = fill(0, m)
    remainingCap = capacity

    for (idx, _, weight, ratio) in item_list
        if weight <= remainingCap && ratio > 0
            packingPlan[idx] = 1
            remainingCap -= weight
        end
    end
    return packingPlan
end


# =========== 3) 整合：solve_ttp_greedy ===========
"""
    solve_ttp_greedy(instance::TTPInstance)

贪心启发式求解 TTP 问题：
1. 使用 greedy_tsp 得到城市访问顺序；
2. 使用 greedy_knapsack_with_time 得到背包选择方案；
3. 构造 TTPSolution，并调用 evaluate 计算目标函数值；
4. 同时记录整个求解过程的计算时间。

返回：TTPSolution 对象。
"""
function solve_ttp_greedy(instance::TTPInstance)
    start_time = time_ns()

    route = greedy_tsp(instance)
    packingPlan = greedy_knapsack_with_time(instance, route)

    sol = TTPSolution(
        route,
        packingPlan;
        fp = -Inf,
        ft = Inf,
        ftraw = typemax(Int),
        ob = -Inf,
        wend = Inf,
        wendUsed = Inf,
        computationTime = 0
    )

    evaluate(instance, sol)
    end_time = time_ns()
    sol.computationTime = (end_time - start_time)
    return sol
end


# =========== 4) 结果可视化 ===========
"""
    plot_ttp_solution(instance, sol)

绘制 TTP 解的示意图：
- 绘制所有城市（散点），并用灰度表示各城市被选中的物品数量（0件 → 白色，≥10件 → 黑色）；
- 绘制路线（连线）；
- 图标题格式：地图名_items=物品总数_obj=目标值_time=运行时间（毫秒）。

同时保存图片到 results/greedy/ 下，文件名中包含标题信息。

返回：绘图对象 plt。
"""
function plot_ttp_solution(instance::TTPInstance, sol::TTPSolution)
    n = instance.numberOfNodes
    X = [instance.nodes[i, 1] for i in 1:n]
    Y = [instance.nodes[i, 2] for i in 1:n]

    route = sol.tspTour
    routeX = [X[c] for c in route]
    routeY = [Y[c] for c in route]

    items = instance.items       # 每行: (profit, weight, city)
    packingPlan = sol.packingPlan

    # 统计各城市选中物品数量
    city_item_count = zeros(Int, n)
    for i in 1:length(packingPlan)
        if packingPlan[i] == 1
            c = items[i, 3]
            city_item_count[c] += 1
        end
    end

    # 为各城市分配灰度颜色：0件→白色，≥10件→黑色
    colors = Vector{RGB}(undef, n)
    max_items = 10
    for c in 1:n
        k = min(city_item_count[c], max_items)
        gray_level = 1.0 - (k / max_items)
        colors[c] = RGB(gray_level, gray_level, gray_level)
    end

    # 构造标题字符串，计算时间以毫秒为单位
    title_str = string(instance.problemName, "_",
                       "items=", instance.numberOfItems, "_",
                       "obj=", Int64(round(sol.ob)), "_",
                       "time=", Int64(round(sol.computationTime/1e6)))

    plt = scatter(
        X, Y,
        marker = :circle,
        color = colors,
        ms = 6,
        label = "Cities",
        title = title_str,
        legend = :topright
    )
    plot!(plt,
          routeX, routeY,
          seriestype = :path,
          lw = 2,
          linecolor = :skyblue,
          label = "Route")
    # 自动保存图片到指定目录
    savefig(plt, "results/greedy/TTP_$title_str.png")
    return plt
end


# =========== 5) 测试函数 ===========
"""
    test_greedy_heuristic()

示例测试函数：
- 读取给定的 TTP 数据文件，
- 用贪心启发式求解 TTP，
- 打印评估结果和绘制解的示意图。
"""
function test_greedy_heuristic()
    # 请根据需要选择合适的数据文件
    # filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/a280_n2790_uncorr_10.ttp.txt"
    filename = "data/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/fnl4461_n22300_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/fnl4461_n44600_uncorr_10.ttp.txt"
    # filename = "data/pla33810_n33809_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/pla33810_n169045_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/pla33810_n338090_uncorr_10.ttp.txt"
    instance = TTPInstance(filename)
    @printf("\n[TEST] Loaded instance: %s\n", instance.problemName)

    sol = solve_ttp_greedy(instance)
    @printf("\n--- Greedy Solution ---\n")
    TTP.printlnSolution(sol)

    plt = plot_ttp_solution(instance, sol)
    display(plt)
end

test_greedy_heuristic()
