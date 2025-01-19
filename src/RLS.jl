include("TTP.jl")
using .TTP
include("utils.jl")
using .Utils
using Plots
function randomLocalSearch(instance::TTPInstance, route::Vector{Int}; maxNoImprovement::Int=50)::TTPSolution
    """
    在给定 TSP 路线 route 的前提下，执行随机局部搜索(RLS) 来寻找背包的最优选择方案。
    - 初始解：不选任何物品
    - 每次迭代：随机翻转一个物品的选取状态，若可行且目标值不降低，则接受；否则回退。
    - 连续 maxNoImprovement 次无改进则停止。

    返回: TTPSolution
"""
    # 1) 创建初始解：空背包
    bestSol = TTPSolution(
        copy(route),
        zeros(Int, instance.numberOfItems),
    )
    # 先评估空背包解
    evaluate(instance, bestSol)
    bestObj = bestSol.ob

    # 2) 开始迭代
    noImprovementCount = 0
    iteration = 0

    while noImprovementCount < maxNoImprovement
        iteration += 1
        # (a) 在当前bestSol基础上随机翻转一个物品
        candidateSol = TTPSolution(
            copy(route),
            copy(bestSol.packingPlan)
        )
        # 随机选一个物品索引 [1..m]
        idx = rand(1:instance.numberOfItems)
        # 翻转 0->1 or 1->0
        candidateSol.packingPlan[idx] = 1 - candidateSol.packingPlan[idx]

        # (b) 检查背包容量是否超限
        #     若不超则 evaluate；若超则直接跳过（记为无改进1次）
        totalWeight = 0
        for i in 1:instance.numberOfItems
            if candidateSol.packingPlan[i] == 1
                totalWeight += instance.items[i, 2]
            end
        end
        if totalWeight <= instance.capacityOfKnapsack
            # 计算目标值
            evaluate(instance, candidateSol)

            # (c) 若候选解不比当前best差，则接受成为新best
            if candidateSol.ob >= bestObj
                bestSol = candidateSol
                bestObj = candidateSol.ob
                noImprovementCount = 0
            else
                noImprovementCount += 1
            end
        else
            noImprovementCount += 1
        end
    end

    # 返回找到的最佳解
    return bestSol
end


function plot_ttp_solution(instance::TTPInstance, sol::TTPSolution)
    n = instance.numberOfNodes
    X = [instance.nodes[i, 1] for i in 1:n]
    Y = [instance.nodes[i, 2] for i in 1:n]

    route = sol.tspTour
    routeX = [X[c] for c in route]
    routeY = [Y[c] for c in route]

    items = instance.items       # 每行: (profit, weight, city)
    packingPlan = sol.packingPlan

    # 统计每个城市选取的物品数量
    city_item_count = zeros(Int, n)
    for i in 1:length(packingPlan)
        if packingPlan[i] == 1
            city = items[i, 3]
            city_item_count[city] += 1
        end
    end

    # 为每个城市分配灰度颜色（0件 -> 白色，10件或以上 -> 黑色）
    colors = Vector{RGB}(undef, n)
    max_items = 10
    for c in 1:n
        k = min(city_item_count[c], max_items)
        gray_level = 1.0 - (k / max_items)
        colors[c] = RGB(gray_level, gray_level, gray_level)
    end

    # 构造标题字符串：
    # 例如： "地图名_每个城市的物品数量_目标函数值_运行时间"
    # 注意：city_item_count 数组较长时可能会占用较多空间，可根据需要调整显示格式
    title_str = string(instance.problemName, "_",
        "items=", instance.numberOfItems, "_",
        "obj=", Int64(round(sol.ob)), "_",
        "time=", sol.computationTime)

    # 绘图
    plt = scatter(
        X, Y,
        marker=:circle,
        color=colors,
        ms=6,
        label="Cities",
        title=title_str,
        legend=:topright
    )
    plot!(plt,
        routeX, routeY,
        seriestype=:path,
        lw=2,
        linecolor=:skyblue,
        label="Route")
    savefig(plt, "results/rls/TTP_$title_str.png")
    return plt
end

for filename in [
    "data/a280_n279_bounded-strongly-corr_01.ttp.txt",
    "data/a280_n1395_uncorr-similar-weights_05.ttp.txt",
    "data/a280_n2790_uncorr_10.ttp.txt",
    "data/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt",
    "data/fnl4461_n22300_uncorr-similar-weights_05.ttp.txt",
    "data/fnl4461_n44600_uncorr_10.ttp.txt",
    "data/pla33810_n33809_bounded-strongly-corr_01.ttp.txt",
    "data/pla33810_n169045_uncorr-similar-weights_05.ttp.txt",
    "data/pla33810_n338090_uncorr_10.ttp.txt"
]

    instance = TTPInstance(filename)
    start_time = Utils.start_timing()
    route = lkh(instance)
    sol = randomLocalSearch(instance, route)
    elapsed = Utils.stop_timing(start_time)
    sol.computationTime = elapsed

    printFullSolution(sol)

    plt = plot_ttp_solution(instance, sol)

end