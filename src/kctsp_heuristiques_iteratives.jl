module KCTSPIterativeHeuristic

include("TTP.jl")  # 内含 TTPInstance/TTPSolution 等基础定义，以及 distances(...) 等方法

using .TTP
using Printf, Random
using Plots
using Base.Threads

# -----------------------------------------------------
# 0) KCTSP 目标函数: evaluate_kctsp
# -----------------------------------------------------

"""
    evaluate_kctsp(instance, route, packingPlan, K)

KCTSP 目标：
  objective = sum(profit_of_chosen_items) 
              - K * ∑_{(i->i+1)} [ d_{i,i+1} * W_i ],

其中 `W_i` 表示在离开城市 i 时背包内的重量。

返回:
  (objective, rawDistance, totalProfit, finalWeight, transportCost)
"""
function evaluate_kctsp(
    instance::TTPInstance, 
    route::Vector{Int}, 
    packingPlan::Vector{Int}, 
    K::Float64
)
    m = instance.numberOfItems
    n = length(route)
    itemsMatrix = instance.items  # 每行: (profit, weight, city)
    distMatrix = [distances(instance, i, j) for i in 1:instance.numberOfNodes, j in 1:instance.numberOfNodes]

    # 1) 先统计 totalProfit
    totalProfit = 0.0
    for i in 1:m
        if packingPlan[i] == 1
            totalProfit += itemsMatrix[i, 1]  # profit
        end
    end

    # 2) 计算运输成本 cost = K * sum(d_{c, nc} * currentWeight)
    currentWeight = 0.0
    transportCost = 0.0
    rawDistance   = 0.0

    # 依次遍历路线 (route[i], route[i+1])
    for i in 1:(n-1)
        c  = route[i]     # 当前城市
        nc = route[i+1]   # 下一个城市

        # 把城市 c 的物品(若被选中)加进背包
        for itemIdx in 1:m
            if packingPlan[itemIdx] == 1 && itemsMatrix[itemIdx, 3] == c
                currentWeight += itemsMatrix[itemIdx, 2]
            end
        end

        localDist = distMatrix[c, nc]
        rawDistance += localDist
        transportCost += localDist * currentWeight * K
    end

    localObjective = totalProfit - transportCost

    return (localObjective, rawDistance, totalProfit, currentWeight, transportCost)
end


# -----------------------------------------------------
# 1) 并行最近邻 + 并行 2-opt (TSP 部分)
# -----------------------------------------------------

"""
    euclidean_distance(nodes, i, j)

根据节点矩阵 nodes（每行为 (x,y)），按需计算城市 i 和 j 间的欧式距离。
"""
@inline function euclidean_distance(nodes, i::Int, j::Int)
    dx = nodes[i, 1] - nodes[j, 1]
    dy = nodes[i, 2] - nodes[j, 2]
    return sqrt(dx*dx + dy*dy)
end

"""
    nearest_neighbor_route(instance)

采用最近邻策略求解初始 TSP 路线。
采用多线程并行计算每一步中所有未访问城市的距离。
返回的路线为 1-based（最后回到起点）。
"""
function nearest_neighbor_route(instance::TTPInstance)
    n = instance.numberOfNodes
    visited = falses(n)
    route = [1]
    visited[1] = true
    nodes = instance.nodes

    # 依次选择下一个城市
    for _ in 2:n
        current = route[end]
        best_d = Inf
        best_city = 0
        # 为每个线程分配一个局部最优解（distance, city）
        nt = nthreads()
        local_best = [(Inf, 0) for _ in 1:nt]
        @threads for city in 1:n
            if !visited[city]
                d = euclidean_distance(nodes, current, city)
                tid = threadid()
                if d < local_best[tid][1]
                    local_best[tid] = (d, city)
                end
            end
        end
        # 归约得到全局最优解
        for (d, city) in local_best
            if d < best_d
                best_d = d
                best_city = city
            end
        end
        push!(route, best_city)
        visited[best_city] = true
    end
    push!(route, 1)  # 回到起点
    return route
end

"""
    two_opt(route, instance; max_iter)

对给定路线采用 2-opt 改进。
利用多线程平行搜索所有可能的交换，选出本轮最大的改善（即使全局目标函数降低最多的交换）。
计算时直接根据节点坐标计算距离（避免预先构造大矩阵）。
"""
function two_opt(route::Vector{Int}, instance::TTPInstance; max_iter::Int=500)
    nodes = instance.nodes
    N = length(route)

    # 计算整个路线的长度
    function route_distance(r)
        s = 0.0
        for i in 1:(length(r)-1)
            s += euclidean_distance(nodes, r[i], r[i+1])
        end
        return s
    end

    best_route = copy(route)
    best_dist = route_distance(best_route)
    iter_count = 0
    changed = true

    while changed && iter_count < max_iter
        iter_count += 1
        changed = false

        # 记录本轮最大的改善(注意：delta < 0表示改善)
        best_delta = 0.0
        best_i = 0
        best_j = 0

        # 并行遍历所有 i, j 对，i 从 2 到 N-3, j 从 i+1 到 N-1
        nt = nthreads()
        local_best = [(0.0, 0, 0) for _ in 1:nt]  # (delta, i, j)

        @threads for i in 2:(N - 2)
            tid = threadid()
            local_delta, local_i, local_j = local_best[tid]
            for j in (i+1):(N-1)
                # 当前两条边： (i-1, i) 与 (j, j+1)
                a = best_route[i-1]
                b = best_route[i]
                c = best_route[j]
                d = best_route[j+1]
                current_dist = euclidean_distance(nodes, a, b) + euclidean_distance(nodes, c, d)
                new_dist     = euclidean_distance(nodes, a, c) + euclidean_distance(nodes, b, d)
                delta = new_dist - current_dist
                if delta < local_delta
                    local_delta = delta
                    local_i = i
                    local_j = j
                end
            end
            local_best[tid] = (local_delta, local_i, local_j)
        end

        # 归约得到全局最优改善
        for (delta, i_candidate, j_candidate) in local_best
            if delta < best_delta
                best_delta = delta
                best_i = i_candidate
                best_j = j_candidate
            end
        end

        if best_delta < 0
            # 应用最优的段翻转
            best_route[best_i:best_j] = reverse(best_route[best_i:best_j])
            best_dist += best_delta
            changed = true
        end
    end

    return best_route
end

"""
    improve_tsp(instance)

综合使用并行最近邻和并行 2-opt 改进得到 TSP 解
"""
function improve_tsp(instance::TTPInstance)
    r = nearest_neighbor_route(instance)
    r2 = two_opt(r, instance; max_iter=200)  # 2-opt 改进
    return r2
end



# -----------------------------------------------------
# 2) 背包部分：模拟退火（SA）搜索
# -----------------------------------------------------

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
    sa_kctsp_knapsack(instance, route, initial_plan, K; 
                      max_iter=10000,
                      initial_temp=100.0,
                      cooling_rate=0.999,
                      multi_flip=1)

在固定路线下，通过模拟退火改进背包选择。
 - 每次随机选取 `multi_flip` 个物品翻转其选取状态，
 - 若新解满足容量则按目标变化决定是否接受，
 - 记录最优解。

返回：改进后的 packingPlan
"""
function sa_kctsp_knapsack(
    instance::TTPInstance,
    route::Vector{Int},
    initial_plan::Vector{Int},
    K::Float64;
    max_iter::Int=10000,
    initial_temp::Float64=100.0,
    cooling_rate::Float64=0.999,
    multi_flip::Int=1
)
    m = instance.numberOfItems
    capacity = instance.capacityOfKnapsack

    # 评估函数(直接用 evaluate_kctsp)
    function evaluate_plan(plan::Vector{Int})
        (obj, _, _, w, _) = evaluate_kctsp(instance, route, plan, K)
        return obj, w
    end

    current_plan = copy(initial_plan)
    current_obj, current_w = evaluate_plan(current_plan)
    best_plan = copy(current_plan)
    best_obj = current_obj

    temp = initial_temp
    rng = MersenneTwister()

    for iter in 1:max_iter
        candidate = copy(current_plan)
        # 同时翻转 multi_flip 个物品
        flip_indices = rand(rng, 1:m, multi_flip)
        for fi in flip_indices
            candidate[fi] = 1 - candidate[fi]
        end

        candidate_obj, candidate_w = evaluate_plan(candidate)

        # 若背包超限，跳过
        if candidate_w > capacity
            continue
        end

        delta = candidate_obj - current_obj

        # 改善必接受；劣化以一定概率接受
        if delta > 0 || exp(delta / temp) > rand(rng)
            current_plan = candidate
            current_obj = candidate_obj
            current_w = candidate_w
        end

        # 更新全局最好
        if current_obj > best_obj
            best_obj = current_obj
            best_plan = copy(current_plan)
            @printf("SA_KC Iter %d: Best_obj=%.2f\n", iter, best_obj)
        end

        temp *= cooling_rate

        # 可自行设定终止条件，如若温度过低或若长时间无改进
        if temp < 1e-4
            break
        end
    end

    return best_plan
end


# -----------------------------------------------------
# 3) 整合：并行 TSP + SA 背包 + 评估
# -----------------------------------------------------

"""
    solve_kctsp_enhanced(instance, K; 
                         tsp_max_iter=200,
                         sa_max_iter=10000,
                         initial_temp=100.0,
                         cooling_rate=0.999,
                         multi_flip=1)

1) 并行最近邻生成初始路线 -> 并行 2-opt 改进
2) 初始背包可先采用“净收益”贪心或全0
3) 用模拟退火改进背包方案
4) 返回 TTPSolution
"""
function solve_kctsp_enhanced(
    instance::TTPInstance, 
    K::Float64;
    tsp_max_iter::Int=200,
    sa_max_iter::Int=10000,
    initial_temp::Float64=100.0,
    cooling_rate::Float64=0.999,
    multi_flip::Int=1
)

    start_time = time_ns()

    # 1) 路线：并行最近邻 + 并行 2-opt
    route0 = nearest_neighbor_route(instance)
    route = two_opt(route0, instance; max_iter=tsp_max_iter)

    # 2) 背包初始方案（示例：先用净收益贪心）
    packing_init = greedy_kctsp_knapsack(instance, route, K)

    # 3) 模拟退火改进背包
    packing_sa = sa_kctsp_knapsack(
        instance, 
        route,
        packing_init,
        K;
        max_iter=sa_max_iter,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        multi_flip=multi_flip
    )

    # 4) 构造最终解并评估
    (obj, rawDist, tp, finalW, transCost) = evaluate_kctsp(instance, route, packing_sa, K)
    sol = TTPSolution(
        route,
        packing_sa;
        fp      = tp,      # 总利润
        ft      = 0.0,     # (KCTSP中没用时间)
        ftraw   = Int(round(rawDist)),
        ob      = obj,
        wend    = instance.capacityOfKnapsack - finalW,
        wendUsed= finalW,
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
    savefig(plt, "results/KCTSP_$title_str.png")
    return plt
end

# -----------------------------------------------------
# 4) 测试：示例函数
# -----------------------------------------------------

"""
    test_kctsp_enhanced()
示例入口函数：对指定文件(与 TTP 相同格式)执行并行 TSP + SA 背包的KCTSP解法
"""
function test_kctsp_enhanced()
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
    K = 0.01

    @info "Running parallel TSP + SA-KP on KCTSP, problem = $(instance.problemName)"

    sol = solve_kctsp_enhanced(instance, K; 
                               tsp_max_iter=1000,
                               sa_max_iter=10000, 
                               initial_temp=100.0,
                               cooling_rate=0.999,
                               multi_flip=2)

    @info "Solution =>"
    TTP.printlnSolution(sol)

    plt = plot_kctsp_solution(instance, sol)
    display(plt)
end

# 自动执行测试
test_kctsp_enhanced()

end # module
