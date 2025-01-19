module KCTSPIterativeHeuristic

include("TTP.jl")

using .TTP
using Printf, Random
using Plots
using Base.Threads

# -----------------------------------------------------
# 0) KCTSP 目标函数: evaluate_kctsp（支持 distMatrix 外部传入）
# -----------------------------------------------------

"""
    evaluate_kctsp(instance, route, packingPlan, K, distMatrix)

KCTSP 目标：
  objective = sum(profit_of_chosen_items) 
              - K * ∑( d_{(i->i+1)} * W_i ),
其中 `W_i` 是离开城市 i 时的背包重量。

参数 `distMatrix` 为预先构造好的城市距离矩阵 (n×n)，
避免在此函数内部重复构造。

返回:
  (objective, rawDistance, totalProfit, finalWeight, transportCost)
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

    # 1) 累计物品收益
    totalProfit = 0.0
    for i in 1:m
        if packingPlan[i] == 1
            totalProfit += itemsMatrix[i, 1]
        end
    end

    # 2) 运输代价计算
    currentWeight = 0.0
    transportCost = 0.0
    rawDistance   = 0.0

    for i in 1:(n-1)
        c  = route[i]
        nc = route[i+1]

        # 将城市 c 中选中的物品加入背包
        for itemIdx in 1:m
            if packingPlan[itemIdx] == 1 && itemsMatrix[itemIdx, 3] == c
                currentWeight += itemsMatrix[itemIdx, 2]
            end
        end

        localDist = distMatrix[c, nc]
        rawDistance   += localDist
        transportCost += localDist * currentWeight * K
    end

    localObjective = totalProfit - transportCost
    return (localObjective, rawDistance, totalProfit, currentWeight, transportCost)
end


# -----------------------------------------------------
# 1) 并行最近邻 + 并行 2-opt (TSP 部分)
# -----------------------------------------------------

@inline function euclidean_distance(nodes, i::Int, j::Int)
    dx = nodes[i, 1] - nodes[j, 1]
    dy = nodes[i, 2] - nodes[j, 2]
    return sqrt(dx*dx + dy*dy)
end

function nearest_neighbor_route(instance::TTPInstance)
    n = instance.numberOfNodes
    visited = falses(n)
    route = [1]
    visited[1] = true
    nodes = instance.nodes

    for _ in 2:n
        current = route[end]
        best_d = Inf
        best_city = 0
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

        for (d, city) in local_best
            if d < best_d
                best_d = d
                best_city = city
            end
        end
        push!(route, best_city)
        visited[best_city] = true
    end
    push!(route, 1)
    return route
end

function two_opt(route::Vector{Int}, instance::TTPInstance; max_iter::Int=500)
    nodes = instance.nodes
    N = length(route)

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

        best_delta = 0.0
        best_i = 0
        best_j = 0

        nt = nthreads()
        local_best = [(0.0, 0, 0) for _ in 1:nt]

        @threads for i in 2:(N - 2)
            tid = threadid()
            local_delta, local_i, local_j = local_best[tid]
            for j in (i+1):(N-1)
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

        for (delta, i_candidate, j_candidate) in local_best
            if delta < best_delta
                best_delta = delta
                best_i = i_candidate
                best_j = j_candidate
            end
        end

        if best_delta < 0
            best_route[best_i:best_j] = reverse(best_route[best_i:best_j])
            best_dist += best_delta
            changed = true
        end
    end
    return best_route
end

function improve_tsp(instance::TTPInstance)
    r = nearest_neighbor_route(instance)
    r2 = two_opt(r, instance; max_iter=200)
    return r2
end


# -----------------------------------------------------
# 2) 背包部分：贪心构造初始背包方案 + 模拟退火（SA）搜索
#     同时修正重复构造距离的计算
# -----------------------------------------------------

"""
    compute_dist_carried(distMatrix, route) -> dist_after

给定路线 route (1-based)，利用预先构造好的 distMatrix 计算
每个城市 c 从其在 route 中的位置到路线末尾的累计距离 dist_after[c]。
"""
function compute_dist_carried(distMatrix::Matrix{Float64}, route::Vector{Int})
    n = length(route)
    nNodes = size(distMatrix, 1)

    dist_after = fill(0.0, nNodes)
    routeDistance = Float64[]
    for i in 1:(n-1)
        push!(routeDistance, distMatrix[ route[i], route[i+1] ])
    end

    position_in_route = fill(-1, nNodes)
    for i in 1:n
        c = route[i]
        position_in_route[c] = i
    end

    totalDistFrom = fill(0.0, n)
    for i in (n-1):-1:1
        totalDistFrom[i] = totalDistFrom[i+1] + routeDistance[i]
    end

    for c in 1:nNodes
        idx = position_in_route[c]
        dist_after[c] = (idx < n) ? totalDistFrom[idx] : 0.0
    end
    return dist_after
end

"""
    greedy_kctsp_knapsack(instance, route, K, distMatrix)

在给定路线下，依据“净收益”构造初始背包方案：
  对每个物品计算：netGain = profit - K * weight * distCarried，
其中 distCarried 为从该物品所在城市到终点的累计距离，
贪心选取收益为正且重量合适的物品。
"""
function greedy_kctsp_knapsack(instance::TTPInstance, route::Vector{Int}, K::Float64,
                               distMatrix::Matrix{Float64})
    m = instance.numberOfItems
    W = instance.capacityOfKnapsack
    itemsMatrix = instance.items

    # 计算每个城市后续累计距离
    dist_after = compute_dist_carried(distMatrix, route)

    item_net_gain = Vector{Tuple{Int,Float64}}(undef, m)
    for i in 1:m
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]
        city   = itemsMatrix[i, 3]
        carriedDist = dist_after[city]
        netGain = profit - K * weight * carriedDist
        item_net_gain[i] = (i, netGain)
    end

    sort!(item_net_gain, by = x -> x[2], rev = true)

    packingPlan = fill(0, m)
    remainCap = W
    for (idx, gain) in item_net_gain
        weight = itemsMatrix[idx, 2]
        if gain <= 0
            continue
        end
        if weight <= remainCap
            packingPlan[idx] = 1
            remainCap -= weight
        end
    end

    return packingPlan
end

"""
    sa_kctsp_knapsack(instance, route, initial_plan, K, distMatrix;
                        max_iter, initial_temp, cooling_rate, maxFlip)

在固定路线及 distMatrix 下，利用模拟退火算法优化背包方案：
  - 从贪心方案 initial_plan 开始；
  - 每次随机翻转 1~maxFlip 个物品的选取状态；
  - 若候选方案满足重量约束，则根据目标函数改善接受候选解，且在退火时允许劣化解接受；
  - 当温度降至很低时执行“重热”操作，防止陷入局部最优。
"""
function sa_kctsp_knapsack(
    instance::TTPInstance,
    route::Vector{Int},
    initial_plan::Vector{Int},
    K::Float64,
    distMatrix::Matrix{Float64};
    max_iter::Int=10000,
    initial_temp::Float64=100.0,
    cooling_rate::Float64=0.999,
    maxFlip::Int=3
)
    m = instance.numberOfItems
    capacity = instance.capacityOfKnapsack

    # 内部函数：给定方案计算 (目标, 当前重量)
    function evaluate_plan(plan::Vector{Int})
        (obj, _, _, w, _) = evaluate_kctsp(instance, route, plan, K, distMatrix)
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
        # 随机决定翻转 1 到 maxFlip 个物品
        num_flip = rand(rng, 1:maxFlip)
        flip_indices = rand(rng, 1:m, num_flip)
        for fi in flip_indices
            candidate[fi] = 1 - candidate[fi]
        end

        candidate_obj, candidate_w = evaluate_plan(candidate)
        # 若违反容量约束，直接跳过
        if candidate_w > capacity
            continue
        end

        delta = candidate_obj - current_obj
        # 改善时总是接受，否则以一定概率接受劣化解
        if delta > 0 || exp(delta / temp) > rand(rng)
            current_plan = candidate
            current_obj = candidate_obj
            current_w = candidate_w
        end

        if current_obj > best_obj
            best_obj = current_obj
            best_plan = copy(current_plan)
            @printf("SA_KC Iter %d: Improved objective = %.2f\n", iter, best_obj)
        end

        temp *= cooling_rate

        # 当温度过低时执行重热，防止局部停滞
        if temp < 1e-3
            temp = initial_temp
        end
    end

    return best_plan
end


# -----------------------------------------------------
# 3) 整合：并行 TSP + SA 背包 + 评估 (只构造一次 distMatrix)
# -----------------------------------------------------

"""
    solve_kctsp_enhanced(instance, K; tsp_max_iter, sa_max_iter, initial_temp, cooling_rate, maxFlip)

求解步骤：
  1. 利用并行 TSP 算法（最近邻 + 2-opt）得到路线；
  2. 构造 distMatrix（只构造一次）；
  3. 用净收益贪心方法初始构造背包方案；
  4. 用模拟退火（SA）改进背包方案；
  5. 调用 evaluate_kctsp 评估目标；
  6. 构造 TTPSolution 并返回。
"""
function solve_kctsp_enhanced(
    instance::TTPInstance, 
    K::Float64;
    tsp_max_iter::Int=200,
    sa_max_iter::Int=10000,
    initial_temp::Float64=100.0,
    cooling_rate::Float64=0.999,
    maxFlip::Int=1
)
    start_time = time_ns()

    # 1) TSP：并行最近邻 + 2-opt
    route0 = nearest_neighbor_route(instance)
    route  = two_opt(route0, instance; max_iter=tsp_max_iter)

    # 2) 构造 distMatrix（只构造一次）
    nNodes = instance.numberOfNodes
    distMatrix = [distances(instance, i, j) for i in 1:nNodes, j in 1:nNodes]

    # 3) 贪心背包初始方案（净收益贪心）
    packing_init = greedy_kctsp_knapsack(instance, route, K, distMatrix)

    # 4) 用 SA 改进背包方案
    packing_sa = sa_kctsp_knapsack(
        instance, route, packing_init, K, distMatrix;
        max_iter=sa_max_iter,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        maxFlip=maxFlip
    )

    # 5) 评估
    (obj, rawDist, tp, finalW, transCost) = evaluate_kctsp(
        instance, route, packing_sa, K, distMatrix
    )

    # 6) 构造 TTPSolution
    sol = TTPSolution(
        route, packing_sa;
        fp      = tp,
        ft      = 0.0,
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

绘制 KCTSP 解的示意图，包括城市位置、路线以及各城市选取物品数量（灰度表示）。
"""
function plot_kctsp_solution(instance::TTPInstance, sol::TTPSolution)
    cities = instance.nodes
    route  = sol.tspTour

    xs = [cities[route[i], 1] for i in 1:length(route)]
    ys = [cities[route[i], 2] for i in 1:length(route)]

    items = instance.items
    packing = sol.packingPlan
    city_item_count = zeros(Int, instance.numberOfNodes)
    for i in 1:length(packing)
        if packing[i] == 1
            c = items[i, 3]
            city_item_count[c] += 1
        end
    end

    colors = Vector{RGB}(undef, instance.numberOfNodes)
    max_items = 10
    for c in 1:instance.numberOfNodes
        k = min(city_item_count[c], max_items)
        g = 1.0 - (k / max_items)
        colors[c] = RGB(g, g, g)
    end

    title_str = string(instance.problemName, "_items=", instance.numberOfItems,
                       "_obj=", Int64(round(sol.ob)), "_time=",
                       Int64(round(sol.computationTime/1e6)))
    cityX = [cities[i,1] for i in 1:instance.numberOfNodes]
    cityY = [cities[i,2] for i in 1:instance.numberOfNodes]

    plt = scatter(cityX, cityY, marker=:circle, color=colors, ms=5,
                  title=title_str, label="Cities")
    plot!(plt, xs, ys, seriestype=:path, linecolor=:blue, label="Route")
    savefig(plt, "results/iterative/KCTSP_$title_str.png")
    return plt
end

"""
    test_kctsp_enhanced()

示例：对指定 TTP 实例执行并行 TSP + SA 背包 (KCTSP) 求解，
注意构造 distMatrix 仅一次。
"""
function test_kctsp_enhanced()
    # 可选择不同数据集：
    # filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/a280_n2790_uncorr_10.ttp.txt"
    # filename = "data/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/fnl4461_n22300_uncorr-similar-weights_05.ttp.txt"
    filename = "data/fnl4461_n44600_uncorr_10.ttp.txt"
    # filename = "data/pla33810_n33809_bounded-strongly-corr_01.ttp.txt"
    # filename = "data/pla33810_n169045_uncorr-similar-weights_05.ttp.txt"
    # filename = "data/pla33810_n338090_uncorr_10.ttp.txt"
    instance = TTPInstance(filename)
    # 根据实例信息计算 K 值
    K = instance.rentingRatio / instance.maxSpeed / instance.capacityOfKnapsack

    @info "Running parallel TSP + SA-KC on KCTSP, problem = $(instance.problemName)"

    sol = solve_kctsp_enhanced(instance, K; 
                               tsp_max_iter=1000,
                               sa_max_iter=10000, 
                               initial_temp=100.0,
                               cooling_rate=0.999,
                               maxFlip=2)

    @info "Solution =>"
    TTP.printlnSolution(sol)

    plt = plot_kctsp_solution(instance, sol)
    display(plt)
end

# 自动执行测试
test_kctsp_enhanced()

end # module
