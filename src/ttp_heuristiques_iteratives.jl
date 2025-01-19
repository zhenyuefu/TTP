include("TTP.jl")

using .TTP
using Printf, Random, Plots
using Base.Threads

# -------------------------------
# 1) 并行版 最近邻 + 2-opt 改进 TSP
# -------------------------------

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

# --------------------------------------------
# 2) 考虑速度减缓的"单位时间收益" 贪心选物品
# --------------------------------------------

"""
    approximate_item_time(instance, route, city, itemWeight; wc)

估算在城市 city 拿 itemWeight 后对后续路程增加的 "额外时间"。
TTP 的速度模型：
  time = sum(distance / (vmax - currentWeight*(vmax-vmin)/W))
这里做个简化计算，假设目前物品重量 wc≈0 或者采用平均值。
"""
function approximate_item_time(instance::TTPInstance, route::Vector{Int}, city::Int, itemWeight::Int; wc::Float64=0.0)
    W   = instance.capacityOfKnapsack
    fac = (instance.maxSpeed - instance.minSpeed) / W

    # 找出城市在路线中的位置
    pos = findfirst(==(city), route)
    if pos === nothing
        return 0.0
    end

    nodes = instance.nodes
    extra_time = 0.0
    for i in pos:(length(route)-1)
        c1 = route[i]
        c2 = route[i+1]
        d = euclidean_distance(nodes, c1, c2)
        old_speed = instance.maxSpeed - wc * fac
        new_speed = instance.maxSpeed - (wc + itemWeight) * fac
        if old_speed < 1e-6; old_speed = 1e-6; end
        if new_speed < 1e-6; new_speed = 1e-6; end
        extra_time += (d / new_speed - d / old_speed)
    end
    return extra_time
end

"""
    greedy_knapsack_with_time(instance, route)

基于“单位时间收益”（profit / approximate extra time）贪心选取物品。
过滤掉起点与终点（城市1和城市n）的物品。
"""
function greedy_knapsack_with_time(instance::TTPInstance, route::Vector{Int})
    itemsMatrix = instance.items
    m = instance.numberOfItems
    W = instance.capacityOfKnapsack

    item_list = []
    for i in 1:m
        city = itemsMatrix[i, 3]
        if city == 1 || city == instance.numberOfNodes
            continue
        end
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]
        dtime = approximate_item_time(instance, route, city, weight; wc=0.0)
        if dtime < 1e-9
            dtime = 1e-9
        end
        ratio = profit / dtime
        push!(item_list, (i, profit, weight, ratio))
    end

    sort!(item_list, by = x -> x[4], rev = true)
    chosen = fill(0, m)
    remainCap = W
    for (idx, p, w, ratio) in item_list
        if w <= remainCap && ratio > 0
            chosen[idx] = 1
            remainCap -= w
        end
    end
    return chosen
end

# ---------------------------------------
# 3) 改进版“迭代”背包求解：模拟退火算法
# ---------------------------------------

"""
    sa_knapsack_with_time(instance, route, initial_plan; max_iter, initial_temp, cooling_rate)

在给定固定路线 route 及初始背包方案 initial_plan 的基础上，
采用 **模拟退火算法** 来搜索 0/1 背包的更优解。
具体过程：
  - 以一定的初始温度 initial_temp 开始，
  - 每次随机选取一个物品（或多个）翻转其选取状态，
  - 若候选解满足容量约束，则接受改善的解；如果效果变差，也有一定概率接受（概率随温度降低而降低），
  - 温度按 cooling_rate 指数衰减，
  - 保留搜索过程中找到的最佳方案。
  
该方法适合大规模数据下的背包求解。
"""

function sa_knapsack_with_time(instance::TTPInstance, route::Vector{Int}, initial_plan::Vector{Int};
    max_iter::Int=10000,
    initial_temp::Float64=100.0,
    cooling_rate::Float64=0.999,
    maxFlip::Int=3
)  # 新增加参数，表示最多同时翻转的物品数量

    m = instance.numberOfItems
    # 内部函数：给定背包方案计算目标函数值
    function evaluate_plan(plan::Vector{Int})
        sol_temp = TTPSolution(route, plan;
            fp=-Inf, ft=Inf, ftraw=typemax(Int),
            ob=-Inf, wend=Inf, wendUsed=Inf,
            computationTime=0)
        evaluate(instance, sol_temp)
        return sol_temp
    end

    current_plan = copy(initial_plan)
    current_sol = evaluate_plan(current_plan)
    current_obj = current_sol.ob
    best_plan = copy(current_plan)
    best_obj = current_obj
    temp = initial_temp

    for iter in 1:max_iter
        candidate = copy(current_plan)
        # 随机决定本次翻转的物品数量（1到 maxFlip 之间）
        num_flip = rand(1:maxFlip)
        # 随机选取 num_flip 个物品的位置进行翻转
        flip_indices = rand(1:m, num_flip)
        for i in flip_indices
            candidate[i] = 1 - candidate[i]
        end

        sol_candidate = evaluate_plan(candidate)
        # 若违反容量约束，直接跳过
        if sol_candidate.wend < 0
            continue
        end

        candidate_obj = sol_candidate.ob
        delta = candidate_obj - current_obj

        # 改善时总是接受，否则以一定概率接受劣化解
        if delta > 0 || exp(delta / temp) > rand()
            current_plan = candidate
            current_obj = candidate_obj
            current_sol = sol_candidate
        end

        if current_obj > best_obj
            best_plan = copy(current_plan)
            best_obj = current_obj
            @printf("SA Iter %d: Improved objective = %.2f\n", iter, best_obj)
        end

        temp *= cooling_rate

        # 温度太低时，也可重置温度（reheating），或者提前终止
        if temp < 1e-3
            temp = initial_temp  # 重置温度，尝试跳出局部停滞
        end
    end

    return best_plan
end

# ----------------------------------
# 4) 综合求解 + 可视化
# ----------------------------------

"""
    solve_ttp_enhanced(instance)

求解步骤：
  1. 利用并行 TSP 算法（最近邻 + 2-opt）得到路线；
  2. 贪心方法初始选取背包；
  3. 利用模拟退火算法改进背包解；
  4. 构造 TTPSolution 并评估目标函数值。
"""
function solve_ttp_enhanced(instance::TTPInstance)

    start_time = time_ns()

    # 1) TSP 路线求解
    route0 = nearest_neighbor_route(instance)
    route = two_opt(route0, instance; max_iter=1000)

    # 2) 初始背包解（贪心策略）
    initial_plan = greedy_knapsack_with_time(instance, route)

    # 3) 模拟退火改进背包：适合大规模数据
    improved_plan = sa_knapsack_with_time(instance, route, initial_plan;
                                          max_iter=100000,
                                          initial_temp=100.0,
                                          cooling_rate=0.999)
    
    # 4) 构造解并评估
    sol = TTPSolution(route, improved_plan;
                      fp=-Inf, ft=Inf, ftraw=typemax(Int),
                      ob=-Inf, wend=Inf, wendUsed=Inf,
                      computationTime=0)
    
    evaluate(instance, sol)
    end_time = time_ns()
    sol.computationTime = (end_time - start_time)
    return sol
end

"""
    plot_ttp_solution(instance, sol)

绘制 TTP 解的示意图：展示城市位置、路线及各城市选取物品数量（灰度表示）。
标题格式为：地图名_每个城市的物品数量_obj_运行时间
"""
function plot_ttp_solution(instance::TTPInstance, sol::TTPSolution)
    n = instance.numberOfNodes
    X = [instance.nodes[i,1] for i in 1:n]
    Y = [instance.nodes[i,2] for i in 1:n]

    route = sol.tspTour
    routeX = [X[c] for c in route]
    routeY = [Y[c] for c in route]

    items = instance.items       # 每行: (profit, weight, city)
    packingPlan = sol.packingPlan

    # 统计每个城市选取的物品数量
    city_item_count = zeros(Int, n)
    for i in 1:length(packingPlan)
        if packingPlan[i] == 1
            city = items[i,3]
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
                       "time=", Int64(round(sol.computationTime/1000000)))

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
    savefig(plt, "results/iterative/TTP_$title_str.png")
    return plt
end


# -------------------------------
# 5) 测试
# -------------------------------

function test_enhanced()
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
    @info "Loaded instance: $(instance.problemName)"

    sol = solve_ttp_enhanced(instance)
    @info "Solution => "
    TTP.printlnSolution(sol)

    plt = plot_ttp_solution(instance, sol)
    display(plt)
end

test_enhanced()
