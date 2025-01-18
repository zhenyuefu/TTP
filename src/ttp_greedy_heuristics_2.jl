include("TTP.jl")  

using .TTP
using Printf, Random, Plots

# -------------------------------
# 1) 最近邻 + 2-opt 改进 TSP
# -------------------------------

"""
    nearest_neighbor_route(instance)

先用最近邻策略得到初始路线(1-based)。
"""
function nearest_neighbor_route(instance::TTPInstance)
    n = instance.numberOfNodes
    dist = [distances(instance, i, j) for i in 1:n, j in 1:n]

    visited = falses(n)
    visited[1] = true
    route = [1]
    current_city = 1

    for _ in 1:(n-1)
        next_city = 0
        min_d = Inf
        for c in 1:n
            if !visited[c] && dist[current_city,c]<min_d
                min_d = dist[current_city,c]
                next_city = c
            end
        end
        push!(route, next_city)
        visited[next_city] = true
        current_city = next_city
    end
    push!(route, 1)  # 回到城市 1
    return route
end

"""
    two_opt(route, instance)

对给定路线做 2-opt 改进 (局部搜索)，在有限轮次内尝试翻转两段来减少总距离。
返回 改进后的路线(1-based)。
"""
function two_opt(route::Vector{Int}, instance::TTPInstance; max_iter::Int=500)
    dist = [distances(instance, i, j) for i in 1:instance.numberOfNodes, j in 1:instance.numberOfNodes]
    
    function route_distance(r)
        s = 0.0
        for i in 1:(length(r)-1)
            s += dist[r[i], r[i+1]]
        end
        return s
    end

    best_route = copy(route)
    best_dist = route_distance(best_route)
    improved = true
    iter_count = 0

    while improved && iter_count<max_iter
        improved = false
        iter_count += 1
        for i in 2:(length(best_route)-2)
            for j in (i+1):(length(best_route)-1)
                # 翻转 route[i:j]
                if j - i == 1
                    continue
                end
                new_route = vcat(best_route[1:i-1], reverse(best_route[i:j]), best_route[j+1:end])
                new_dist = 0.0
                # 只需比较翻转后局部变动也可，但这里直接算总长
                new_dist = route_distance(new_route)
                if new_dist < best_dist
                    best_dist = new_dist
                    best_route = new_route
                    improved = true
                    break
                end
            end
            if improved
                break
            end
        end
    end

    return best_route
end

"""
    improve_tsp(instance)

综合: 最近邻 -> 2-opt 改进
"""
function improve_tsp(instance::TTPInstance)
    r = nearest_neighbor_route(instance)
    r2 = two_opt(r, instance; max_iter=200)  # 2-opt
    return r2
end


# --------------------------------------------
# 2) 考虑速度减缓的"单位时间收益" 贪心选物品
# --------------------------------------------

"""
    approximate_item_time(instance, itemWeight, city, route, vmin, vmax)

估算在城市 city 拿 itemWeight 后，对后续路程增加的 "额外时间"。
TTP 时间公式:
  time = sum( distance / [vmax - currentWeight*(vmax-vmin)/W ] )
这里简单做个近似: 
  - 找到 route 中 city 出现的位置 pos
  - 对 [pos, end-1] 的所有边, 计算差值:
      d / [vmax - (wc+itemWeight)*fac] - d / [vmax - wc*fac]
  其中 wc 是假设性的已有重量(此处若要精确, 需先估算城市前面物品的重量).
此示例仅做简化: 假设 "当前wc" ≈ 0 (或一个平均值), 主要突出 itemWeight 对后续速度的影响.
"""
function approximate_item_time(instance::TTPInstance, route::Vector{Int}, city::Int, itemWeight::Int; 
                               wc::Float64=0.0)
    W   = instance.capacityOfKnapsack
    fac = (instance.maxSpeed - instance.minSpeed)/W

    # 找 city 在 route 中的下标 pos
    pos = findfirst(==(city), route)
    if pos == nothing
        # 说明路线没包含这个city？(不太可能,除非 city=1 or error)
        return 0.0
    end

    distmat = [distances(instance, i, j) for i in 1:instance.numberOfNodes, j in 1:instance.numberOfNodes]
    extra_time = 0.0
    for i in pos:(length(route)-1)
        c1 = route[i]
        c2 = route[i+1]
        d  = distmat[c1,c2]

        old_speed  = instance.maxSpeed - wc*fac
        new_speed  = instance.maxSpeed - (wc+itemWeight)*fac

        if old_speed<1e-6 
            old_speed = 1e-6
        end
        if new_speed<1e-6
            new_speed = 1e-6
        end

        old_t = d/old_speed
        new_t = d/new_speed
        extra_time += (new_t - old_t)
    end
    return extra_time
end

"""
    greedy_knapsack_with_time(instance, route)

基于 "单位时间收益" = profit / 近似(额外时间) 进行贪心选取:
1) 对每个物品 i, 计算 extra_time_i ~ approximate_item_time(...)
2) ratio_i = profit_i / max(extra_time_i, ε)
3) 按 ratio_i 降序排序，若 ratio>0, 且不超容量则选.
"""
function greedy_knapsack_with_time(instance::TTPInstance, route::Vector{Int})
    itemsMatrix = instance.items
    m = instance.numberOfItems
    W = instance.capacityOfKnapsack

    vmin = instance.minSpeed
    vmax = instance.maxSpeed

    # 过滤起点/终点
    # city=1 or city=n 不拿
    n = instance.numberOfNodes

    item_list = []
    for i in 1:m
        city = itemsMatrix[i, 3]
        if city==1 || city==n
            continue
        end
        profit = itemsMatrix[i, 1]
        weight = itemsMatrix[i, 2]

        # 近似额外时间(这里默认wc=0, 也可用个估值)
        dtime = approximate_item_time(instance, route, city, weight; wc=0.0)
        if dtime<1e-9
            # 如果dtime很小(城市接近终点), 避免除0
            dtime=1e-9
        end
        ratio = profit / dtime
        push!(item_list, (i, profit, weight, ratio))
    end

    # 按 ratio 降序
    sort!(item_list, by=x->x[4], rev=true)

    chosen = fill(0, m)
    remainCap = W

    for (idx, p, w, ratio) in item_list
        if w<=remainCap && ratio>0
            chosen[idx] = 1
            remainCap -= w
        end
    end
    return chosen
end


# ----------------------------------
# 3) 综合求解 + 可视化
# ----------------------------------

"""
    solve_ttp_enhanced(instance)

1) 先用 nearest_neighbor + 2-opt 改进路线
2) 用 "单位时间收益" 的贪心选物品
3) evaluate
"""
function solve_ttp_enhanced(instance::TTPInstance)
    # 1) 路线
    route0 = nearest_neighbor_route(instance)
    route = two_opt(route0, instance; max_iter=200)

    # 2) 背包
    packingPlan = greedy_knapsack_with_time(instance, route)

    # 3) 构造TTPSolution并评估
    sol = TTPSolution(route,
                      packingPlan;
                      fp=-Inf,
                      ft=Inf,
                      ftraw=typemax(Int),
                      ob=-Inf,
                      wend=Inf,
                      wendUsed=Inf,
                      computationTime=0)

    start_time = time_ns()
    evaluate(instance, sol)
    end_time = time_ns()
    sol.computationTime = (end_time - start_time)

    return sol
end

"""
    plot_ttp_solution(instance, sol)

可视化:
1) 散点显示所有城市 (X,Y)
2) 路线连线
3) 在城市有拿物品时(若你想突出显示), 可标注一下
"""
# function plot_ttp_solution(instance::TTPInstance, sol::TTPSolution)

#     n = instance.numberOfNodes
#     X = Float64[]
#     Y = Float64[]
#     for i in 1:n
#         push!(X, instance.nodes[i,1])
#         push!(Y, instance.nodes[i,2])
#     end

#     # 路线
#     route = sol.tspTour
#     routeX = [X[c] for c in route]
#     routeY = [Y[c] for c in route]

#     # 画散点
#     plt = scatter(X, Y, marker=:circle, label="Cities", 
#                   title="TTP Enhanced Route", legend=:topright)

#     # 画路线
#     plot!(plt, routeX, routeY, seriestype=:path, lw=2, linecolor=:red, label="Route")

#     # 标注拿物品的城市(可选)
#     items = instance.items  # row: (profit, weight, city)
#     for i in 1:length(sol.packingPlan)
#         if sol.packingPlan[i]==1
#             c = items[i, 3]  # city
#             annotate!(plt, (X[c], Y[c]), text("Item$i", :blue, 10))
#         end
#     end

#     return plt
# end


function plot_ttp_solution(instance::TTPInstance, sol::TTPSolution)
    # 1) 获取城市坐标
    n = instance.numberOfNodes
    X = Vector{Float64}(undef, n)
    Y = Vector{Float64}(undef, n)
    for i in 1:n
        X[i] = instance.nodes[i, 1]
        Y[i] = instance.nodes[i, 2]
    end

    # 2) 路线
    route = sol.tspTour         # 1-based城市序列 (比如 [1, 5, 10, ..., 1])
    routeX = [X[c] for c in route]
    routeY = [Y[c] for c in route]

    # 3) 初步画出散点 + 路线
    plt = scatter(
        X, Y, 
        marker=:circle, 
        label="Cities", 
        title="TTP Route & Picked Items", 
        legend=:topright
    )
    plot!(
        plt, routeX, routeY, 
        seriestype=:path, 
        lw=2, 
        linecolor=:red, 
        label="Route"
    )

    # 4) 找出在每个城市拿了哪些物品
    #    items[i,:] = (profit, weight, city)
    items = instance.items
    packingPlan = sol.packingPlan

    # city_items_map 存储： city -> [(itemIdx, profit, weight), ...]
    city_items_map = Dict{Int, Vector{Tuple{Int,Int,Int}}}()

    for i in 1:length(packingPlan)
        if packingPlan[i] == 1
            city = items[i, 3]  # 第 i 个物品所在城市(1-based)
            if !haskey(city_items_map, city)
                city_items_map[city] = Vector{Tuple{Int,Int,Int}}()
            end
            push!(city_items_map[city], (i, items[i,1], items[i,2]))
        end
    end

    # 5) 对在该城市拿到物品的地方进行“文字标注”
    for (city, itemlist) in city_items_map
        # city 是 1-based
        # itemlist: [(itemIdx, profit, weight), ...]
        # 示例: "Pick i1,i2" 或详细 "i1(p=100,w=2), i3(p=200,w=5)" 等
        str_items = join(
            [ "i$(itemIdx)(p=$(profit),w=$(weight))" for (itemIdx, profit, weight) in itemlist ], 
            "; "
        )
        # 生成标注文字
        note_str = "City $city: $str_items"
        # 在图上 (X[city], Y[city]) 位置加文字标注
        annotate!(
            plt, 
            (X[city], Y[city]), 
            text(note_str, :blue, 8, :center)
        )
    end

    return plt
end


# -------------------------------
# 4) 测试
# -------------------------------

function test_enhanced()
    filename = "data/a280_n1395_uncorr-similar-weights_05.ttp.txt"
    instance = TTPInstance(filename)
    @info "Loaded instance: $(instance.problemName)"

    sol = solve_ttp_enhanced(instance)
    @info "Solution => "
    TTP.printlnSolution(sol)

    # 画图
    plt = plot_ttp_solution(instance, sol)
    
    display(plt)
end

test_enhanced()
