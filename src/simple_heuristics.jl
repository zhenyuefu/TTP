include("TTP.jl")
using .TTP
include("utils.jl")
using .Utils

function simpleHeuristic(instance::TTPInstance, route::Vector{Int})::TTPSolution
    """
    给定 TSP 路线的前提下，为 TTP 问题构造背包规划方案。
    返回一个完整的 TTPSolution内部调用 evaluate() 进行评估，并在解中存储目标值。

    参数:
        instance: TTPInstance, 包含城市坐标、物品信息、背包容量、速度上下限等
        route:    给定的 TSP 路线 (城市序列)，其中 route[1] == route[end]

    返回:
        TTPSolution包括:
          - tspTour = route
          - packingPlan = 选取物品的 0/1 列表
          - ob = 最终目标值 (已经比较过 -R * t' )
          - 其它指标 (fp, ft, 等) 会在 evaluate() 时被更新
    """

    n = length(route)
    W = instance.capacityOfKnapsack
    R = instance.rentingRatio
    v_max = instance.maxSpeed
    v_min = instance.minSpeed
    v = (v_max - v_min) / W   # 速度衰减系数

    # 1) 预计算: d[x_i] 表示从城市 x_i 出发到终点(沿给定 route)的距离
    #    这里 x_i = route[i]，i 从 2 到 n-1(忽略起点与最后的回城)
    D = zeros(Float64, n)  # 存储每个索引 i 对应的剩余距离
    for i in 2:(n-1)
        dist_sum = 0.0
        for k in i:(n-1)
            dist_sum += distances(instance, route[k], route[k+1])
        end
        D[i] = dist_sum
    end

    # 2) 在不携带任何物品的情况下，小偷按最大速度 v_max 完成整条路线的总时间 t'
    t_prime = 0.0
    for i in 1:(n-1)
        d_seg = distances(instance, route[i], route[i+1])
        t_prime += d_seg / v_max
    end

    # 3) 为每个可能物品计算其:
    #    - t_{x_i, k}        = D[i] / (v_max - v * w_{ik})
    #    - t'_{x_i, k}       = t' - (D[i]/v_max) + t_{x_i,k}
    #    - score_{x_i, k}    = p_{ik} - R * t_{x_i, k}
    #    - u_{x_i, k}        = R * t' + ( p_{ik} - R * t'_{x_i, k} )
    #    (city = route[i], i in [2..(n-1)]，因为城市1和城市n没有物品)

    # 收集所有物品信息，存到一个数组中，后面排序再选
    # 每个元素可以存: (globalItemIndex, cityIndexInRoute, profit, weight, score, u)
    item_candidates = Vector{Tuple{Int,Int,Float64,Float64,Float64,Float64}}()
    itemCount = size(instance.items, 1)  # 总物品数

    for itemIdx in 1:itemCount
        profit = instance.items[itemIdx, 1]
        weight = instance.items[itemIdx, 2]
        city = instance.items[itemIdx, 3]  # 此物品所在的城市(1-based)

        # 找到 city 在 route 中的位置 idxInRoute
        #   route 里可能 city 出现在哪个 i？
        #   一般我们只在 [2..(n-1)] 找即可
        #   如果找不到，说明该城市不在 route 的中间段(或是起终点)
        idxInRoute = findall(i -> route[i] == city, 2:(n-1))
        if length(idxInRoute) == 0
            continue  # 说明这个物品不在 2..(n-1) 之间的城市里
        end
        # 简化起见，假设每个 city 在 route 中只出现一次
        iRoute = first(idxInRoute)

        # 计算 t_{x_i,k} = D[iRoute] / (v_max - v*weight)
        # 若 weight = 0，则速度 = v_max，不会出问题
        denom_speed = v_max - v * weight
        # 如果 denom_speed <= 0，说明物品太重导致速度衰减不可行(或出现极端情况)，可跳过
        if denom_speed <= 0
            continue
        end

        t_xi_k = D[iRoute] / denom_speed
        # t'_{x_i, k} = t' - (D[iRoute]/v_max) + t_xi_k
        t_xi_k2 = t_prime - (D[iRoute] / v_max) + t_xi_k

        score = profit - R * t_xi_k
        u_val = R * t_prime + (profit - R * t_xi_k2)

        push!(item_candidates, (
            itemIdx,         # 该物品全局索引
            iRoute,          # 物品所处路线下标
            profit,          # p
            weight,          # w
            score,           # score
            u_val            # u
        ))
    end

    # 4) 将物品按 score 从大到小排序
    sort!(item_candidates, by=x -> x[5], rev=true)

    # 5) 背包容量已用变量
    W_c = 0.0

    # 准备一个全零的 packingPlan 数组 (与 instance.items 行对应)
    packingPlan = zeros(Int, itemCount)

    # 6) 逐个物品按分数从大到小尝试放入背包
    for (idxGlob, _, p, w, s, u) in item_candidates
        if (W_c + w <= W) && (u > 0)  # 符合: 有容量且适应值>0
            packingPlan[idxGlob] = 1
            W_c += w
            if W_c >= W
                break  # 背包已满, 跳出循环
            end
        end
    end

    # 7) 构造 TTPSolution 并评估
    #    evaluate 会基于整条路线 + 选中的物品，计算最终 ob, fp, ft 等
    sol = TTPSolution(route, packingPlan)
    TTP.evaluate(instance, sol)

    # 对照纯空背包时的收益: -R * t'
    # 算法描述说: Z^* = max( Z(\Pi, P), -R * t' )
    sol.ob = max(sol.ob, -R * t_prime)

    return sol
end


filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
instance = TTPInstance(filename)
start_time = Utils.start_timing()
route = lkh(instance)
sol = simpleHeuristic(instance, route)
elapsed = Utils.stop_timing(start_time)
sol.computationTime = elapsed

printFullSolution(sol)