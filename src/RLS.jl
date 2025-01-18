include("TTP.jl")
using .TTP
include("utils.jl")
using .Utils

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



filename = "data/a280_n279_bounded-strongly-corr_01.ttp.txt"
instance = TTPInstance(filename)
start_time = Utils.start_timing()
route = lkh(instance)
sol = randomLocalSearch(instance, route)
elapsed = Utils.stop_timing(start_time)
sol.computationTime = elapsed

printFullSolution(sol)