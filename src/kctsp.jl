using JuMP, Gurobi

# 问题参数
function solve_kctsp(n, m, d, w, p, W, K)
    # 创建模型
    model = Model(Gurobi.Optimizer)
    
    # 决策变量
    @variable(model, x[1:n,1:n], Bin)  # 路径选择
    @variable(model, y[i=1:n,k=1:m[i]], Bin)  # 物品选择
    @variable(model, W_i[1:n] >= 0)  # 离开每个城市时的重量
    @variable(model, 1 <= u[1:n] <= n)  # MTZ子回路消除变量
    
    # 目标函数
    @objective(model, Max, 
        sum(p[i][k] * y[i,k] for i=1:n for k=1:m[i]) - 
        K * sum(d[i,j] * W_i[i] * x[i,j] for i=1:n for j=1:n if i!=j)
    )
    
    # 约束条件
    # 1. TSP约束 - 每个城市访问一次
    for i=1:n
        @constraint(model, sum(x[i,j] for j=1:n if j!=i) == 1)  # 出度
        @constraint(model, sum(x[j,i] for j=1:n if j!=i) == 1)  # 入度
    end
    
    # 2. MTZ子回路消除约束
    for i=2:n
        for j=2:n
            if i != j
                @constraint(model, u[i] - u[j] + n*x[i,j] <= n-1)
            end
        end
    end
    @constraint(model, u[1] == 1)
    
    # 3. 背包容量约束
    @constraint(model, 
        sum(w[i][k] * y[i,k] for i=1:n for k=1:m[i]) <= W
    )
    
    # 4. 累计重量计算
    # 初始城市重量
    @constraint(model, W_i[1] == sum(w[1][k] * y[1,k] for k=1:m[1]))
    
    # 其他城市重量
    M = sum(maximum(w[i]) for i=1:n) # 足够大的常数
    for i=2:n
        for j=1:n
            if i != j
                @constraint(model, 
                    W_i[i] >= W_i[j] + sum(w[i][k] * y[i,k] for k=1:m[i]) - 
                    M * (1 - x[j,i])
                )
            end
        end
    end
    
    # 求解
    optimize!(model)
    
    return objective_value(model), 
           value.(x), 
           value.(y), 
           value.(W_i)
end

# 使用示例
function example()
    n = 3  # 城市数
    m = [2, 2, 2]  # 每个城市的物品数
    d = [
        0 10 20;
        10 0 30;
        20 30 0
    ]  # 距离矩阵
    w = [
        [2, 3],  # 城市1的物品重量
        [4, 1],  # 城市2的物品重量
        [3, 2]   # 城市3的物品重量
    ]
    p = [
        [10, 15],  # 城市1的物品价值
        [20, 5],   # 城市2的物品价值
        [12, 8]    # 城市3的物品价值
    ]
    W = 10  # 背包容量
    K = 0.1  # 每公里每公斤的运输成本
    
    obj, x, y, W_i = solve_kctsp(n, m, d, w, p, W, K)
    println("最优目标值: ", obj)
    println("路径选择: ", x)
    println("物品选择: ", y)
    println("各城市离开时重量: ", W_i)
end

example()