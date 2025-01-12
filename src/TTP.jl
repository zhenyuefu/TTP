module TTP
export TTPSolution, TTPInstance, distances, evaluate, printInstance
using Printf

mutable struct TTPSolution
    tspTour::Vector{Int}
    packingPlan::Vector{Int}
    fp::Float64
    ft::Float64
    ftraw::Int
    ob::Float64
    wend::Float64
    wendUsed::Float64
    computationTime::Int

    function TTPSolution(
        tspTour::Vector{Int},
        packingPlan::Vector{Int};
        fp::Float64 = -Inf,
        ft::Float64 = Inf,
        ftraw::Int = typemax(Int),
        ob::Float64 = -Inf,
        wend::Float64 = Inf,
        wendUsed::Float64 = Inf,
        computationTime::Int = typemax(Int)
    )
        new(
            tspTour,
            packingPlan,
            fp, ft, ftraw, ob, wend, wendUsed, computationTime
        )
    end
end

function reset!(sol::TTPSolution)
    sol.fp   = -Inf
    sol.ft   = Inf
    sol.ftraw = typemax(Int)
    sol.ob   = -Inf
    sol.wend = Inf
    sol.wendUsed = Inf
    sol.computationTime = typemax(Int)
end

function printSolution(sol::TTPSolution)
    @printf("%.2f %.2f %.2f %d %.2f %.2f %d",
        sol.wend,
        sol.wendUsed,
        sol.fp,
        sol.ftraw,
        sol.ft,
        sol.ob,
        sol.computationTime
    )
end

function printlnSolution(sol::TTPSolution)
    printSolution(sol)
    println()
end

function printFullSolution(sol::TTPSolution)
    printlnSolution(sol)
    println("tspTour = ", sol.tspTour)
    println("packingPlan = ", sol.packingPlan)
end

function getObjective(sol::TTPSolution)::Float64
    return sol.ob
end

function answer(sol::TTPSolution)::String
    tourLength = length(sol.tspTour)
    tourOut = Vector{Int}(undef, tourLength - 1)
    for i in 1:(tourLength - 1)
        # tspTour[i] 0-based -> +1
        tourOut[i] = sol.tspTour[i] + 1
    end

    itemsPerCity = div(length(sol.packingPlan), (tourLength - 2))
    packingPlanList = Int[]
    packingPlanIndex = 1
    for i in 2:(tourLength - 1)
        city = sol.tspTour[i]
        for j in 1:itemsPerCity
            if sol.packingPlan[packingPlanIndex] == 1
                itemIndex = (j-1)*(tourLength - 2) + (city - 1)
                push!(packingPlanList, itemIndex + 1)
            end
            packingPlanIndex += 1
        end
    end
    sort!(packingPlanList)

    str_tour = string(tourOut)
    str_items = string(packingPlanList)
    return str_tour * "\n" * str_items * "\n"
end

function writeResult(sol::TTPSolution, filename::String)
    open(filename, "w") do io
        write(io, answer(sol))
    end
end


struct TTPInstance
    problemName::String
    knapsackDataType::String
    numberOfNodes::Int
    numberOfItems::Int
    capacityOfKnapsack::Int
    minSpeed::Float64
    maxSpeed::Float64
    rentingRatio::Float64
    edgeWeightType::String
    nodes::Matrix{Float64}     # 节点坐标信息: numberOfNodes × 3
    items::Matrix{Int}         # 物品信息: numberOfItems × 4

    # 构造函数
    function TTPInstance(filename::String)
        # 先定义一些默认值
        problemName       = ""
        knapsackDataType  = ""
        numberOfNodes     = 0
        numberOfItems     = 0
        capacityOfKnapsack= 0
        minSpeed          = 0.0
        maxSpeed          = 0.0
        rentingRatio      = 0.0
        edgeWeightType    = ""

        # 为了能在定义时先声明，再在读到具体值后再赋值，这里先构造一个空数组
        # 后续遇到 NODE_COORD_SECTION 或 ITEMS SECTION 时再进行实际初始化
        nodes = Matrix{Int}(undef, 0, 0)
        items = Matrix{Int}(undef, 0, 0)

        # 打开文件并读取
        file = open(filename, "r")

        # 用一个临时变量，用于控制当读到 NODE_COORD_SECTION 或 ITEMS SECTION 后
        # 后续行应该如何读取
        readingNodeCoords = false
        readingItems      = false
        nodeCount         = 0  # 用于计数已经读取了多少行 node
        itemCount         = 0  # 用于计数已经读取了多少行 item

        while !eof(file)
            line = strip(readline(file))
            # 根据关键字解析
            if startswith(line, "PROBLEM NAME")
                # 如: "PROBLEM NAME: a280_n1395_bounded-strongly-corr_1"
                # 取冒号后部分
                parts = split(line, ":")
                # 去除空白
                namePart = replace(strip(parts[end]), r"\s+" => "")
                problemName = namePart

            elseif startswith(line, "KNAPSACK DATA TYPE")
                parts = split(line, ":")
                dataTypePart = replace(strip(parts[end]), r"\s+" => "")
                knapsackDataType = dataTypePart

            elseif startswith(line, "DIMENSION") || startswith(line, "NUMBER OF NODES")
                parts = split(line, ":")
                numNodes = parse(Int, replace(strip(parts[end]), r"\s+" => ""))
                numberOfNodes = numNodes

            elseif startswith(line, "NUMBER OF ITEMS")
                parts = split(line, ":")
                numItems = parse(Int, replace(strip(parts[end]), r"\s+" => ""))
                numberOfItems = numItems

            elseif startswith(line, "CAPACITY OF KNAPSACK")
                parts = split(line, ":")
                cap = parse(Int, replace(strip(parts[end]), r"\s+" => ""))
                capacityOfKnapsack = cap

            elseif startswith(line, "MIN SPEED")
                parts = split(line, ":")
                ms = parse(Float64, replace(strip(parts[end]), r"\s+" => ""))
                minSpeed = ms

            elseif startswith(line, "MAX SPEED")
                parts = split(line, ":")
                ms = parse(Float64, replace(strip(parts[end]), r"\s+" => ""))
                maxSpeed = ms

            elseif startswith(line, "RENTING RATIO")
                parts = split(line, ":")
                rr = parse(Float64, replace(strip(parts[end]), r"\s+" => ""))
                rentingRatio = rr

            elseif startswith(line, "EDGE_WEIGHT_TYPE")
                parts = split(line, ":")
                ewt = replace(strip(parts[end]), r"\s+" => "")
                edgeWeightType = ewt

            elseif startswith(line, "NODE_COORD_SECTION")
                # 根据已知 numberOfNodes 初始化 nodes 数组
                # 大小为 numberOfNodes × 3
                nodes = Matrix{Float64}(undef, numberOfNodes, 2)
                readingNodeCoords = true
                readingItems = false
                nodeCount = 0
                continue

            elseif startswith(line, "ITEMS SECTION")
                # 根据已知 numberOfItems 初始化 items 数组
                # 大小为 numberOfItems × 4
                items = Matrix{Int}(undef, numberOfItems, 3)
                readingNodeCoords = false
                readingItems = true
                itemCount = 0
                continue
            end

            # 如果在读 NODE_COORD_SECTION 的行
            if readingNodeCoords && nodeCount < numberOfNodes
                splittedLine = split(line)
                # splittedLine 理论上长度为 3，分别是: index, x, y
                # TTP 文件中城市索引从 1 开始，
                cityIndex = parse(Int, splittedLine[1])
                xCoord    = parse(Float64, splittedLine[2])
                yCoord    = parse(Float64, splittedLine[3])

                nodeCount += 1
            
                nodes[cityIndex, 1] = xCoord
                nodes[cityIndex, 2] = yCoord

            # 如果在读 ITEMS SECTION 的行
            elseif readingItems && itemCount < numberOfItems
                splittedLine = split(line)
                # splittedLine 理论上长度为 4，分别是: itemIndex, profit, weight, city
               
                itemIdx = parse(Int, splittedLine[1])
                profit  = parse(Int, splittedLine[2])
                weight  = parse(Int, splittedLine[3])
                city    = parse(Int, splittedLine[4])

                itemCount += 1
                items[itemIdx, 1] = profit
                items[itemIdx, 2] = weight
                items[itemIdx, 3] = city
            end
        end

        close(file)

        # 返回构造好的实例
        new(problemName,
            knapsackDataType,
            numberOfNodes,
            numberOfItems,
            capacityOfKnapsack,
            minSpeed,
            maxSpeed,
            rentingRatio,
            edgeWeightType,
            nodes,
            items)
    end
end

function distances(instance::TTPInstance, i::Int, j::Int)::Float64
    # i, j 理论上是城市索引(从0开始)；而存储在 instance.nodes 的是从1开始的行号
    # 如果 instance.nodes 的第 i 行存储的是城市 i-1 的信息，就需要注意转化
    # 这里简单假设传入 i, j 都是 "行号"(1-based)，或者自己按实际需求做映射。

    x_i = instance.nodes[i, 1]
    y_i = instance.nodes[i, 2]
    x_j = instance.nodes[j, 1]
    y_j = instance.nodes[j, 2]

    return sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
end


function evaluate(instance::TTPInstance, solution::TTPSolution)
    tour = solution.tspTour
    z    = solution.packingPlan

    # 读取背包容量/租用率/速度上下限
    weightofKnapsack = instance.capacityOfKnapsack
    rentRate         = instance.rentingRatio
    vmin             = instance.minSpeed
    vmax             = instance.maxSpeed

    # 初始化结果
    solution.ftraw = 0.0
    solution.ft    = 0.0
    solution.fp    = 0.0

    # 判断路径首尾是否相同
    if tour[1] != tour[end]
        @printf("ERROR: The last city must be the same as the first city\n")
        # 如果需要可以清空 solution 或做其他操作
        return
    end

    # 当前背包重量
    wc = 0.0

    # 假设 itemsPerCity = length(z) ÷ (length(tour) - 2)

    itemsPerCity = length(z) ÷ (length(tour) - 2)

    # 遍历路线
    for i = 1:(length(tour)-1)
        currentCityTEMP = tour[i]        # 0-based
        # i=1 的时候是第一座城市(往往是起点)

        # ========== 拿物品的逻辑 ==========
        # 在 Java 代码中：if(i>0) { ... }
        # 相当于跳过第一个城市(起点不拿东西)
        if i > 1
            # cityIndexForItem = currentCityTEMP - 1
            cityIndexForItem = currentCityTEMP - 1

            # 对应 Java 中的 for (int itemNumber=0; itemNumber<itemsPerCity; itemNumber++)
            for itemNumber in 0:(itemsPerCity-1)
                # indexOfPackingPlan = (i-1)*itemsPerCity+itemNumber
                indexOfPackingPlan = (i-2)*itemsPerCity + (itemNumber+1)
                # 注意 Julia 是 1-based，所以要注意加减

                # itemIndex = cityIndexForItem + itemNumber*(instance.numberOfNodes-1)
                itemIndex = cityIndexForItem + itemNumber*(instance.numberOfNodes-1)

                # 如果 z[indexOfPackingPlan] == 1，说明要拿这个物品
                # 注意 Julia 下标
                if z[indexOfPackingPlan] == 1
                    currentWC  = instance.items[itemIndex+1, 3]  # weight
                    currentFP  = instance.items[itemIndex+1, 2]  # profit
                    wc        += currentWC
                    solution.fp += currentFP
                end
            end
        end

        # ========== 计算距离 + 时间 ==========
        # 下一个城市的下标 h
        h = (i+1)
        # 计算距离
        d = distances(instance, tour[i], tour[h])
        solution.ftraw += d

        # 速度衰减模型
        # ( distance / (vmax - wc * (vmax - vmin)/weightofKnapsack) )
        solution.ft += d / (vmax - wc * (vmax - vmin)/weightofKnapsack)
    end

    # 记录最终的背包使用情况
    solution.wendUsed = wc
    solution.wend     = weightofKnapsack - wc

    # 目标函数
    solution.ob = solution.fp - solution.ft * rentRate

    return
end

function printInstance(instance::TTPInstance; shortSummary::Bool=true)
    if shortSummary
        @printf("TTP Instance: %s %s %d %d %d %.3f %.3f %.3f\n",
            instance.problemName,
            instance.knapsackDataType,
            instance.numberOfNodes,
            instance.numberOfItems,
            instance.capacityOfKnapsack,
            instance.minSpeed,
            instance.maxSpeed,
            instance.rentingRatio
        )
    else
        println("---- TTP Instance START ----")
        @printf("Name: %s\n", instance.problemName)
        @printf("Type: %s\n", instance.knapsackDataType)
        @printf("Nodes: %d\n", instance.numberOfNodes)
        @printf("Items: %d\n", instance.numberOfItems)
        @printf("Capacity: %d\n", instance.capacityOfKnapsack)
        @printf("Speed Range: [%.3f, %.3f]\n", instance.minSpeed, instance.maxSpeed)
        @printf("Renting Ratio: %.3f\n", instance.rentingRatio)
        @printf("Edge Weight Type: %s\n", instance.edgeWeightType)

        println("NODE_COORD_SECTION:")
        for i in 1:instance.numberOfNodes
            println(instance.nodes[i, :]) 
        end

        println("ITEMS SECTION:")
        for i in 1:instance.numberOfItems
            println(instance.items[i, :])
        end

        println("---- TTP Instance END ----")
    end
end

end # module TTP