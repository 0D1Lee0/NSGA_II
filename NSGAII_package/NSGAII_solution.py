from random_task import *
from nsgaii import *
from ranking import *

#输入设备信息
equipments = pd.DataFrame({'number':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   'state':['wait_for_production', 'leisure', 'leisure', 'stop', 'wait_for_production', 'leisure',
                            'leisure', 'leisure', 'leisure', 'wait_for_quality_inspection', 'wait_for_the_packing',
                            'wait_for_production', 'stop', 'leisure', 'in_production','in_production', 'leisure',
                            'wait_for_the_packing', 'wait_for_quality_inspection', 'stop'],
                   'production_capacity':[1000, 1000, 1000, 800, 500, 700, 1200, 1300, 1000, 900,
                                          900, 1200, 1200, 1500, 2000, 2000, 1800, 1200, 1000, 1000],
                   'Support_product_line':['A|B|C|D|E|F|G|H', 'A|B|C|D', 'A|B|C|D', 'E|F|G|H', 'E|F|G|H','A|B|C|G|F',
                                             'D|E|F', 'A', 'H', 'A|B|C|D|E|F|G|H', 'A|B|C|D|E|F|G|H', 'A|B|C|D|E|F|G|H',
                                             'A|B|C|D|E|F|G|H', 'A|B|C|D|E|F|H', 'E|F|G|H', 'E|F|G|H', 'A|B|C', 'A|B|C',
                                             'A|B|C|D|E|F|G|H', 'A|B|C|D|E|F|G|H'],
                   'Maintenance_date':['2020-02-17', '2020-02-17', '2020-02-17', '2020-02-17', '2020-02-17',
                                       '2020-02-17', '2020-02-17', '2020-02-14', '2020-02-14', '2020-02-14',
                                       '2020-02-14', '2020-02-14', '2020-02-14', '2020-02-14', '2020-02-27',
                                       '2020-02-14', '2020-02-14', '2020-02-14', '2020-02-14', '2020-02-14']})

#已有分配方案
assigned_scheme = pd.DataFrame({'number': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                'task_number': ['LT_IT001', 'LT_IT002', 'LT_IT003', 'LT_IT004', 'LT_IT007', 'LT_IT008',
                                                'LT_IT009', 'LT_IT005', 'LT_IT006'],
                                'production_quantity':[900, 1150, 900, 1000, 1200, 2000, 2000, 1000, 500],
                                'production_series':['G', 'A', 'H', 'G', 'E', 'F', 'H', 'C', 'H'],
                                'machine_use': [11, 18, 10, 19, 12, 15, 16, 1, 5],
                                'state':['wait_for_the_packing', 'wait_for_the_packing', 'wait_for_quality_inspection', 'wait_for_quality_inspection', 'in_production', 'in_production', 'in_production', 'wait_for_production', 'wait_for_production'],
                                'delivery_date': ['2020-02-27', '2020-02-27', '2020-02-28', '2020-02-29', '2020-03-20',
                                                  '2020-03-20', '2020-03-20', '2020-03-09', '2020-03-09'],
                                'machine_use': [[11], [18], [10], [19], [12], [15], [16], [1], [5]]})

#输入待分配任务
next_task = pd.DataFrame({'seria_number':['LT_IT030', 'LT_IT032', 'LT_IT033', 'LT_IT034', 'LT_IT035', 'LT_IT036'],
                        'total_production_capacity':[1500, 1000, 500, 1300, 1800, 3000],
                        'production_series':['C', 'G', 'F', 'D', 'E', 'D'],
                        'delivery_date':['2020-03-19','2020-03-01','2020-03-01','2020-03-06','2020-02-29','2020-03-01'],
                        'need_material':[[500,0,0,0,300,0,0,0,150,0,400,0,0,0,0,0,0,0,0,150], [0,400,300,290,0,0,0,0,0,0,0,0,0,0,0,0,3,5,2,0],
                                         [0,150,200,0,0,0,0,0,0,0,0,0,120,0,0,0,30,0,0,0], [0,0,0,0,200,0,0,0,0,0,0,0,0,600,500,0,0,0,0,0],
                                         [600,600,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [300,400,100,500,200,1000,500,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                        'product_cycle':[14.5, 27, 9, 23, 36, 27.5]})
#输入物料的信息
stock = pd.DataFrame({'Name_of_goods':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    'inventory':[10000, 20000, 5000, 6000, 1300, 1000, 9000, 1500, 800, 900, 1200, 106, 108, 12000, 15000, 7800, 1256, 78, 95, 10000]})


"""#初始化种群
population_size = 300
iteration = 500

start = time.time()
max_crossover_rate = 0.9
min_crossover_rate = 0.4
max_variation_rate = 0.1
min_variation_rate = 0.01
crossover_rate = max_crossover_rate
variation_rate = max_variation_rate

equipments = equip_coding(equipments)
assigned_scheme = assigned_coding(assigned_scheme)
next_task = task_coding(next_task)
next_task =bubble_sorted_task(next_task)
present_time = [2020,2,26,0]
solution = [Solution(next_task) for i in range(0, population_size)]
#print(assigned_scheme)
print("初始化种群")
for i in range(0, population_size):
    solution[i] = Solution(next_task)
    solution[i].match(equipments, next_task)
    solution[i].record_state(equipments)
    solution[i].find_rank_number(assigned_scheme, next_task, stock)
    solution[i].function1(next_task)
    solution[i].function6()
    solution[i].function5(equipments, next_task)
    solution[i].function7(next_task, equipments)
print("初始化成功")
print("开始迭代")
original_diversity = population_diversity(solution)
print("初代种群多样性：")
print(original_diversity)
print(solution[0].func1)
print(solution[0].rank_number)
print(solution[0].machine_use)
print("----------------")
generation = 1
x = []

y = []

while generation <= iteration:
    solution = crossover(solution, crossover_rate, variation_rate, population_size, equipments, next_task)
    for i in range(population_size, len(solution)):
        solution[i].record_state(equipments)
        solution[i].find_rank_number(assigned_scheme, next_task, stock)
        solution[i].function1(next_task)
        #s.function2(next_task, present_time)
        #s.function3()
        #s.function4(next_task)
        solution[i].function6()
        solution[i].function5(equipments, next_task)
        solution[i].function7(next_task, equipments)

    g = str(generation)
    di = "第"
    dai = "代"
    ss = di + g + dai
    #print(ss)
    diversity = population_diversity(solution)
    #print("种群多样性：")
    #print(diversity)
    #print("----------------")
    x.append(generation)
    y.append(diversity)
    if diversity == 1:
        break
    control = govern(solution)
    classification = fast_nondomination(control, solution)
    count = 0
    for c in classification:
        count = count + len(c)
    solution = crowding_degree(solution)
    solution = crowded_comparison_operator(classification, solution, population_size)
    crossover_rate = adaptive_crossover_rate(generation, iteration, max_crossover_rate, min_crossover_rate)
    variation_rate = adaptive_variation_rate(generation, iteration, max_variation_rate, min_variation_rate)
    generation = generation + 1


end = time.time()
print(end-start)
print(solution[0].func1)
print(solution[0].rank_number)
print(solution[0].machine_use)
print("************************")
print(solution[-1].func1)
print(solution[-1].rank_number)
print(solution[-1].machine_use)


x = np.arange(1, generation+1, 1)

plt.xlabel("iteration")  #x轴上的名字
plt.ylabel("Solution Diverity") #y轴上的名字
plt.plot(x, y, color='green', linewidth=5)
plt.show()"""



#50个任务，20种物料
weight = [0.2, 0.4, 0.2, 0.1, 0.1]
#weight = input_weight()
task_quanity_1 = 10
material_quanity_1 = 20
new_task_1 = randomly_generated_task(task_quanity_1, material_quanity_1)
new_stock_1 = random_material(material_quanity_1)

run_time = []
f1 = []
f2 = []
f3 = []
f4 = []
f5 = []

equipments = equip_coding(equipments)
assigned_scheme = assigned_coding(assigned_scheme)
"""next_task_1 = random_task_coding(new_task_1)
next_task = bubble_sorted_task(next_task_1)"""
next_task = task_coding(next_task)
next_task = bubble_sorted_task(next_task)
present_time = [2020, 2, 26, 0]


start = time.time()

population_size = 200
iteration = 30
max_crossover_rate = 0.9
min_crossover_rate = 0.4
max_variation_rate = 0.1
min_variation_rate = 0.01
crossover_rate = max_crossover_rate
variation_rate = max_variation_rate


solution = [Solution(next_task) for i in range(0, population_size)]
print("初始化种群")
for i in range(0, population_size):
    solution[i] = Solution(next_task)
    solution[i].match(equipments, next_task)
    solution[i].record_state(equipments)
    solution[i].find_rank_number(assigned_scheme, next_task, stock)
    solution[i].function1(next_task)
    solution[i].function2(next_task, present_time)
    solution[i].function3(equipments, next_task)
    solution[i].function4()
    solution[i].function5(next_task, equipments)
print("初始化成功")
print("开始迭代")
original_diversity = population_diversity(solution)
generation = 0
while generation < iteration:
    solution = crossover(solution, crossover_rate, variation_rate, population_size, equipments, next_task)
    for i in range(population_size, len(solution)):
        solution[i].record_state(equipments)
        solution[i].find_rank_number(assigned_scheme, next_task, stock)
        solution[i].function1(next_task)
        solution[i].function2(next_task, present_time)
        solution[i].function3(equipments, next_task)
        solution[i].function4()
        solution[i].function5(next_task, equipments)

    g = str(generation+1)
    di = "第"
    dai = "代"
    ss = di + g + dai
    # print(ss)
    #diversity = population_diversity(solution)
    # print("种群多样性：")
    # print(diversity)
    # print("----------------")
    control = govern(solution)
    classification = fast_nondomination(control, solution)
    count = 0
    #for c in classification:
        #count = count + len(c)
    solution = crowding_degree(solution)
    new_solution = fixed_comparison_operator(classification, solution, population_size)
    crossover_rate = adaptive_crossover_rate(generation, iteration, max_crossover_rate, min_crossover_rate)
    variation_rate = adaptive_variation_rate(generation, iteration, max_variation_rate, min_variation_rate)
    generation = generation + 1
    solution = new_solution

final_solution = function_normalize(solution, population_size)


final_solution = calculate_fitness(final_solution, weight, population_size)
sorted_solution = optimal_individual(final_solution, population_size)

print("------------最优方案--------------")
print("排位号：", end=" ")
print(final_solution[sorted_solution[0]].rank_number)
print("分配的设备：", end=" ")
print(final_solution[sorted_solution[0]].machine_use)

"""print(final_solution[sorted_solution[0]].func1)
print(final_solution[sorted_solution[0]].func2)
print(final_solution[sorted_solution[0]].func3)
print(final_solution[sorted_solution[0]].func4)
print(final_solution[sorted_solution[0]].func5)
print(final_solution[sorted_solution[0]].degree)"""





"""end = time.time()
run_time.append(end-start)
f1.append(solution[0].func1)
f2.append(solution[0].func2)
f3.append(solution[0].func3)
f4.append(solution[0].func4)
f5.append(solution[0].func5)



mean_time = sum(run_time)/(len(run_time))
mean_f1 = sum(f1)/(len(f1))
mean_f2 = sum(f2)/(len(f2))
mean_f3 = sum(f3)/(len(f3))
mean_f4 = sum(f4)/(len(f4))
mean_f5 = sum(f5)/(len(f5))

print("mean_time:", end=' ')
print(mean_time)
print("mean_f1:", end=' ')
print(mean_f1)
print("mean_f2:", end=' ')
print(mean_f2)
print("mean_f3:", end=' ')
print(mean_f3)
print("mean_f4:", end=' ')
print(mean_f4)
print("mean_f5:", end=' ')
print(mean_f5)"""


