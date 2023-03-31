from random_task import *
from ranking import *
from nsgaii import *
import numpy as np

#随机生成任务和物料
task_quanity_1 = 100
material_quanity = 30

task = randomly_generated_task(task_quanity_1, material_quanity)
material = random_material(material_quanity)

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

task = random_task_coding(task)
task = bubble_sorted_task(task)
equipments = equip_coding(equipments)
assigned_scheme = assigned_coding(assigned_scheme)
present_time = [2020, 2, 26, 0]

times = 5

population_size = 200  #
iteration = 500

f1 = []
f2 = []
f3 = []
f4 = []
f5 = []


#固定拥挤度排序，固定交叉变异率
fixed_crossover_rate = 0.9
fixed_variation_rate = 0.05

run_time_1 = []
f1_1 = []
f2_1 = []
f3_1 = []
f4_1 = []
f5_1 = []
print("----------固定拥挤度排序，固定交叉变异率---------")
for ti in range(0, times):
    start_1 = time.time()

    solution = [Solution(task) for i in range(0, population_size)]

    for i in range(0, population_size):
        solution[i] = Solution(task)
        solution[i].match(equipments, task)
        solution[i].record_state(equipments)
        solution[i].find_rank_number(assigned_scheme, task, material)
        solution[i].function1(task)
        solution[i].function2(task, present_time)
        solution[i].function3(equipments, task)
        solution[i].function4()
        solution[i].function5(task, equipments)

    original_diversity = population_diversity(solution)
    generation = 1
    while generation <= iteration:
        solution = crossover(solution, fixed_crossover_rate, fixed_variation_rate, population_size, equipments, task)
        for i in range(population_size, len(solution)):
            solution[i].record_state(equipments)
            solution[i].find_rank_number(assigned_scheme, task, material)
            solution[i].function1(task)
            solution[i].function2(task, present_time)
            solution[i].function3(equipments, task)
            solution[i].function4()
            solution[i].function5(task, equipments)
        diversity = population_diversity(solution)
        if diversity == 1:
            break
        control = govern(solution)
        classification = fast_nondomination(control, solution)
        count = 0
        for c in classification:
            count = count + len(c)
        solution = crowding_degree(solution)
        solution = fixed_comparison_operator(classification, solution, population_size)
        #crossover_rate = adaptive_crossover_rate(generation, iteration, max_crossover_rate, min_crossover_rate)
        #variation_rate = adaptive_variation_rate(generation, iteration, max_variation_rate, min_variation_rate)
        generation = generation + 1

    end_1 = time.time()
    run_time_1.append(end_1 - start_1)
    f1_1.append(solution[0].func1)
    f2_1.append(solution[0].func2)
    f3_1.append(solution[0].func3)
    f4_1.append(solution[0].func4)
    f5_1.append(solution[0].func5)

mean_time_1 = sum(run_time_1) / (len(run_time_1))
mean_f1_1 = sum(f1_1) / (len(f1_1))
mean_f2_1 = sum(f2_1) / (len(f2_1))
mean_f3_1 = sum(f3_1) / (len(f3_1))
mean_f4_1 = sum(f4_1) / (len(f4_1))
mean_f5_1 = sum(f5_1) / (len(f5_1))

f1.append(f1_1)
f2.append(f2_1)
f3.append(f3_1)
f4.append(f4_1)
f5.append(f5_1)

print("mean_time:", end=' ')
print(mean_time_1)
print("mean_f1:", end=' ')
print(mean_f1_1)
print("mean_f2:", end=' ')
print(mean_f2_1)
print("mean_f3:", end=' ')
print(mean_f3_1)
print("mean_f4:", end=' ')
print(mean_f4_1)
print("mean_f7:", end=' ')
print(mean_f5_1)

print("-----------------------------------")
#固定拥挤度排序，动态交叉变异率
max_crossover_rate = 0.9
min_crossover_rate = 0.4
max_variation_rate = 0.1
min_variation_rate = 0.01
dynamic_crossover_rate = max_crossover_rate
dynamic_variation_rate = max_variation_rate

run_time_2 = []
f1_2 = []
f2_2 = []
f3_2 = []
f4_2 = []
f5_2 = []

print("----------固定拥挤度排序，动态交叉变异率---------")
for ti in range(0, times):
    start_2 = time.time()

    solution = [Solution(task) for i in range(0, population_size)]

    for i in range(0, population_size):
        solution[i] = Solution(task)
        solution[i].match(equipments, task)
        solution[i].record_state(equipments)
        solution[i].find_rank_number(assigned_scheme, task, material)
        solution[i].function1(task)
        solution[i].function2(task, present_time)
        solution[i].function3(equipments, task)
        solution[i].function4()
        solution[i].function5(task, equipments)

    original_diversity = population_diversity(solution)
    generation = 0
    while generation < iteration:
        solution = crossover(solution, dynamic_crossover_rate, dynamic_variation_rate, population_size, equipments, task)
        for i in range(population_size, len(solution)):
            solution[i].record_state(equipments)
            solution[i].find_rank_number(assigned_scheme, task, material)
            solution[i].function1(task)
            solution[i].function2(task, present_time)
            solution[i].function3(equipments, task)
            solution[i].function4()
            solution[i].function5(task, equipments)
        diversity = population_diversity(solution)
        if diversity == 1:
            break
        control = govern(solution)
        classification = fast_nondomination(control, solution)
        count = 0
        for c in classification:
            count = count + len(c)
        solution = crowding_degree(solution)
        solution = fixed_comparison_operator(classification, solution, population_size)
        dynamic_crossover_rate = adaptive_crossover_rate(generation, iteration, max_crossover_rate, min_crossover_rate)
        dynamic_variation_rate = adaptive_variation_rate(generation, iteration, max_variation_rate, min_variation_rate)
        generation = generation + 1

    end_2 = time.time()
    run_time_2.append(end_2 - start_2)
    f1_2.append(solution[0].func1)
    f2_2.append(solution[0].func2)
    f3_2.append(solution[0].func3)
    f4_2.append(solution[0].func4)
    f5_2.append(solution[0].func5)

mean_time_2 = sum(run_time_2) / (len(run_time_2))
mean_f1_2 = sum(f1_2) / (len(f1_2))
mean_f2_2 = sum(f2_2) / (len(f2_2))
mean_f3_2 = sum(f3_2) / (len(f3_2))
mean_f4_2 = sum(f4_2) / (len(f4_2))
mean_f5_2 = sum(f5_2) / (len(f5_2))

f1.append(f1_2)
f2.append(f2_2)
f3.append(f3_2)
f4.append(f4_2)
f5.append(f5_2)

print("mean_time:", end=' ')
print(mean_time_2)
print("mean_f1:", end=' ')
print(mean_f1_2)
print("mean_f2:", end=' ')
print(mean_f2_2)
print("mean_f3:", end=' ')
print(mean_f3_2)
print("mean_f4:", end=' ')
print(mean_f4_2)
print("mean_f7:", end=' ')
print(mean_f5_2)

print("-----------------------------------")


#动态拥挤度排序，固定交叉变异率
fixed_crossover_rate = 0.9
fixed_variation_rate = 0.05

run_time_3 = []
f1_3 = []
f2_3 = []
f3_3 = []
f4_3 = []
f5_3 = []
print("----------动态拥挤度排序，固定交叉变异率---------")
for ti in range(0, times):
    start_3 = time.time()

    solution = [Solution(task) for i in range(0, population_size)]

    for i in range(0, population_size):
        solution[i] = Solution(task)
        solution[i].match(equipments, task)
        solution[i].record_state(equipments)
        solution[i].find_rank_number(assigned_scheme, task, material)
        solution[i].function1(task)
        solution[i].function2(task, present_time)
        solution[i].function3(equipments, task)
        solution[i].function4()
        solution[i].function5(task, equipments)

    original_diversity = population_diversity(solution)
    generation = 0
    while generation < iteration:
        solution = crossover(solution, fixed_crossover_rate, fixed_variation_rate, population_size, equipments, task)
        for i in range(population_size, len(solution)):
            solution[i].record_state(equipments)
            solution[i].find_rank_number(assigned_scheme, task, material)
            solution[i].function1(task)
            solution[i].function2(task, present_time)
            solution[i].function3(equipments, task)
            solution[i].function4()
            solution[i].function5(task, equipments)
        diversity = population_diversity(solution)
        if diversity == 1:
            break
        control = govern(solution)
        classification = fast_nondomination(control, solution)
        count = 0
        for c in classification:
            count = count + len(c)
        solution = crowding_degree(solution)
        solution = dynamic_comparison_operator(classification, solution, population_size)
        generation = generation + 1

    end_3 = time.time()
    run_time_3.append(end_3 - start_3)
    f1_3.append(solution[0].func1)
    f2_3.append(solution[0].func2)
    f3_3.append(solution[0].func3)
    f4_3.append(solution[0].func4)
    f5_3.append(solution[0].func5)

mean_time_3 = sum(run_time_3) / (len(run_time_3))
mean_f1_3 = sum(f1_3) / (len(f1_3))
mean_f2_3 = sum(f2_3) / (len(f2_3))
mean_f3_3 = sum(f3_3) / (len(f3_3))
mean_f4_3 = sum(f4_3) / (len(f4_3))
mean_f5_3 = sum(f5_3) / (len(f5_3))

f1.append(f1_3)
f2.append(f2_3)
f3.append(f3_3)
f4.append(f4_3)
f5.append(f5_3)

print("mean_time:", end=' ')
print(mean_time_3)
print("mean_f1:", end=' ')
print(mean_f1_3)
print("mean_f2:", end=' ')
print(mean_f2_3)
print("mean_f3:", end=' ')
print(mean_f3_3)
print("mean_f4:", end=' ')
print(mean_f4_3)
print("mean_f7:", end=' ')
print(mean_f5_3)

print("-----------------------------------")
#动态拥挤度排序，动态交叉变异率
max_crossover_rate = 0.9
min_crossover_rate = 0.4
max_variation_rate = 0.1
min_variation_rate = 0.01
dynamic_crossover_rate = max_crossover_rate
dynamic_variation_rate = max_variation_rate

run_time_4 = []
f1_4 = []
f2_4 = []
f3_4 = []
f4_4 = []
f5_4 = []

print("----------动态拥挤度排序，动态交叉变异率---------")
for ti in range(0, times):
    start_4 = time.time()

    solution = [Solution(task) for i in range(0, population_size)]

    for i in range(0, population_size):
        solution[i] = Solution(task)
        solution[i].match(equipments, task)
        solution[i].record_state(equipments)
        solution[i].find_rank_number(assigned_scheme, task, material)
        solution[i].function1(task)
        solution[i].function2(task, present_time)
        solution[i].function3(equipments, task)
        solution[i].function4()
        solution[i].function5(task, equipments)

    original_diversity = population_diversity(solution)
    generation = 0
    while generation < iteration:
        solution = crossover(solution, dynamic_crossover_rate, dynamic_variation_rate, population_size, equipments, task)
        for i in range(population_size, len(solution)):
            solution[i].record_state(equipments)
            solution[i].find_rank_number(assigned_scheme, task, material)
            solution[i].function1(task)
            solution[i].function2(task, present_time)
            solution[i].function3(equipments, task)
            solution[i].function4()
            solution[i].function5(task, equipments)
        diversity = population_diversity(solution)
        if diversity == 1:
            break
        control = govern(solution)
        classification = fast_nondomination(control, solution)
        count = 0
        for c in classification:
            count = count + len(c)
        solution = crowding_degree(solution)
        solution = fixed_comparison_operator(classification, solution, population_size)
        dynamic_crossover_rate = adaptive_crossover_rate(generation, iteration, max_crossover_rate, min_crossover_rate)
        dynamic_variation_rate = adaptive_variation_rate(generation, iteration, max_variation_rate, min_variation_rate)
        generation = generation + 1

    end_4 = time.time()
    run_time_4.append(end_4 - start_4)
    f1_4.append(solution[0].func1)
    f2_4.append(solution[0].func2)
    f3_4.append(solution[0].func3)
    f4_4.append(solution[0].func4)
    f5_4.append(solution[0].func5)

mean_time_4 = sum(run_time_4) / (len(run_time_4))
mean_f1_4 = sum(f1_4) / (len(f1_4))
mean_f2_4 = sum(f2_4) / (len(f2_4))
mean_f3_4 = sum(f3_4) / (len(f3_4))
mean_f4_4 = sum(f4_4) / (len(f4_4))
mean_f5_4 = sum(f5_4) / (len(f5_4))

f1.append(f1_4)
f2.append(f2_4)
f3.append(f3_4)
f4.append(f4_4)
f5.append(f5_4)

print("mean_time:", end=' ')
print(mean_time_4)
print("mean_f1:", end=' ')
print(mean_f1_4)
print("mean_f2:", end=' ')
print(mean_f2_4)
print("mean_f3:", end=' ')
print(mean_f3_4)
print("mean_f4:", end=' ')
print(mean_f4_4)
print("mean_f7:", end=' ')
print(mean_f5_4)

print("-----------------------------------")


x = np.arange(1, 5)