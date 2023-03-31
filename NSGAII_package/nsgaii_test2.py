import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from itertools import product
from itertools import combinations
import itertools
import sys
import time
from ranking import *
from random_task import *



class Solution:
    def __init__(self, next_task):
        self.task = next_task['seria_number'].values.tolist()
        self.machine_use = []  # 设备分配
        self.record = []  # 记录任务分配设备时设备的状态
        self.rank_number = []  # 排位号
        self.duration_time = []  # 每个排位号任务集的最大结束时间
        self.total_product_time = 0  # 总的生产时间
        self.lack = []  # 存放任务的缺乏情况
        self.assigned_time = []
        self.func1 = 0
        self.func2 = 0
        self.func3 = 0
        self.func4 = 0
        self.func5 = 0
        self.distance = 0  # 距离
        self.end = []
        self.start = []


    # 最大完工时间
    def function1(self, next_task):
        product_time = self.assigned_time
        rank_num = self.rank_number
        duration_time = []
        """ptime = product_time[0]
        for i in range(1, len(rank_num)):
            if rank_num[i] == rank_num[i-1]:
                if product_time[i] > product_time[i-1]:
                    ptime = product_time[i]
            else:
                if self.record[i] == 5:
                    ptime
                duration_time.append(ptime)
                time = product_time[i]"""
        rank_class = []
        rank = min(rank_num)
        max_rank = max(rank_num)
        while rank <= max_rank:
            rank_list = []
            for i in range(0, len(self.rank_number)):
                if self.rank_number[i] == rank:
                    rank_list.append(i)
            rank = rank + 1
            rank_class.append(rank_list)

        for rc in rank_class:
            ptime = []
            for rci in rc:
                if self.record[rci] == 5:
                    ptime.append(10000)
                    break
                else:
                    ptime.append(next_task['product_cycle'][rci])
            if ptime != []:
                maxtime = max(ptime)
            else:
                maxtime = 0
            duration_time.append(maxtime)

        self.duration_time = duration_time
        self.func1 = sum(duration_time)
        return Solution

    # 统计订单延误数
    def function2(self, next_task, present_time):
        count = 0
        end_time = []
        start = []
        rank = self.rank_number[0]
        for i in range(0, len(self.rank_number)):
            d = self.rank_number[i] - rank
            if d == 0:
                time = next_task['product_cycle'][i]
                start.append(0)
            else:
                wait_time = 0
                for j in range(0, d):
                    wait_time = wait_time + self.duration_time[j]

                start.append(wait_time)
                time = wait_time + next_task['product_cycle'][i]
            end_time.append(time)
        self.start = start
        self.end = end_time
        deadline = next_task['delivery_date'].values.tolist()
        for i in range(0, len(end_time)):
            ddl = date_distance(present_time, deadline[i])
            if end_time[i] > ddl:
                count = count + 1
        self.func2 = count
        return Solution

    # 判断设备分配情况（得分）
    def function3(self, equipments, next_task):
        e_point = 0
        c_point = 0
        i = 0
        e_list = []
        c_list = []
        production_lines = equipments['Support_product_line'].values.tolist()
        for m in self.machine_use:
            kind = next_task['production_series'].values.tolist()[i]  # 产品系列
            ep = []
            cp = []
            for mi in m:
                if equipments['state'][mi - 1] == 0:  # 分配设备处于空闲状态
                    ep.append(10)
                elif equipments['state'][mi - 1] == 1:  # 分配设备处于待产中状态
                    ep.append(8)
                elif equipments['state'][mi - 1] == 2:
                    ep.append(6)
                elif equipments['state'][mi - 1] == 3:
                    ep.append(4)
                elif equipments['state'][mi - 1] == 4:
                    ep.append(2)
                else:  # 处于其他状态
                    ep.append(-10)

                if kind in production_lines[mi - 1]:  # 符合产品系列
                    cp.append(100)
                else:
                    cp.append(0)
            i = i + 1
            e_list.append(min(ep))
            c_list.append(min(cp))
        e_point = sum(e_list)
        c_point = sum(c_list)
        point = 110 * len(self.machine_use) - e_point - c_point
        self.func5 = point
        return Solution

    # 所有订单使用的机器数
    def function4(self):
        count = 0
        for item in self.machine_use:
            l = len(item)
            count = count + l
        self.func6 = count
        return Solution

    # 计算产能溢出
    def function5(self, next_task, equipments):
        sum = 0
        i = 0
        capacity = equipments['production_capacity'].values.tolist()
        max_capacity = max(capacity)
        for m in self.machine_use:
            capacity = 0
            need = next_task['total_production_capacity'][i]
            for mi in m:
                capacity = capacity + equipments['production_capacity'][mi - 1]
            overflow = capacity - need
            sum = sum + overflow
            i = i + 1
        self.func7 = sum / max_capacity
        return Solution

    def find_rank_number(self, assigned_scheme, next_task, stock):
        state = assigned_scheme['state'].values.tolist()  # 已分配任务使用设备的状态
        delivery_date = assigned_scheme['delivery_date'].tolist()  # 已分配任务交货期
        number = assigned_scheme['number'].values.tolist()  # 已分配任务使用的排位号
        machine_use = assigned_scheme['machine_use'].values.tolist()  # 已分配任务使用的设备
        ptime = assigned_scheme['product_cycle'].values.tolist()
        rank_number = [0 for i in range(0, len(next_task['seria_number']))]  # 待分配任务的排位号
        assigned_date = []  # 已分配待产中任务的交货期
        assigned_num = []  # 已分配待产中任务的排位号

        assigned_machine = []  #
        assigned_state = []
        assigned_time = []
        lack = [0 for i in range(0, len(next_task['seria_number']))]
        for i in range(0, len(state)):
            if state[i] == 1:
                date = delivery_date[i]
                rank = number[i]
                machine = machine_use[i]
                state_i = state[i]
                time_i = ptime[i]
                assigned_date.append(date)
                assigned_num.append(rank)
                assigned_machine.append(machine)
                assigned_state.append(state_i)
                assigned_time.append(time_i)
        need = next_task['need_material'].tolist()  # 需求量
        inventory = stock['inventory'].tolist()  # 库存
        for j in range(0, len(need)):
            for ni in range(0, len(need[j])):
                if need[j][ni] > inventory[ni]:
                    lack[j] = 1
                    break
                else:
                    surplus = inventory[ni] - need[j][ni]
                    inventory[ni] = surplus
        lack_task = []  # 存放物料不足任务
        rubbish = []  # 存放使用停用设备任务
        for m in range(0, len(self.record)):  # m是任务对应的索引
            if lack[m] == 0:  # 物料能满足
                mu = self.machine_use[m]  # 待分配任务使用的设备
                count = 0  # 和已分配任务使用相同设备的次数
                similar = []
                for i in range(0, len(assigned_machine)):  # 寻找
                    for mui in mu:
                        if mui in assigned_machine[i]:
                            count = count + 1
                            similar.append(i)
                            break
                if count == 0:  # 如果使用停用设备，肯定不存在使用相同设备的任务
                    flag = 1
                    for ri in self.record[m]:
                        if ri == 5:
                            flag = 0
                            rubbish.append(m)
                            break
                    if flag == 1:
                        smaller = 0
                        index = 0
                        al = len(assigned_date)
                        for i in range(0, al):
                            if date_cmoparsion(next_task['delivery_date'][m], assigned_date[i]) == 1:
                                smaller = 1
                                index = i
                        if smaller == 0:  # 交货期最早,找不到更早的
                            r_num = min(assigned_num)
                            rank_number[m] = r_num
                            assigned_num.insert(0, r_num)
                            assigned_date.insert(0, next_task['delivery_date'][m])
                            assigned_machine.insert(0, mu)
                            assigned_time.insert(0, next_task['product_cycle'][m])
                        else:
                            if index == al - 1:  # 交货期最晚
                                r_num = max(assigned_num)
                                rank_number[m] = r_num
                                assigned_num.append(r_num)
                                assigned_machine.append(mu)
                                assigned_date.append(next_task['delivery_date'][m])
                                assigned_time.append(next_task['product_cycle'][m])
                            else:  # 交货期处于中间
                                r_num = assigned_num[index]
                                rank_number[m] = r_num
                                assigned_num.insert(index + 1, r_num)
                                assigned_machine.insert(index + 1, mu)
                                assigned_date.insert(index + 1, next_task['delivery_date'][m])
                                assigned_time.insert(index + 1, next_task['product_cycle'][m])

                    """else:
                        r_num = max(assigned_num) + 1
                        rank_number.append(r_num)
                        assigned_num.append(r_num)
                        assigned_machine.append(mu)
                        assigned_date.append(next_task['delivery_date'][m])
                        assigned_time.append(next_task['product_cycle'][m])
                        rubbish.append(len(assigned_num) - 1)"""
                else:  # 存在使用相同设备
                    s_smaller = 0
                    s_index = -1
                    real_similar = []
                    for si in similar:
                        if si not in lack_task:
                            real_similar.append(si)
                        if si in rubbish:
                            real_similar = []
                            break
                    if len(real_similar) != 0:
                        for i in real_similar:
                            if date_cmoparsion(next_task['delivery_date'][m], assigned_date[i]) == 1:
                                s_smaller = 1
                                s_index = i
                        if s_smaller == 0:
                            for rs in real_similar:
                                assigned_num[rs] = assigned_num[rs] + 1
                            smaller = 0
                            index = -1
                            for i in range(0, len(assigned_date)):
                                if date_cmoparsion(next_task['delivery_date'][m], assigned_date[i]) == 1:
                                    smaller = 1
                                    index = i
                            if smaller == 0:  # 交货期最早,找不到更早的
                                r_num = min(assigned_num)
                                rank_number[m] = r_num
                                assigned_num.insert(0, r_num)
                                assigned_date.insert(0, next_task['delivery_date'][m])
                                assigned_machine.insert(0, mu)
                                assigned_time.insert(0, next_task['product_cycle'][m])
                            else:
                                r_num = assigned_num[index]
                                rank_number[m] = r_num
                                assigned_num.insert(index + 1, r_num)
                                assigned_machine.insert(index + 1, mu)
                                assigned_date.insert(index + 1, next_task['delivery_date'][m])
                                assigned_time.insert(index + 1, next_task['product_cycle'][m])
                        else:
                            if s_index == real_similar[len(real_similar) - 1]:  # 是使用相同设备里面交货期最晚的
                                index = -1
                                for i in range(0, len(assigned_date)):
                                    if date_cmoparsion(next_task['delivery_date'][m], assigned_date[i]) == 1:
                                        index = i
                                if assigned_num[index] != assigned_num[s_index]:
                                    r_num = assigned_num[index]
                                    rank_number[m] = r_num
                                    assigned_num.insert(index + 1, r_num)
                                    assigned_machine.insert(index + 1, mu)
                                    assigned_date.insert(index + 1, next_task['delivery_date'][m])
                                    assigned_time.insert(index + 1, next_task['product_cycle'][m])
                                else:
                                    r_num = assigned_num[index] + 1
                                    rank_number[m] = r_num
                                    assigned_num.insert(index + 1, r_num)
                                    assigned_machine.insert(index + 1, mu)
                                    assigned_date.insert(index + 1, next_task['delivery_date'][m])
                                    assigned_time.insert(index + 1, next_task['product_cycle'][m])
                            else:
                                weizhi = -1
                                for k in range(0, len(real_similar)):
                                    if real_similar[k] == s_index:
                                        weizhi = k
                                        break
                                index = 0
                                for i in range(0, len(assigned_date)):
                                    if date_cmoparsion(next_task['delivery_date'][m], assigned_date[i]) == 1:
                                        index = i
                                if assigned_num[index] != assigned_num[real_similar[-1]]:
                                    for k in range(weizhi + 1, len(real_similar)):
                                        assigned_num[k] = assigned_num[k] + 1
                                    r_num = assigned_num[index]
                                    rank_number[m] = r_num
                                    assigned_num.insert(index + 1, r_num)
                                    assigned_machine.insert(index + 1, mu)
                                    assigned_date.insert(index + 1, next_task['delivery_date'][m])
                                    assigned_time.insert(index + 1, next_task['product_cycle'][m])

                                else:
                                    for k in range(weizhi + 1, len(real_similar)):
                                        assigned_num[k] = assigned_num[k] + 1
                                    r_num = assigned_num[index] + 1
                                    rank_number[m] = r_num
                                    assigned_num.insert(index + 1, r_num)
                                    assigned_machine.insert(index + 1, mu)
                                    assigned_date.insert(index + 1, next_task['delivery_date'][m])
                                    assigned_time.insert(index + 1, next_task['product_cycle'][m])
                    else:
                        r_num = max(assigned_num) + 1
                        rank_number[m] = r_num
                        assigned_num.append(r_num)
                        assigned_machine.append(mu)
                        assigned_date.append(next_task['delivery_date'][m])
                        assigned_time.append(next_task['product_cycle'][m])
            else:
                lack_task.append(m)
        # 物料不足的任务进行排序
        lack_sorted = []
        lack_machine = []
        la = len(lack_task)
        ls = len(lack_sorted)
        min_l = 0
        while ls != la:
            for li in lack_task:
                if date_cmoparsion(next_task['delivery_date'][lack_task[min_l]],
                                   next_task['delivery_date'][li]) != -1:  # < min
                    min_l = li
                    lack_sorted.append(min_l)
                    lack_machine.append(self.machine_use[min_l])
                    ls = len(lack_sorted)
        for lsi in range(0, len(lack_sorted)):
            if lsi == 0:
                mu = self.machine_use[lack_sorted[lsi]]
                r_num = max(assigned_num) + 1
                rank_number[lack_sorted[lsi]] = r_num
                assigned_num.append(r_num)
                assigned_machine.append(mu)
                assigned_date.append(next_task['delivery_date'][lack_sorted[lsi]])
                assigned_time.append(next_task['product_cycle'][lack_sorted[lsi]])
            else:
                for lsk in range(0, lsi):
                    mu = self.machine_use[lack_sorted[lsi]]  # 当前任务使用的设备
                    mu_before = self.machine_use[lack_sorted[lsk]]  # 之前的任务使用的设备
                    count = 0
                    for mui in mu:
                        if mui in mu_before:
                            count = count + 1
                            break
                    r_num = max(assigned_num) + count
                    rank_number[lack_sorted[lsi]] = r_num
                    assigned_num.append(r_num)
                    assigned_machine.append(mu)
                    assigned_date.append(next_task['delivery_date'][lack_sorted[lsi]])
                    assigned_time.append(next_task['product_cycle'][lack_sorted[lsi]])
        # 使用停用设备的任务直接添加到末尾
        for ri in rubbish:
            mu = self.machine_use[ri]
            r_num = max(assigned_num) + 1
            rank_number[ri] = r_num
            assigned_num.append(r_num)
            assigned_machine.append(mu)
            assigned_date.append(next_task['delivery_date'][ri])
            assigned_time.append(next_task['product_cycle'][ri])

        self.assigned_time = assigned_time
        self.lack = lack
        self.rank_number = rank_number
        return Solution

    def match(self, equipments, next_task):
        state = equipments['state'].values.tolist()
        el = len(state)
        capacity = equipments['production_capacity'].values.tolist()
        sl = len(self.task)
        demand = next_task['total_production_capacity'].values.tolist()
        machine_use = []
        for i in range(0, sl):
            used = []
            jilu = []
            num = equipments['number'].values.tolist()
            shengyu = demand[i]
            while shengyu > 0:
                index = random.sample(range(0, el), 1)
                shengyu = shengyu - capacity[index[0]]
                e_num = num[index[0]]
                jilu.append(state[e_num - 1])
                if state[e_num - 1] == 0:
                    state[e_num - 1] = 1
                used.append(e_num)
            machine_use.append(used)
        self.machine_use = machine_use
        return Solution

    # 记录分配设备前设备的状态
    def record_state(self, equipments):
        record = []
        state = equipments['state'].values.tolist()
        sl = len(self.machine_use)
        for i in range(0, sl):
            for mi in self.machine_use[i]:
                record_i = []
                record_i.append(state[mi - 1])
                if state[mi - 1] == 0:  # 设备处于空闲
                    state[mi - 1] = 1  # 修改设备状态，改为待产中
            record.append(record_i)
        self.record = record
        return Solution

    # 判断两个对象的值是否相同
    def is_equal(self, other):
        if (self.rank_number == other.rank_number) and (self.machine_use == other.machine_use):
            return True
        else:
            return False


# 计算种群多样性
def population_diversity(solution):
    record = []
    for si in range(0, len(solution) - 1):
        record_i = []
        for sk in range(si + 1, len(solution)):
            if solution[si].is_equal(solution[sk]):
                record_i.append(sk)
        record.append(record_i)
    record_copy = record.copy()
    rl = len(record_copy)
    for ri in range(0, rl - 1):
        if set(record_copy[rl - 1 - ri]) <= set(record_copy[rl - ri - 2]):
            record.remove(record_copy[rl - 1 - ri])
    count = len(record)
    return count


# 判断日期大小，< 返回-1，=返回0，> 返回1
def date_cmoparsion(date1, date2):
    if date1[0] == date2[0]:
        if date1[1] == date2[1]:
            if date1[2] == date2[2]:
                if date1[3] == date2[3]:
                    return 0
                elif date1[3] < date2[3]:
                    return -1
                else:
                    return 1
            elif date1[2] < date2[2]:
                return -1
            else:
                return 1
        elif date1[1] < date2[1]:
            return -1
        else:
            return 1
    elif date1[0] < date2[0]:
        return -1
    else:
        return 1

def govern(solution):
    control = []
    for i in range(0, len(solution) - 1):
        for j in range(i + 1, len(solution)):
            if solution[i].func1 == solution[j].func1:
                if solution[i].func2 == solution[j].func2:
                    if solution[i].func3 == solution[j].func3:
                        if solution[i].func4 == solution[j].func4:
                            if solution[i].func5 < solution[j].func5:
                                control.append([i, j])
                            elif solution[i].func5 > solution[j].func5:
                                control.append([j, i])
                        elif solution[i].func4 < solution[j].func4:
                            if solution[i].func5 <= solution[j].fun5:
                                control.append([i, j])
                        else:
                            if solution[i].func5 >= solution[j].fun5:
                                control.append([j, i])
                    elif solution[i].func3 < solution[j].func3:
                        if solution[i].func4 <= solution[j].func4:
                            if solution[i].func5 <= solution[j].func5:
                                control.append([i, j])
                        else:
                            continue
                    else:
                        if solution[i].func4 >= solution[j].func4:
                            if solution[i].func5 >= solution[j].func5:
                                control.append([j, i])
                        else:
                            continue
                elif solution[i].func2 < solution[j].func2:
                    if solution[i].func3 <= solution[j].func3:
                        if solution[i].func4 <= solution[j].func4:
                            if solution[i].func5 <= solution[j].func5:
                                control.append([i, j])
                        else:
                            continue
                    else:
                        continue
                else:
                    if solution[i].func3 >= solution[j].func3:
                        if solution[i].func4 >= solution[j].func4:
                            if solution[i].func5 >= solution[j].func5:
                                control.append([j, i])
                        else:
                            continue
                    else:
                        continue
            elif solution[i].func1 < solution[j].func1:
                if solution[i].func2 <= solution[j].func2:
                    if solution[i].func3 <= solution[j].func3:
                        if solution[i].func4 <= solution[j].func4:
                            if solution[i].func5 <= solution[j].func5:
                                control.append([i, j])
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                if solution[i].func2 >= solution[j].func2:
                    if solution[i].func3 >= solution[j].func3:
                        if solution[i].func4 >= solution[j].func4:
                            if solution[i].func5 >= solution[j].func5:
                                control.append([j, i])
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
    return control


# solution_i 是Solution的集合
def fast_nondomination(control, solution_i):
    np = []  # 支配个体p的个数
    sp = []  # 被个体p支配的集合, sp[0]表示被solutuon[0]支配的个体数
    solutioni = solution_i[:]
    for index in range(0, len(solutioni)):
        count = 0
        s = []
        for c in control:
            if c[1] == index:  # p被支配
                count = count + 1
            if c[0] == index:  # 被p支配
                s.append(c[1])

        np.append(count)
        sp.append(s)

    classification = []
    first = []
    for i in range(0, len(np)):
        if np[i] == 0:
            first.append(i)
            solutioni.remove(solution_i[i])
    classification.append(first)

    f = first
    while len(solutioni) > 0:
        s = []
        h = []
        for index in f:
            s.append(sp[index])  # s存放被sp[index]支配的个体集合
        for si in s:
            for sii in si:
                np[sii] = np[sii] - 1
                if np[sii] == 0:
                    h.append(sii)
                    if solution_i[sii] in solutioni:  # 存在一个个体被多个个体支配的情况，避免多删
                        solutioni.remove(solution_i[sii])

        classification.append(h)
        f = h

    return classification


# solution表示Solution的集合
def crowding_degree(solution):
    l = len(solution)
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}
    dict5 = {}

    f1_sorted = []
    f2_sorted = []
    f3_sorted = []
    f4_sorted = []
    f5_sorted = []

    for i in range(0, l):
        dict1[i] = solution[i].func1
        dict2[i] = solution[i].func2
        dict3[i] = solution[i].func3
        dict4[i] = solution[i].func4
        dict5[i] = solution[i].func5

    dict1 = sorted(dict1.items(), key=lambda k: k[1])
    dict2 = sorted(dict2.items(), key=lambda k: k[1])
    dict3 = sorted(dict3.items(), key=lambda k: k[1])
    dict4 = sorted(dict4.items(), key=lambda k: k[1])
    dict5 = sorted(dict5.items(), key=lambda k: k[1])

    for i in dict1:  # dict1是一个列表，元素为一个元组(key, value)
        f1_sorted.append(i[0])
    for i in dict2:
        f2_sorted.append(i[0])
    for i in dict3:
        f3_sorted.append(i[0])
    for i in dict4:
        f4_sorted.append(i[0])
    for i in dict5:
        f5_sorted.append(i[0])

    f = [f1_sorted, f2_sorted, f3_sorted, f4_sorted, f5_sorted]
    i = 0
    for fi in f:
        solution[fi[0]].distance = float("inf")
        solution[fi[-1]].distance = float("inf")
        if i == 0:
            for index in range(1, len(fi) - 1):
                if (dict1[-1][1] - dict1[0][1]) != 0:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                                solution[fi[index + 1]].func1 - solution[fi[index - 1]].func1) / (
                                                               dict1[-1][1] - dict1[0][1])
                else:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                                solution[fi[index + 1]].func1 - solution[fi[index - 1]].func1)
        elif i == 1:
            for index in range(1, len(fi) - 1):
                if (dict2[-1][1] - dict2[0][1]) != 0:
                    solution[fi[index]].distance = solution[fi[index]].distance + (solution[fi[index + 1]].func2 - solution[fi[index - 1]].func2) / (dict2[-1][1] - dict2[0][1])
                else:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                                solution[fi[index + 1]].func2 - solution[fi[index - 1]].func2)
        elif i == 2:
            for index in range(1, len(fi) - 1):
                if (dict3[-1][1] - dict3[0][1]) != 0:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                            solution[fi[index + 1]].func3 - solution[fi[index - 1]].func3) / (
                                                           dict3[-1][1] - dict3[0][1])
                else:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                            solution[fi[index + 1]].func3 - solution[fi[index - 1]].func3)
        elif i == 3:
            for index in range(1, len(fi) - 1):
                if (dict4[-1][1] - dict4[0][1]) != 0:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                            solution[fi[index + 1]].func4 - solution[fi[index - 1]].func4) / (
                                                           dict4[-1][1] - dict4[0][1])
                else:
                    solution[fi[index]].distance = solution[fi[index]].distance + (solution[fi[index + 1]].func4 - solution[fi[index - 1]].func4)
        elif i == 4:
            for index in range(1, len(fi) - 1):

                if (dict5[-1][1] - dict5[0][1]) != 0:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                                solution[fi[index + 1]].func5 - solution[fi[index - 1]].func5) / (
                                                               dict5[-1][1] - dict5[0][1])
                else:
                    solution[fi[index]].distance = solution[fi[index]].distance + (
                                solution[fi[index + 1]].func5 - solution[fi[index - 1]].func5)
        i = i + 1

    return solution




def crossover(solution, crossover_rate, variation_rate, population_size, equipments, next_task):
    l = len(solution)
    sl = len(solution[0].task)
    size = population_size
    new_solution = []
    while size < 2 * population_size:
        for i in range(0, l):
            init_rate = random.uniform(0, 1)
            rate = round(init_rate, 2)
            if rate < crossover_rate:
                cross = random.sample(range(0, l), 2)
                s = solution[cross[0]]
                cross_s = solution[cross[1]]

                cross_num = random.sample(range(1, sl + 1), 1)  # 交叉位置的数目
                cross_index = random.sample(range(0, sl), cross_num[0])  # 交叉位置
                for ci in cross_index:
                    s.machine_use[ci] = cross_s.machine_use[ci]
                s = variation(s, variation_rate, equipments, next_task)
                new_solution.append(s)
                size = size + 1
            else:
                index = random.sample(range(0, l), 1)
                s = solution[index[0]]
                s = variation(s, variation_rate, equipments, next_task)
                new_solution.append(s)
                size = size + 1
    solution.extend(new_solution)
    return solution


def variation(s, variation_rate, equipments, next_task):
    l = len(s.task)
    capacity = equipments['production_capacity'].values.tolist()
    state = equipments['state'].tolist()
    need = next_task['total_production_capacity'].values.tolist()
    init_rate = random.uniform(0, 1)
    rate = round(init_rate, 2)
    if rate < variation_rate:
        num = random.sample(range(1, l + 1), 1)
        d = random.sample(range(0, l), num[0])
        for di in d:
            used = []
            jilu = []
            num = equipments['number'].values.tolist()
            shengyu = need[di]
            while shengyu > 0:
                index = random.sample(range(0, len(num)), 1)
                shengyu = shengyu - capacity[index[0]]
                jilu.append(state[num[index[0]] - 1])
                if state[num[index[0]] - 1] == 0:
                    state[num[index[0]] - 1] = 1
                used.append(num[index[0]])
                num.remove(num[index[0]])
            s.machine_use[di] = used
    return s


def adaptive_crossover_rate(present_generation, max_generation, max_rate, min_rate):
    crossover_rate = max_rate - (max_rate - min_rate) * present_generation / max_generation
    return crossover_rate


def adaptive_variation_rate(present_generation, max_generation, max_rate, min_rate):
    variation_rate = max_rate - (max_rate - min_rate) * present_generation / max_generation
    return variation_rate


def crowded_comparison_operator(classification, solution, population_size):
    new_population = []
    count = 0
    for c in classification:
        l = len(c)
        haicha = population_size - count
        if l <= haicha:
            count = count + l
            for ci in c:
                new_population.append(solution[ci])
        else:
            s = len(c)
            while s != haicha:
                d = {}
                for ci in c:
                    d[ci] = solution[ci].distance

                d = sorted(d.items(), key=lambda item: item[1]) #d的元素是一个元组(solution_index, distance_i)
                deleted_index = d[0][0]     #要删除任务的索引
                c.remove(deleted_index)
                del solution[deleted_index]
                for c_i in range(0, len(c)):
                    if c[c_i] > deleted_index:
                        c[c_i] = c[c_i] - 1
                solution = crowding_degree(solution)
                s = s - 1
            for cii in c:
                new_population.append(solution[cii])
            break

    return new_population


def calculate_product_time(product_quantity, production_serie):
    if production_serie == 0:
        product_time = product_quantity / 150 + 10
    elif production_serie == 1:
        product_time = product_quantity / 100 + 10
    elif production_serie == 2:
        product_time = product_quantity / 150 + 4.5
    elif production_serie == 3:
        product_time = 4.5 * product_quantity / 1700 + 19.6
    elif production_serie == 4:
        product_time = product_quantity / 200 + 27
    elif production_serie == 5:
        product_time = product_quantity / 500 + 8
    elif production_serie == 6:
        product_time = product_quantity / 100 + 17
    else:
        product_time = product_quantity / 200 + 10
    return product_time


def year_month_date(year, month, date, hour, time):
    new_year = year
    new_month = month
    new_date = date

    if hour + time < 24:
        new_hour = hour + time
        return [new_year, new_month, new_date, new_hour]
    else:
        day = (hour + time) // 24
        new_hour = (hour + time) % 24
        if month in [1, 3, 5, 7, 8, 10, 12]:
            if date + day > 31:
                new_month = new_month + 1
                new_date = date + day - 31
            else:
                new_date = date + day
        elif month in [4, 6, 9, 11]:
            if date + day > 30:
                new_month = new_month + 1
                new_date = date + day - 30
            else:
                new_date = date + day
        else:
            if year % 100 == 0:
                if year % 400 == 0:
                    if date + day > 29:
                        new_month = new_month + 1
                        new_date = date + day - 29
                    else:
                        new_date = date + day
            else:
                if year % 4 == 0:
                    if date + day > 29:
                        new_month = new_month + 1
                        new_date = date + day - 29
                    else:
                        new_date = date + day
                else:
                    if date + day > 28:
                        new_month = new_month + 1
                        new_date = date + day - 28
                    else:
                        new_date = date + day

        if new_month > 12:
            new_month = new_month - 1
            new_year = new_year + 1
        return [new_year, new_month, new_date, new_hour]


"""
    *算法基于以下假设：
        **任务排序基于最早交货期排序规则(先将任务按交货时间从早到晚进行排序)
        **当前时间为2020-02-26，任务最早可于0点开始生产
        **待产中的任务从2-26 0点开始生产
        **使用待产中的设备从待产任务结束时开始，中间无等待时间
        **物料不足的任务最早可从2-27号开始（需要时间补充物料）
"""
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


def date_digitization(date):
    string_date = date.split("-")
    digital_date = [0 for i in range(0, len(string_date))]
    for i in range(0, len(string_date)):
        digital_date[i] = int(string_date[i])

    return digital_date


def seriaTonumber(seria):
    if seria == 'A':
        return 0
    elif seria == 'B':
        return 1
    elif seria == 'C':
        return 2
    elif seria == 'D':
        return 3
    elif seria == 'E':
        return 4
    elif seria == 'F':
        return 5
    elif seria == 'G':
        return 6
    else:
        return 7


# 对机器映射编码
def equip_coding(equipments):
    mapping_state = {'leisure': 0, 'wait_for_production': 1, 'in_production': 2, 'wait_for_the_packing': 3,
                     'wait_for_quality_inspection': 4, 'stop': 5}
    equipments['state'] = equipments['state'].map(mapping_state)
    production_line = equipments['Support_product_line'].values.tolist()
    line = []
    for pl in production_line:
        pl = pl.split("|")
        for i in range(0, len(pl)):
            pl[i] = seriaTonumber(pl[i])
        line.append(pl)
    equipments['Support_product_line'] = line
    return equipments


def assigned_coding(assigned_scheme):
    assigned_Product_series = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    assigned_scheme['production_series'] = assigned_scheme['production_series'].map(assigned_Product_series)
    delivery_date = assigned_scheme['delivery_date'].values.tolist()
    for di in range(0, len(delivery_date)):
        date = date_digitization(delivery_date[di])
        date.append(24)
        delivery_date[di] = date
    assigned_scheme['delivery_date'] = delivery_date
    state = {'leisure': 0, 'wait_for_production': 1, 'in_production': 2, 'wait_for_the_packing': 3,
             'wait_for_quality_inspection': 4, 'stop': 5}
    assigned_scheme['state'] = assigned_scheme['state'].map(state)
    product_cycle = []
    for i in range(0, len(assigned_scheme['production_quantity'])):
        product_time = calculate_product_time(assigned_scheme['production_quantity'][i],
                                              assigned_scheme['production_series'][i])
        product_cycle.append(product_time)
    assigned_scheme['product_cycle'] = product_cycle
    return assigned_scheme


# 对将要执行的任务进行编码
def task_coding(next_task):
    # next_task_seria_number = {'LT_IT030': 0, 'LT_IT032': 1, 'LT_IT033': 2, 'LT_IT034': 3, 'LT_IT035': 4, 'LT_IT036': 5}
    # next_task['seria_number'] = next_task['seria_number'].map(next_task_seria_number)
    next_task_Product_series = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    next_task['production_series'] = next_task['production_series'].map(next_task_Product_series)  # 对于任务生产的物资进行映射
    delivery_date = next_task['delivery_date'].values.tolist()
    for di in range(0, len(delivery_date)):
        date = date_digitization(delivery_date[di])
        date.append(24)
        delivery_date[di] = date
    next_task['delivery_date'] = delivery_date
    return next_task


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


#初始化种群
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
#print(equipments['state'])
#print(equipments['Support_product_line'])
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
        solution[i].function2(next_task, present_time)
        solution[i].function3(equipments, next_task)
        solution[i].function4()
        solution[i].function5(next_task, equipments)

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
print(solution[0].func2)
print(solution[0].rank_number)
print(solution[0].machine_use)
print("start:", end=' ')
print(solution[0].start)
print("end:", end=' ')
print(solution[0].end)
x = np.arange(1, generation+1, 1)

plt.xlabel("iteration")  #x轴上的名字
plt.ylabel("Solution Diverity") #y轴上的名字
plt.plot(x, y, color='green', linewidth=5)
plt.show()