import random
import pandas as pd
from ranking import *
import numpy as np

"""
    *随机任务的需求量最小为1000，最大为5000
"""
#计算任务所需时间
def calculate_product_time(product_quantity, production_serie):
    if production_serie == 'A':
        product_time = product_quantity/150 + 10
    elif production_serie == 'B':
        product_time = product_quantity/100 + 10
    elif production_serie == 'C':
        product_time = product_quantity/150 + 4.5
    elif production_serie == 'D':
        product_time = 4.5*product_quantity/1700 + 19.6
    elif production_serie == 'E':
        product_time = product_quantity/200 + 27
    elif production_serie == 'F':
        product_time = product_quantity/500 + 8
    elif production_serie == 'G':
        product_time = product_quantity/100 + 17
    else:
        product_time = product_quantity/200 + 10
    return product_time

s = 'LT_IT'
#随机生成任务
def randomly_generated_task(task_quanity, material_quanity):
    new_task = pd.DataFrame({'seria_number': [],
                                'total_production_capacity': [],
                                'production_series': [],
                                'delivery_date': [],
                                'need_material': [],
                                'product_cycle': []})

    seria_number = []
    total_production_capacity = []
    production_series = []
    delivery_date = []
    need_material = []
    product_cycle = []
    material = [0 for i in range(0, material_quanity)]
    seria = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    sl = len(seria)
    ml = len(material)
    for i in range(0, task_quanity):
        si = str(i+6)
        sni = s+si
        seria_number.append(sni)    #任务序号
        need = random.randint(10, 50)
        total_production_capacity.append(need)  #需求量
        seria_index = random.randint(0, sl-1)
        seria_i = seria[seria_index]
        production_series.append(seria_i)    #产品系列
        date = []
        year = random.randint(2020, 2021)
        if year == 2020:
            month = random.randint(3, 12)
            if month in [1, 3, 5, 7, 8, 10, 12]:
                day = random.randint(1, 31)
            else:
                day = random.randint(1, 30)

        else:
            month = random.randint(1, 12)
            if month == 2:
                day = random.randint(1, 28)
            elif month in [1, 3, 5, 7, 8, 10, 12]:
                day = random.randint(1, 31)
            else:
                day = random.randint(1, 30)
        date.append(year)
        date.append(month)
        date.append(day)
        date.append(24)
        delivery_date.append(date)      #交货期

        material_num = random.sample(range(1, ml), 1)
        material_kind = random.sample(range(0, ml-1), material_num[0])
        material_copy = material.copy()
        for mk in material_kind:
            count = random.randint(1, 1000)
            material_copy[mk] = count
        need_material.append(material_copy) #需要的物料

        ptime = calculate_product_time(need, seria_i)
        product_cycle.append(ptime)

    new_task['seria_number'] = seria_number
    new_task['total_production_capacity'] = total_production_capacity
    new_task['production_series'] = production_series
    new_task['delivery_date'] = delivery_date
    new_task['need_material'] = need_material
    new_task['product_cycle'] = product_cycle

    return new_task

#随机生成物料库存
def random_material(material_quanity):
    stock = pd.DataFrame({'Name_of_goods':[],
                    'inventory':[]})
    Name_of_goods = []
    inventory = []
    for i in range(0, material_quanity):
        Name_of_goods.append(i+1)
        inventory_num = random.randint(500, 20000)
        inventory.append(inventory_num)

    stock['Name_of_goods'] = Name_of_goods
    stock['inventory'] = inventory
    return stock

#输入待分配任务信息
def input_task(material_quanity):
    task_num = int(input("请输入添加任务的个数："))
    task = pd.DataFrame({'seria_number': [],
                         'total_production_capacity': [],
                         'production_series': [],
                         'delivery_date': [],
                         'need_material': [],
                         'product_cycle': []})

    seria_number = []
    total_production_capacity = []
    production_series = []
    delivery_date = []
    need_material = []
    product_cycle = []
    for i in range(0, task_num):
        seria_number_i = input("请输入任务号")
        total_production_capacity_i = int(input("请输入需求量："))
        production_series_i = input("请输入产品系列：")
        delivery_date_i = input("请输入交货期(以-间隔)：")
        need_material_num = int(input("需要物料的类别数："))
        material = [0 for j in range(0, material_quanity)]
        for k in range(0, need_material_num):
            kind = int(input("请输入所需物料的类别(数字)："))
            num = int(input("请输入所需物料的数量："))
            material[kind-1] = num
        ptime = calculate_product_time(total_production_capacity_i, production_series)
        seria_number.append(seria_number_i)
        total_production_capacity.append(total_production_capacity_i)
        production_series.append(production_series_i)
        delivery_date.append(delivery_date_i)
        need_material.append(material)
        product_cycle.append(ptime)

    task['seria_number'] = seria_number
    task['total_production_capacity'] = total_production_capacity
    task['production_series'] = production_series
    task['delivery_date'] = delivery_date
    task['need_material'] = need_material
    task['product_cycle'] = product_cycle

    return task

#输入设备信息
def input_equipment():
    equipment = pd.DateFrame({'number': [], 'state': [], 'production_capacity': [],
                   'Support_product_line':[],
                   'Maintenance_date':[]})
    number = []
    state = []
    production_capacity = []
    Support_product_line = []
    Maintenance_date = []
    equipment_quanity = int(input("请输入设备数："))
    for i in range(0, equipment_quanity):
        num = int(input("请输入设备号(数字)："))
        state_i= input("请输入设备的状态：")
        capacity = int(input("请输入设备的产能："))
        production_line = input("请输入支持的产品系列(用|间隔)：")
        date = input("请输入保养日期(以-间隔)：")
        number.append(num)
        state.append(state_i)
        production_capacity.append(capacity)
        Support_product_line.append(production_line)
        Maintenance_date.append(date)

    equipment['number'] = number
    equipment['state'] = state
    equipment['production_capacity'] = production_capacity
    equipment['Support_product_line'] = Support_product_line
    equipment['Maintenance_date'] = Maintenance_date

    return equipment

def input_weight():
    significance = input("请输入各目标函数的权重(0~1范围内，以英文逗号隔开，)：")
    significance = significance.split(",")
    weight = [0 for i in range(0, 5)]
    for i in range(0, 5):
        weight[i] = float(significance[i])

    return weight














