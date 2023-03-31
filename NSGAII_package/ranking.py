import pandas as pd
import numpy as py
import random

task = pd.DataFrame({'seria_number':['LT_IT030', 'LT_IT032', 'LT_IT033', 'LT_IT034', 'LT_IT035', 'LT_IT036'],
                        'total_production_capacity':[1500, 1000, 500, 1300, 1800, 3000],
                        'production_series':['C', 'G', 'F', 'D', 'E', 'D'],
                        'delivery_date':['2020-03-19','2020-03-01','2020-03-01','2020-03-06','2020-02-29','2020-03-01'],
                        'need_material':[[500,0,0,0,300,0,0,0,150,0,400,0,0,0,0,0,0,0,0,150], [0,400,300,290,0,0,0,0,0,0,0,0,0,0,0,0,3,5,2,0],
                                         [0,150,200,0,0,0,0,0,0,0,0,0,120,0,0,0,30,0,0,0], [0,0,0,0,200,0,0,0,0,0,0,0,0,600,500,0,0,0,0,0],
                                         [600,600,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [300,400,100,500,200,1000,500,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                        'product_cycle':[14.5, 27, 9, 23, 36, 27.5]})

def date_digitization(date):
    string_date = date.split("-")
    digital_date = [0 for i in range(0, len(string_date))]
    for i in range(0, len(string_date)):
        digital_date[i] = int(string_date[i])

    return digital_date
#判断日期大小，< 返回-1，=返回0，> 返回1
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

#冒泡算法对任务按交货期从早到晚进行排序
def bubble_sorted_task(task):
    sorted_task = pd.DataFrame({'seria_number': [],
                                'total_production_capacity': [],
                                'production_series': [],
                                'delivery_date': [],
                                'need_material': [],
                                'product_cycle': []})
    delivery_date = task['delivery_date'].values.tolist()
    total_production_capacity = task['total_production_capacity'].values.tolist()
    production_series = task['production_series'].values.tolist()
    seria_number = task['seria_number'].values.tolist()
    need_material = task['need_material'].values.tolist()
    product_cycle = task['product_cycle'].values.tolist()

    l = len(task['seria_number'])
    for i in range(0, l-1):
        count = 0
        for j in range(0, l-1-i):
            if date_cmoparsion(delivery_date[j], delivery_date[j+1]) == 1: #date1 > date2
                delivery_date[j], delivery_date[j+1] = delivery_date[j+1], delivery_date[j]
                seria_number[j], seria_number[j+1] = seria_number[j+1], seria_number[j]
                total_production_capacity[j], total_production_capacity[j+1] = total_production_capacity[j+1], total_production_capacity[j]
                production_series[j], production_series[j+1] = production_series[j+1], production_series[j]
                need_material[j], need_material[j+1] = need_material[j+1], need_material[j]
                product_cycle[j], product_cycle[j+1] = product_cycle[j+1], product_cycle[j]
                count = count + 1
        if count == 0:
            break

    sorted_task['seria_number'] = seria_number
    sorted_task['total_production_capacity'] = total_production_capacity
    sorted_task['production_series'] = production_series
    sorted_task['delivery_date'] = delivery_date
    sorted_task['need_material'] = need_material
    sorted_task['product_cycle'] = product_cycle

    return sorted_task

#计算date2和date1之间相差多少小时
def date_distance(date1, date2):
    month_one = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_two = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if date2[0] == date1[0]:    #年份相同
        if date2[0] % 100 != 0: #不是100的倍数
            if date2[0] % 4 != 0:   #不是4的倍数
                if date2[1] != date1[1]:    #月份不同
                    day = month_one[date1[1] - 1] - date1[2]    #到月底的距离
                    for i in range(date1[1], date2[1]-1):
                        day = day + month_one[i]                #间隔几个月的天数
                    day = day + date2[2]                        #离月初的距离
                else:
                    day = date2[2] - date1[2]
                hour = day*24 + date2[3] - date1[3]
            else:
                if date2[1] != date1[1]:
                    day = month_two[date1[1] - 1] - date1[2]
                    for i in range(date1[1], date2[1]-1):
                        day = day + month_two[i]
                    day = day + date2[2]
                else:
                    day = date2[2] - date1[2]
                hour = day*24 + date2[3] - date1[3]
        else:
            if date2[0] % 400 != 0:
                if date2[2] != date1[2]:
                    day = month_one[date1[1] - 1] - date1[2]
                    for i in range(date1[1], date2[1] - 1):
                        day = day + month_one[i]
                    day = day + date2[2]
                else:
                    day = date2[2] - date1[2]
                hour = day * 24 + date2[3] - date1[3]
            else:
                if date2[1] != date1[1]:    #月份不同
                    day = month_two[date1[1] - 1] - date1[2]
                    for i in range(date1[1], date2[1] - 1):
                        day = day + month_two[i]
                    day = day + date2[2]
                else:
                    day = date2[2] - date1[2]
                hour = day * 24 + date2[3] - date1[3]
    else:
        #根据实际情况，订单年份之间相差最大为1
        if date2[0] % 4 != 0 and date1[0] % 4 != 0:
            day = month_one[date1[1]-1] - date1[2]
            for i in range(date1[1], 12):
                day = day + month_one[i]
            for k in range(0, date2[2]-1):
                day = day + month_one[k]
            hour = (day + date2[2])*24 + date2[3] - date1[3]
        elif date2[0] % 4 == 0:
            day = month_one[date1[1] - 1] - date1[2]
            for i in range(date1[1], 12):
                day = day + month_one[i]
            for k in range(0, date2[2] - 1):
                day = day + month_two[k]
            hour = (day + date2[2])*24 + date2[3] - date1[3]
        else:
            day = month_one[date1[1] - 1] - date1[2]
            for i in range(date1[1], 12):
                day = day + month_two[i]
            for k in range(0, date2[1] - 1):
                day = day + month_one[k]
            hour = (day + date2[2])*24 + date2[3] - date1[3]

    return hour


def random_task_coding(next_task):
    next_task_Product_series = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    next_task['production_series'] = next_task['production_series'].map(next_task_Product_series)  # 对于任务生产的物资进行映射
    return next_task




