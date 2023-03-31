from nsgaii import *






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