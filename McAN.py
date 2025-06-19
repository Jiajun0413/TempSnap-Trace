from multiprocessing import Pool
import pandas as pd
import numpy as np
import datetime as dt
import re

global_mut_lst = None


def str2date(tt):
    """将各种格式的日期字符串转换为datetime.date对象"""
    # 处理非字符串和NaN值
    if pd.isna(tt):
        return None
        
    # 转换为字符串
    if not isinstance(tt, str):
        tt = str(tt).strip()
    
    # 过滤无效字符串
    if tt in ('', 'nan', 'NaT', 'None'):
        return None
        
    # 匹配标准格式
    date_pat = re.compile(r'\A([0-9]{1,4})-([0-9]{1,2})-([0-9]{1,2})\Z')
    matched = date_pat.match(tt)
    
    if matched:
        try:
            return dt.date(int(matched[1]), int(matched[2]), int(matched[3]))
        except ValueError:
            return None
    
    return None



class Calcdate:
    def __init__(self, samples=None, time_interval='', start_date=None, end_date=None):
        self.samples = samples
        self.samples['Date'] = self.samples['Date'].apply(str2date)
        self.time_interval = dt.timedelta(days=int(time_interval))
        
        # 直接使用datetime对象，无需转换
        # 提供向后兼容的字符串转换
        if isinstance(start_date, str):
            self.start_date = str2date(start_date)
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = str2date(end_date)
        else:
            self.end_date = end_date
            
        # 初始化current_date
        self.current_date = self.start_date
        self.samples_available = None
        
    def next_date(self):
        # 现在可以直接相加，因为都是datetime对象
        self.current_date += self.time_interval
        
    def update_data_available(self):
        if self.samples is not None:
            # 直接使用datetime比较，无需字符串转换
            self.samples_available = self.samples.loc[self.samples['Date'] <= self.current_date, :]

def str2mut_list(str_mutations):
    if str_mutations == '':
        return set()
    if str_mutations is None:
        return set()
    if pd.isna(str_mutations):
        return set()
    return set(str_mutations.split(';'))

class McAN:
    def __init__(self, output_path=None, samples=None, current_date=None, time_interval=None):
        self.samples = None
        self.samples_grouped = None
        self.df_haps = None
        self.graph = None
        self.mcan_data_with_labels=None
        self.out_path = None
        self.current_date = None
        self.time_interval = None
        if output_path is not None:
            self.out_path = output_path
        if samples is not None:
            self.samples = samples
        if current_date is not None:
            self.current_date = current_date
        if time_interval is not None:
            self.time_interval = time_interval
#无参数的_init_方法不能在对象创建时传入参数，只能在创建对象后手动设置属性的值

    # McAN
    def find_haps(self):
        # time complexity O(n), n is the number of samples
        
        # 先对samples按Mutations_str和sDate_str排序，确保每组中最早日期的记录在前
        sorted_samples = self.samples.sort_values(['Mutations_str', 'Date'])
        
        # 使用drop_duplicates保留每个Mutations_str组的第一条记录(最早日期)
        earliest_records = sorted_samples.drop_duplicates('Mutations_str')
        
        # 创建辅助DataFrame用于快速查找最早记录的Location
        earliest_locations = earliest_records.set_index('Mutations_str')['Location']
        
        # 正常分组聚合
        self.samples_grouped = self.samples.groupby('Mutations_str')
        agg_dict = {
            'Date': 'min',
            'ID': lambda x: list(x)
        }
        
        # 如果存在Clade列，添加到聚合字典
        if 'Clade' in self.samples.columns:
            agg_dict['Clade'] = 'first'
        if 'Lineage' in self.samples.columns:
            agg_dict['Lineage'] = 'first'
        
        # 执行聚合
        self.df_haps = self.samples_grouped.agg(agg_dict)
        # 添加来自最早记录的Location
        self.df_haps['Location'] = self.df_haps.index.map(lambda x: earliest_locations.get(x))
        
        # 添加其他列
        self.df_haps['size'] = self.samples_grouped.agg({'Date': len})
        self.df_haps['Mutations'] = self.df_haps.index.to_series().apply(str2mut_list)
        self.df_haps['number_of_mutations'] = self.df_haps['Mutations'].apply(len)

    def sort_haps(self):
        # very fast, running time can be ignored
        self.df_haps = \
            self.df_haps.sort_values(axis=0,
                                     ascending=[False, False, True],
                                     by=['number_of_mutations', 'size', 'Date'])
#按 number_of_mutations 列降序排列,如果 number_of_mutations 相同，则按 size 列降序排列。如果 size 也相同，则按 sDate_str 列升序排列
    def hap_index(self, int_hap_index):
        return self.df_haps.index[int_hap_index]

    @staticmethod
    def find_ancestor_single(location_start: int) -> int:
        global global_mut_lst
        i = location_start
        current_mutations = global_mut_lst[i]
        
        # 从i+1开始搜索最小的j
        for j in range(i + 1, len(global_mut_lst)):
            ancestor_mutations = global_mut_lst[j]
            # 检查真子集关系
            if ancestor_mutations.issubset(current_mutations):  # 确保是真子集
                return j  

    def find_ancestor_parallel(self, num_of_processes=1):
        """优化的并行祖先查找"""
        global global_mut_lst
        # 使用列表而不是Series转换
        sorted_indices = self.df_haps.index.values  # 直接使用numpy数组
        global_mut_lst = self.df_haps['Mutations'].values  # 使用numpy数组
        try:
            with Pool(processes=num_of_processes) as pool:
                # 直接使用map而不是map_async，减少内存开销
                ancestor_indices = pool.map(self.find_ancestor_single, 
                                        range(len(global_mut_lst) - 1),)
                  
        finally:
            global_mut_lst = None  # 清理全局变量
        
        # 使用列表推导式优化映射过程
        lst_anc = [sorted_indices[idx] if idx is not None else None 
                for idx in ancestor_indices]
        lst_anc.append(None)  # 添加最后一个元素
        
        # 直接赋值
        self.df_haps['Ancestor'] = lst_anc
        #return ancestor_indices
    def mcan(self, num_of_processes=1):
        self.find_haps()
        self.sort_haps()
        self.find_ancestor_parallel(num_of_processes=num_of_processes)

        # 1. 一次性预处理所有ancestor值，确保空值为None
        self.df_haps['Ancestor'] = self.df_haps['Ancestor'].map(
            lambda x: None if pd.isna(x) or x == '' or (isinstance(x, (list, np.ndarray)) and len(x) == 0) else x
        )
        
        # 2. 创建高效的索引到Acc映射 - 只在这里创建一次
        index_to_acc = {}
        for mut_str, acc in zip(self.df_haps.index, self.df_haps['ID']):
            index_to_acc[mut_str] = acc
        
        # 3. 优化后的ancestor_Acc计算函数
        def get_ancestor_acc(ancestor):
            # 快速路径：处理空值
            if ancestor is None or pd.isna(ancestor):
                return None
            
            # 处理字符串类型
            if isinstance(ancestor, str):
                if not ancestor.strip():  # 处理空字符串
                    return None
                return index_to_acc.get(ancestor)  # 直接使用get方法，避免额外的in检查
            
            # 处理列表类型 (一次性检查)
            if isinstance(ancestor, (list, np.ndarray)):
                if not ancestor:  # 空列表
                    return None
                    
                # 使用列表推导式代替循环，更加高效
                all_accs = []
                for anc in ancestor:
                    if anc and isinstance(anc, str) and anc in index_to_acc:
                        acc_value = index_to_acc[anc]
                        if acc_value:  # 简化的非空检查
                            all_accs.extend(acc_value)
                
                return all_accs if all_accs else None
            
            # 默认返回None
            return None

        # 4. 直接应用优化函数 - 无需额外的验证步骤
        self.df_haps['Ancestor_ID'] = self.df_haps['Ancestor'].apply(get_ancestor_acc)
        return self.df_haps 
    



