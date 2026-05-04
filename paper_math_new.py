# ——————————————————————————————————————————————在用，不同长度的通用对比的 
# 输入：  
# 第一行：提取\multirow{5}{*}{\rotatebox[origin=c]{90}{  }}之间的字符串，是数据集名
# 第二行： “&长度”，就提取其中的预测长度
# 第三行：是ori的结果，四列数字是taq损失
# 第四行：是ori的结果，第一列是MSE，第二列是MAE
# 第五行：是TAQ的结果，四列数字是taq损失
# 第六行：是TAQ的结果，第一列是MSE，第二列是MAE
# 如果遇到了“\cmidrule(r){2-14}”、“&Avg”、“\\”、“\hline”开头的行跳过。
# 功能：
# 1、取同一个数据集的，同一个长度，TAQ的结果的6个值，与ORI的6个结果，比较同一列，数值更小或两个相同的加粗。输出是同样的格式，只是添加了latex格式的粗体。
# 2、取同一个数据集的，同一个指标的，比如TAQ的结果的第1个值，在4个预测长度的平均值，四舍五入，作为平均值列。
# 3、对平均值列也进行功能1的加粗，输出也是同样的格式，只是添加了latex格式的粗体。

raw_data_crosslinear = """
ETTh1}}
      &96  
& 0.200 & 0.159 & 0.241 & 0.200
& 0.378 & 0.400   
& 0.201 & 0.149 & 0.243 & 0.191
& 0.377 & 0.392     \\
      &192   
& 0.213 & 0.167 & 0.263 & 0.218
& 0.433 & 0.431
& 0.207 & 0.157 & 0.262 & 0.212
& 0.426 & 0.419      \\
      &336   
& 0.218 & 0.170 & 0.285 & 0.237
& 0.487 & 0.455
& 0.212 & 0.161 & 0.285 & 0.234
& 0.478 & 0.445      \\
      &720     
& 0.204 & 0.175 & 0.311 & 0.283
& 0.517 & 0.486
& 0.201 & 0.163 & 0.307 & 0.269
& 0.483 & 0.470      \\
      \cmidrule(r){2-14}
      &Avg     
      \\
      \hline
ETTh2}}
      &96 
& 0.174 & 0.148 & 0.194 & 0.167
& 0.285 & 0.342
& 0.176 & 0.142 & 0.188 & 0.154
& 0.280 & 0.330      \\
      &192   
& 0.200 & 0.172 & 0.219 & 0.191
& 0.366 & 0.391
& 0.207 & 0.168 & 0.216 & 0.176
& 0.365 & 0.384      \\
      &336     
& 0.209 & 0.199 & 0.239 & 0.228
& 0.437 & 0.437
& 0.218 & 0.186 & 0.234 & 0.203
& 0.411 & 0.421      \\
      &720  
& 0.212 & 0.204 & 0.245 & 0.237
& 0.443 & 0.449
& 0.218 & 0.193 & 0.248 & 0.223
& 0.437 & 0.441      \\
      \cmidrule(r){2-14}
      &Avg     
      \\
      \hline
ETTm1}}
      &96   
& 0.181 & 0.140 & 0.217 & 0.176
& 0.317 & 0.357
& 0.170 & 0.129 & 0.209 & 0.168
& 0.309 & 0.338      \\
      &192  
& 0.193 & 0.147 & 0.233 & 0.187
& 0.354 & 0.380
& 0.182 & 0.138 & 0.229 & 0.185
& 0.359 & 0.368      \\
      &336   
& 0.203 & 0.155 & 0.246 & 0.199
& 0.382 & 0.401
& 0.194 & 0.146 & 0.243 & 0.195
& 0.387 & 0.389      \\
      &720    
& 0.224 & 0.169 & 0.267 & 0.212
& 0.440 & 0.436
& 0.213 & 0.160 & 0.265 & 0.212
& 0.447 & 0.425     \\
      \cmidrule(r){2-14}
      &Avg     
      \\
      \hline
ETTm2}}
      &96    
& 0.130 & 0.111 & 0.144 & 0.125
& 0.171 & 0.255
& 0.134 & 0.107 & 0.143 & 0.115
& 0.172 & 0.249      \\ 
      &192   
& 0.154 & 0.130 & 0.170 & 0.147
& 0.238 & 0.300
& 0.159 & 0.127 & 0.167 & 0.135
& 0.236 & 0.294       \\
      &336  
& 0.167 & 0.148 & 0.189 & 0.171
& 0.295 & 0.337
& 0.174 & 0.144 & 0.187 & 0.158
& 0.293 & 0.331      \\
      &720    
& 0.200 & 0.174 & 0.224 & 0.198
& 0.394 & 0.398
& 0.215 & 0.172 & 0.226 & 0.183
& 0.402 & 0.398      \\
      \cmidrule(r){2-14}
      &Avg     
      \\
      \hline
Electricity}}
      &96    
& 0.124 & 0.102 & 0.153 & 0.131
& 0.155 & 0.254
& 0.116 & 0.093 & 0.160 & 0.137
& 0.165 & 0.253      \\
      &192   
& 0.128 & 0.106 & 0.161 & 0.139
& 0.171 & 0.267
& 0.120 & 0.097 & 0.168 & 0.145
& 0.179 & 0.266      \\
      &336   
& 0.135 & 0.113 & 0.172 & 0.150
& 0.189 & 0.284
& 0.127 & 0.103 & 0.181 & 0.157
& 0.198 & 0.284      \\
      &720   
& 0.146 & 0.126 & 0.194 & 0.174
& 0.231 & 0.320
& 0.140 & 0.115 & 0.204 & 0.179
& 0.242 & 0.319      \\
      \cmidrule(r){2-14}
      &Avg     
      \\
      \hline
Weather}}
       &96   
& 0.098 & 0.075 & 0.129 & 0.106
& 0.157 & 0.205
& 0.089 & 0.070 & 0.123 & 0.104
& 0.156 & 0.193      \\
      &192    
& 0.118 & 0.095 & 0.152 & 0.130
& 0.203 & 0.248
& 0.109 & 0.090 & 0.147 & 0.127
& 0.202 & 0.236      \\
      &336   
& 0.138 & 0.112 & 0.179 & 0.153
& 0.262 & 0.291
& 0.129 & 0.107 & 0.173 & 0.151
& 0.260 & 0.280      \\
      &720     
& 0.167 & 0.131 & 0.213 & 0.176
& 0.342 & 0.344
& 0.158 & 0.125 & 0.208 & 0.175
& 0.341 & 0.333      \\
"""

import re
import numpy as np

# -------------------------- 工具函数（全量修复） --------------------------
def extract_dataset(line):
    """提取数据集名：从 XXX}} 中取 XXX"""
    pattern = r'^([A-Za-z0-9]+)}}'
    match = re.search(pattern, line.strip())
    return match.group(1).strip() if match else None

def parse_floats(line):
    """按 & 分割提取所有可转浮点数的数值，兼容末尾\\、空格"""
    parts = [p.strip() for p in line.split('&') if p.strip()]
    nums = []
    for p in parts:
        # 彻底清理干扰字符，确保数字提取准确
        cleaned_p = re.sub(r'[\\\s]', '', p).strip()
        try:
            nums.append(float(cleaned_p))
        except:
            continue
    return nums

def bold_num(val, is_bold):
    """加LaTeX粗体，兼容字符串/数字，保留3位小数"""
    if isinstance(val, float):
        s = f"{val:.3f}"
    else:
        s = str(val).strip()
    return r"\textbf{" + s + "}" if is_bold else s

def rebuild_line(orig_line, new_vals):
    """
    核心修复：用new_vals替换原行数字，100%保留原行格式
    兼容缩进、&分隔、末尾\\、空格，解决最后一列无法识别问题
    """
    # 正则匹配所有数字（兼容带空格、末尾\\的数字）
    num_pattern = r'(?<!\w)-?\d+\.?\d*(?!\w)'
    # 按顺序替换数字，不破坏原行任何格式
    def replace_match(match):
        nonlocal val_idx
        if val_idx < len(new_vals):
            res = new_vals[val_idx]
            val_idx += 1
            return res
        return match.group(0)
    
    val_idx = 0
    new_line = re.sub(num_pattern, replace_match, orig_line)
    return new_line

def fix_backslash(s):
    """修复反斜杠输出为两个反斜杠"""
    return s.replace('\\', '\\\\')

# -------------------------- 主处理逻辑（全量修复） --------------------------
def process_crosslinear_data(raw):
    lines = [line.rstrip("\n") for line in raw.strip().split("\n")]
    output = []
    i = 0
    n = len(lines)
    current_dataset = None
    dataset_records = {}  # {数据集: {长度: {"ori": [...], "taq": [...]} } }
    all_datasets = []  # 记录所有数据集，用于补全Avg行

    # 第一次遍历：完整解析所有数据集、长度、数值，无遗漏
    while i < n:
        line = lines[i]
        strip_line = line.strip()
        # 跳过空行、格式行
        if (strip_line.startswith(r"\cmidrule") 
            or strip_line.startswith(r"\hline") 
            or strip_line == ""
            or "&Avg" in strip_line):
            i += 1
            continue
        
        # 提取数据集
        ds = extract_dataset(line)
        if ds:
            current_dataset = ds
            all_datasets.append(ds)
            dataset_records[current_dataset] = {}
            i += 1
            continue
        
        # 提取预测长度（&96, &192, &336, &720）
        if strip_line.startswith("&") and strip_line[1:].strip().isdigit():
            length = strip_line.replace("&", "").strip()
            i += 1
            # 严格读取4行数据：ori_taq、ori_mse、taq_taq、taq_mse
            if i + 3 >= n:
                break
            ori_taq_line = lines[i]
            ori_mse_line = lines[i+1]
            taq_taq_line = lines[i+2]
            taq_mse_line = lines[i+3]
            i += 4
            
            # 提取6个指标：4列TAQ损失 + MSE + MAE（最后一列）
            ori_vals = parse_floats(ori_taq_line) + parse_floats(ori_mse_line)
            taq_vals = parse_floats(taq_taq_line) + parse_floats(taq_mse_line)
            
            # 严格对齐6个指标，不足补NaN（避免0值污染），超出截断
            ori_vals = ori_vals[:6]
            taq_vals = taq_vals[:6]
            while len(ori_vals) < 6: ori_vals.append(np.nan)
            while len(taq_vals) < 6: taq_vals.append(np.nan)
            
            # 完整存储原始行和数值，用于后续重建
            dataset_records[current_dataset][length] = {
                "ori": ori_vals,
                "taq": taq_vals,
                "ori_taq_line": ori_taq_line,
                "ori_mse_line": ori_mse_line,
                "taq_taq_line": taq_taq_line,
                "taq_mse_line": taq_mse_line,
            }
            continue
        
        i += 1

    # 第二次遍历：逐行输出 + 全列加粗 + 补全/修复Avg行
    i = 0
    current_dataset = None
    processed_avg_datasets = set()  # 记录已处理Avg的数据集，避免重复

    while i < n:
        line = lines[i]
        strip_line = line.strip()
        
        # 格式行直接输出，跳过原有的空Avg行
        if strip_line.startswith(r"\cmidrule"):
            output.append(line)
            i += 1
            continue
        if strip_line.startswith(r"\hline") or strip_line == "":
            output.append(line)
            i += 1
            continue
        
        # 数据集行
        ds = extract_dataset(line)
        if ds:
            current_dataset = ds
            output.append(line)
            i += 1
            continue
        
        # 长度行：处理数值加粗，覆盖所有列（含最后一列MAE）
        if strip_line.startswith("&") and strip_line[1:].strip().isdigit():
            length = strip_line.replace("&", "").strip()
            rec = dataset_records[current_dataset].get(length)
            if not rec:
                output.append(line)
                i += 1
                continue
            
            # 严格匹配需求：更小或相等都加粗
            ori = rec["ori"]
            taq = rec["taq"]
            bold_ori = [taq[j] > ori[j] if not np.isnan(ori[j]) and not np.isnan(taq[j]) else False for j in range(6)]
            bold_taq = [taq[j] <= ori[j] if not np.isnan(ori[j]) and not np.isnan(taq[j]) else False for j in range(6)]
            
            # 重建4行，100%保留原格式，所有列（含最后一列）都加粗
            ori_taq_out = rebuild_line(rec["ori_taq_line"], [bold_num(ori[j], bold_ori[j]) for j in range(4)])
            ori_mse_out = rebuild_line(rec["ori_mse_line"], [bold_num(ori[4], bold_ori[4]), bold_num(ori[5], bold_ori[5])])
            taq_taq_out = rebuild_line(rec["taq_taq_line"], [bold_num(taq[j], bold_taq[j]) for j in range(4)])
            taq_mse_out = rebuild_line(rec["taq_mse_line"], [bold_num(taq[4], bold_taq[4]), bold_num(taq[5], bold_taq[5])])
            
            output.append(line)
            output.append(fix_backslash(ori_taq_out))
            output.append(fix_backslash(ori_mse_out))
            output.append(fix_backslash(taq_taq_out))
            output.append(fix_backslash(taq_mse_out))
            i += 5  # 长度行 + 4行数据行，总共5行
            continue
        
        # 处理Avg行：核心修复平均值计算+全列加粗，自动补全缺失的Avg行
        if "&Avg" in strip_line:
            ds = current_dataset
            processed_avg_datasets.add(ds)
            records = list(dataset_records[ds].values())
            
            # 仅当有4个长度的完整数据时计算平均值
            if len(records) >= 4:
                  # 按列计算平均值，跳过NaN，保留3位小数
                  ori_matrix = np.array([r["ori"] for r in records[:4]])
                  taq_matrix = np.array([r["taq"] for r in records[:4]])
                  
                  avg_ori = np.round(np.nanmean(ori_matrix, axis=0), 3).tolist()
                  avg_taq = np.round(np.nanmean(taq_matrix, axis=0), 3).tolist()
                  
                  # 平均值加粗逻辑
                  bold_avg_ori = [avg_taq[j] > avg_ori[j] if not np.isnan(avg_ori[j]) and not np.isnan(avg_taq[j]) else False for j in range(6)]
                  bold_avg_taq = [avg_taq[j] <= avg_ori[j] if not np.isnan(avg_ori[j]) and not np.isnan(avg_taq[j]) else False for j in range(6)]
                  
                  # 用第一组数据的行格式重建Avg行，确保格式对齐
                  sample = records[0]
                  avg_ori_taq = rebuild_line(sample["ori_taq_line"], [bold_num(avg_ori[j], bold_avg_ori[j]) for j in range(4)])
                  avg_ori_mse = rebuild_line(sample["ori_mse_line"], [bold_num(avg_ori[4], bold_avg_ori[4]), bold_num(avg_ori[5], bold_avg_ori[5])])
                  avg_taq_taq = rebuild_line(sample["taq_taq_line"], [bold_num(avg_taq[j], bold_avg_taq[j]) for j in range(4)])
                  avg_taq_mse = rebuild_line(sample["taq_mse_line"], [bold_num(avg_taq[4], bold_avg_taq[4]), bold_num(avg_taq[5], bold_avg_taq[5])])
                  
                  # 输出Avg行+完整4行平均值数据
                  output.append(line)
                  output.append(fix_backslash(avg_ori_taq))
                  output.append(fix_backslash(avg_ori_mse))
                  output.append(fix_backslash(avg_taq_taq))
                  output.append(fix_backslash(avg_taq_mse))
            else:
                  output.append(line)
            i += 1
            continue
        
        output.append(line)
        i += 1

    # 自动补全缺失Avg行的数据集（如Weather）
    for ds in all_datasets:
        if ds not in processed_avg_datasets and ds in dataset_records:
            records = list(dataset_records[ds].values())
            if len(records) >= 4:
                # 补全格式行+Avg行+平均值数据
                output.append("      \\cmidrule(r){2-14}")
                output.append("      &Avg     ")
                # 计算平均值
                ori_matrix = np.array([r["ori"] for r in records[:4]])
                taq_matrix = np.array([r["taq"] for r in records[:4]])
                avg_ori = np.round(np.nanmean(ori_matrix, axis=0), 3).tolist()
                avg_taq = np.round(np.nanmean(taq_matrix, axis=0), 3).tolist()
                # 加粗逻辑
                bold_avg_ori = [avg_taq[j] > avg_ori[j] if not np.isnan(avg_ori[j]) and not np.isnan(avg_taq[j]) else False for j in range(6)]
                bold_avg_taq = [avg_taq[j] <= avg_ori[j] if not np.isnan(avg_ori[j]) and not np.isnan(avg_taq[j]) else False for j in range(6)]
                # 重建行
                sample = records[0]
                avg_ori_taq = rebuild_line(sample["ori_taq_line"], [bold_num(avg_ori[j], bold_avg_ori[j]) for j in range(4)])
                avg_ori_mse = rebuild_line(sample["ori_mse_line"], [bold_num(avg_ori[4], bold_avg_ori[4]), bold_num(avg_ori[5], bold_avg_ori[5])])
                avg_taq_taq = rebuild_line(sample["taq_taq_line"], [bold_num(avg_taq[j], bold_avg_taq[j]) for j in range(4)])
                avg_taq_mse = rebuild_line(sample["taq_mse_line"], [bold_num(avg_taq[4], bold_avg_taq[4]), bold_num(avg_taq[5], bold_avg_taq[5])])
                # 输出
                output.append(fix_backslash(avg_ori_taq))
                output.append(fix_backslash(avg_ori_mse))
                output.append(fix_backslash(avg_taq_taq))
                output.append(fix_backslash(avg_taq_mse))
                output.append("      \\hline")
    
    return "\n".join(output)

# 生成并打印结果
result = process_crosslinear_data(raw_data_crosslinear)
print(result)