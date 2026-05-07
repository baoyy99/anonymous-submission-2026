import pandas as pd
import numpy as np

start_time = "2025-01-01 00:00:00"  
freq = "10min"                     
total_rows = 50000                 
seg_len = 20                       
total_segments = total_rows // seg_len  

time_index = pd.date_range(
    start=start_time,
    freq=freq,
    periods=total_rows
)


def generate_variable(rule_type: str) -> np.ndarray:

    segments = []  
    
    if rule_type == "var1":
        segments = [0 if i % 2 == 0 else 1 for i in range(total_segments)]
    
    elif rule_type == "var2":
        segments = [0 if i % 3 != 2 else 1 for i in range(total_segments)]
    
    elif rule_type == "var3":
        segments = [0]*1000 + [1]*1000 + [0]*500
    
    elif rule_type == "var4":

        segments = [np.random.choice([0,1])]  
        for i in range(1, total_segments):
            next_val = 1 - segments[-1] 
            segments.append(next_val)
    
    elif rule_type == "var5":
        segments = np.random.choice([0,1], size=total_segments, p=[0.8, 0.2]).tolist()
    
    elif rule_type == "var6":
        segments = np.random.choice([0,1], size=total_segments, p=[0.2, 0.8]).tolist()
    
    elif rule_type == "var7":
        cycle_len = 100
        segments = []
        for i in range(total_segments):
            if i % cycle_len < 50:
                segments.append(0)
            else:
                segments.append(1)
    
    elif rule_type == "var8":
        segments = np.random.choice([0,1], size=total_segments, p=[0.55, 0.45]).tolist()
    
    elif rule_type == "var9":
        segments = [0]*2000 + [1]*500
    
    elif rule_type == "var10":
        base_cycle = [0,0,0,1,1]
        segments = []
        while len(segments) < total_segments:
            np.random.shuffle(base_cycle)  
            segments.extend(base_cycle[:total_segments - len(segments)])  
    
    return np.repeat(segments, seg_len)

df = pd.DataFrame(index=time_index)

var_rules = [f"var{i}" for i in range(1, 11)]  # ["var1", "var2", ..., "var10"]
var_names = [f"VAR{str(i).zfill(2)}" for i in range(1, 11)]  # ["VAR01", "VAR02", ..., "VAR10"]

for var_name, rule in zip(var_names, var_rules):
    df[var_name] = generate_variable(rule)

df.reset_index(inplace=True)
df.rename(columns={"index": "TIME"}, inplace=True)


output_path = "custom_01_dataset.csv"  
df.to_csv(output_path, index=False, encoding="utf-8")