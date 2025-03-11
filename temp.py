import pandas as pd
import numpy as np

# Load dữ liệu từ file CSV
df = pd.read_csv('Time-GNN/datasets/data.csv')

# Chuyển đổi cột 'day' sang dạng chu kỳ (giả sử số ngày tối đa là 31)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

# Chuyển đổi cột 'month' sang dạng chu kỳ (giả sử có 12 tháng)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['city_DaNang'] = df['city_DaNang'].astype(int)
df['city_Hanoi'] = df['city_Hanoi'].astype(int)
df['city_HoChiMinh'] = df['city_HoChiMinh'].astype(int)

# Xoá các cột ban đầu 'day' và 'month'
df.drop(columns=['day', 'month'], inplace=True)

# Lưu lại file mới sau khi chuyển đổi
df.to_csv("Time-GNN/datasets/data_converted.csv", index=False)

print("Đã chuyển đổi và lưu file thành công!")
