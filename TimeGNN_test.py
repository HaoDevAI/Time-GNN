import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import sys
import yaml
from models.TimeGNN import TimeGNN
from utils.data_utils import StandardScaler
from utils.utils import masked_mae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA = "datasets/data_converted.csv"
MODEL = "outputs/experiment_0/TimeGNN_models/best_run0.pt"


def predict(city: int, day: str) -> np.ndarray:
    """
    Dự đoán giá trị [temp, humidity, precip] cho một thành phố và ngày cụ thể.
    Nếu ngày cần dự đoán nằm ngoài dataset (tương lai), sẽ dự đoán tuần tự từng ngày từ ngày cuối dataset.

    Parameters:
        city (int): Mã thành phố (0: Hanoi, 1: DaNang, 2: HoChiMinh)
        day (str): Ngày cần dự đoán theo định dạng dd/mm/yyyy

    Returns:
        np.ndarray: Dự đoán đã inverse transform (mảng gồm 3 giá trị)

    Raises:
        ValueError: Nếu gặp lỗi đọc file config, dataset hoặc dữ liệu không hợp lệ.
    """
    # Đọc file cấu hình
    try:
        with open('Experiment_config.yaml', 'r') as f:
            config = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]
    except Exception as e:
        raise ValueError("Error reading config file: " + str(e))

    # Load dataset
    try:
        df = pd.read_csv(DATA)
    except Exception as e:
        raise ValueError("Error loading dataset: " + str(e))

    # Chuyển đổi cột 'datetime' sang datetime objects (dayfirst=True)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)

    # Map giá trị city thành tên cột phù hợp
    valid_cities = {
        0: "city_Hanoi",
        1: "city_DaNang",
        2: "city_HoChiMinh"
    }
    if city not in valid_cities:
        raise ValueError("Invalid city. Choose city from 0, 1, or 2.")
    city_col = valid_cities[city]

    # Lọc dữ liệu cho thành phố được chọn và sắp xếp theo datetime
    df_city = df[df[city_col] == 1].copy()
    df_city_sorted = df_city.sort_values("datetime").reset_index(drop=True)

    # Xác định cột đặc trưng (loại bỏ cột datetime)
    feature_columns = df.columns[1:]
    # Giả sử target [temp, humidity, precip] nằm ở các chỉ số [0, 3, 4] trong feature_columns
    target_indices = [0, 3, 4]
    target_names = [feature_columns[i] for i in target_indices]

    # Lấy scaler được huấn luyện trên toàn bộ dataset của thành phố (giữ nguyên các tham số)
    scaler = StandardScaler()
    scaler.fit(df_city_sorted[feature_columns].values)

    # Chuyển đổi chuỗi ngày thành datetime object
    try:
        target_date = datetime.datetime.strptime(day, "%d/%m/%Y")
    except ValueError:
        raise ValueError("Incorrect date format. Please use dd/mm/yyyy.")

    # Load mô hình và checkpoint (khởi tạo 1 lần)
    input_dim = len(feature_columns)
    hidden_dim = 32  # cần khớp với quá trình training
    output_dim = 3
    seq_len = 7
    batch_size = 1

    model_args = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "aggregate": config.get("aggregate", "last"),
        "keep_self_loops": config.get("keep_self_loops", False),
        "enforce_consecutive": config.get("enforce_consecutive", False),
        "block_size": config.get("block_size", 3)
    }

    model = TimeGNN(loss=masked_mae, **model_args).to(device)
    try:
        state_dict = torch.load(MODEL, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError("Error loading checkpoint: " + str(e))
    model = model.float()

    # Xác định ngày cuối hiện có trong dataset
    last_date = df_city_sorted["datetime"].max()

    # Nếu target_date nằm trong quá khứ hoặc đã có trong dataset, dự đoán theo cách cũ:
    if target_date <= last_date:
        # Tìm hàng ứng với target_date
        target_rows = df_city_sorted[df_city_sorted["datetime"] == target_date]
        if target_rows.empty:
            raise ValueError("The specified date was not found in the dataset for this city.")
        target_idx_city = target_rows.index[0]
        if target_idx_city < 7:
            raise ValueError("Not enough historical data (need 7 days before the target date) for this city.")
        input_seq_df = df_city_sorted.iloc[target_idx_city - 7: target_idx_city][feature_columns]
        input_seq = input_seq_df.values.astype(np.float32)
        input_seq_scaled = scaler.transform(input_seq)
        x = torch.tensor(input_seq_scaled).unsqueeze(0).to(device)  # (1, 7, input_dim)
        x = x.float()
        model.eval()
        with torch.no_grad():
            prediction = model(x)
            prediction = prediction.squeeze().cpu().numpy()  # (output_dim,)
        scaler_target = StandardScaler(targets=target_indices)
        scaler_target.mean = scaler.mean
        scaler_target.std = scaler.std
        prediction_inv = scaler_target.inverse_transform(prediction)
        return prediction_inv

    # Nếu target_date nằm trong tương lai: dự đoán tuần tự từ ngày cuối dataset cho tới target_date
    else:
        # Vòng lặp dự đoán từ last_date đến target_date (không bao gồm target_date đã có trong dataset)
        while last_date < target_date:
            # Lấy 7 ngày cuối cùng từ dataset hiện tại
            input_seq_df = df_city_sorted.iloc[-7:][feature_columns]
            input_seq = input_seq_df.values.astype(np.float32)
            input_seq_scaled = scaler.transform(input_seq)
            x = torch.tensor(input_seq_scaled).unsqueeze(0).to(device)
            x = x.float()

            model.eval()
            with torch.no_grad():
                prediction = model(x)
                prediction = prediction.squeeze().cpu().numpy()
            scaler_target = StandardScaler(targets=target_indices)
            scaler_target.mean = scaler.mean
            scaler_target.std = scaler.std
            prediction_inv = scaler_target.inverse_transform(prediction)

            # Tạo 1 row mới dựa trên row cuối hiện tại, cập nhật datetime và các target với giá trị dự đoán.
            new_row = df_city_sorted.iloc[-1].copy()
            new_date = last_date + datetime.timedelta(days=1)
            new_row["datetime"] = new_date
            # Cập nhật các giá trị mục tiêu (temp, humidity, precip)
            for j, col in enumerate(target_names):
                new_row[col] = prediction_inv[j]

            # Nếu cần, có thể cập nhật thêm các feature khác theo một logic nhất định.
            # Ở đây, ta giữ nguyên các giá trị feature còn lại từ row cuối.
            df_city_sorted = pd.concat([df_city_sorted, new_row.to_frame().T], ignore_index=True)
            last_date = new_date  # cập nhật last_date

        # Sau khi vòng lặp hoàn thành, prediction_inv là giá trị dự đoán cho target_date.
        return prediction_inv


# Ví dụ cách gọi hàm:
if __name__ == "__main__":
    # Giả sử muốn dự đoán cho thành phố Hanoi (0) vào ngày "01/01/2025"
    y_pred = predict(0, "13/03/2025")
    print("Dự đoán [temp, humidity, precip]:", y_pred)
