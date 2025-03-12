import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import pickle
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

MODEL_PATH = "models/xgb_model.pkl"
HORIZON = 7


# -----------------------------
# 1. Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess():
    """
    Load train, validation, test CSVs, combine theo thứ tự thời gian,
    convert datetime và tạo các lag features.
    """
    # Load dữ liệu
    train_df = pd.read_csv("datasets/train.csv")
    val_df = pd.read_csv("datasets/val.csv")
    test_df = pd.read_csv("datasets/test.csv")

    # Combine dữ liệu
    df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Định nghĩa các cột để tạo lag
    # Loại trừ cột datetime và các cột định danh thành phố
    lag_columns = [c for c in df.columns if c not in ["datetime", "city_DaNang", "city_Hanoi", "city_HoChiMinh"]]

    # Các cột mục tiêu cần dự báo
    target_cols = ["tempmax", "tempmin", "humidity", "precip"]

    # Tạo lag features cho mọi cột (ngoại trừ datetime và các cột chỉ định thành phố)
    for lag in range(1, HORIZON + 1):
        for col in lag_columns:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Loại bỏ những dòng có giá trị NaN (do lag)
    df = df.dropna().reset_index(drop=True)

    # Tạo lại các split theo ngày dựa vào dữ liệu ban đầu (nếu cần)
    train_start_date = pd.to_datetime(train_df["datetime"]).min()
    train_end_date = pd.to_datetime(train_df["datetime"]).max()
    val_start_date = pd.to_datetime(val_df["datetime"]).min()
    val_end_date = pd.to_datetime(val_df["datetime"]).max()
    test_start_date = pd.to_datetime(test_df["datetime"]).min()
    test_end_date = pd.to_datetime(test_df["datetime"]).max()

    train_mask = (df["datetime"] >= train_start_date) & (df["datetime"] <= train_end_date)
    val_mask = (df["datetime"] >= val_start_date) & (df["datetime"] <= val_end_date)
    test_mask = (df["datetime"] >= test_start_date) & (df["datetime"] <= test_end_date)

    train_data = df[train_mask].copy()
    val_data = df[val_mask].copy()
    test_data = df[test_mask].copy()

    # Xác định feature columns: tất cả các cột lag
    all_lag_cols = [c for c in df.columns if "_lag" in c]

    # Lấy ra danh sách các cột gốc (base) dùng để xây dựng new row khi dự báo tương lai.
    base_cols = [c for c in df.columns if
                 c not in ["datetime", "city_DaNang", "city_Hanoi", "city_HoChiMinh"] and "_lag" not in c]

    return df, train_data, val_data, test_data, all_lag_cols, target_cols, base_cols


# -----------------------------
# 2. Model Training
# -----------------------------
def train_model(X_train, y_train):
    """
    Huấn luyện mô hình XGBoost theo chế độ đa đầu ra.
    """
    xgb_reg = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    multi_output_reg = MultiOutputRegressor(xgb_reg)
    multi_output_reg.fit(X_train, y_train)
    return multi_output_reg


def save_model(model, path=MODEL_PATH):
    """
    Lưu mô hình vào file sử dụng pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path=MODEL_PATH):
    """
    Load mô hình từ file.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        return None


def prepare_features(data, all_lag_cols):
    """
    Lấy các feature từ dataframe theo danh sách các lag columns.
    """
    return data[all_lag_cols]


# -----------------------------
# 3. Tạo new row dự báo từ 7 dòng cuối của df mới
# -----------------------------
def create_new_row(last_7_df, base_cols, all_lag_cols, target_cols, city, predicted_vals):
    """
    Tạo new row cho ngày mới dựa trên 7 dòng cuối của last_7_df.
      - Các target columns được cập nhật từ predicted_vals.
      - Các base columns (trừ target_cols) sao chép từ dòng cuối của last_7_df.
      - Các lag columns được tính lại từ 7 dòng cuối hiện có (sau khi thêm new row).
    """
    # Lấy dòng cuối của 7 dòng hiện có
    last_row = last_7_df.iloc[-1].copy()

    # Tạo new_row ban đầu, sao chép toàn bộ từ last_row
    new_row = last_row.copy()

    # Cập nhật datetime: new_date = last_row["datetime"] + 1 ngày
    new_date = last_row["datetime"] + timedelta(days=1)
    new_row["datetime"] = new_date

    # Cập nhật target columns với giá trị dự báo
    for col, pred in zip(target_cols, predicted_vals):
        new_row[col] = pred

    # Với các base columns (ngoại trừ target_cols), copy từ last_row
    for col in base_cols:
        if col not in target_cols:
            new_row[col] = last_row[col]

    # Cập nhật cột thành phố theo lựa chọn
    city_dict = {0: "city_Hanoi", 1: "city_DaNang", 2: "city_HoChiMinh"}
    for col in ["city_DaNang", "city_Hanoi", "city_HoChiMinh"]:
        new_row[col] = 1 if col == city_dict.get(city) else 0

    # Tạo dataframe tạm thời bằng cách ghép last_7_df và new_row
    new_row_df = pd.DataFrame([new_row])
    temp_df = pd.concat([last_7_df, new_row_df], ignore_index=True)

    # Sau khi có temp_df, ta tạo lại các lag features cho dòng cuối (new_row)
    for lag in range(1, HORIZON + 1):
        for col in base_cols:
            lag_col = f"{col}_lag{lag}"
            # Lấy giá trị từ dòng cách new_row lag+0 vị trí
            temp_df.loc[temp_df.index[-1], lag_col] = temp_df.loc[temp_df.index[-lag - 1], col]

    # Lấy lại các lag columns cho new_row
    for col in all_lag_cols:
        new_row[col] = temp_df.loc[temp_df.index[-1], col]

    return new_row


# -----------------------------
# 4. Forecast Web Function
# -----------------------------
def predict(city: int, day: str):
    """
    Hàm này dùng để dự báo giá trị y_pred (các biến [tempmax, tempmin, humidity, precip])
    từ web.
      - city: số nguyên (0: Hà Nội, 1: Đà Nẵng, 2: Hồ Chí Minh)
      - day: chuỗi định dạng "dd/mm/yyyy"

    Nếu target_date đã có trong dataset cho thành phố đã chọn thì dự báo trực tiếp.
    Nếu target_date > max(df["datetime"]) thì:
      - Khởi tạo một df mới với 7 dòng cuối của dataset hiện có.
      - Dự báo lặp từ ngày mới (dùng 7 dòng cuối hiện có) cho đến khi đạt target_date.
      - Mỗi new row được tạo theo cách:
          + target columns: giá trị dự báo từ mô hình.
          + base columns (ngoại trừ target_cols): copy từ dòng cuối.
          + lag columns: được cập nhật lại từ 7 dòng cuối.
    Hàm trả về mảng y_pred dự báo cho target_date.
    """
    # Load dữ liệu và tiền xử lý
    df, _, _, _, all_lag_cols, target_cols, base_cols = load_and_preprocess()

    model = load_model()
    if model is None:
        raise ValueError("Không tìm thấy xgb_model.pkl")

    # Chuyển đổi day sang datetime
    try:
        target_date = pd.to_datetime(day, format="%d/%m/%Y")
    except ValueError:
        raise ValueError("Sai định dạng ngày. Vui lòng sử dụng định dạng dd/mm/yyyy.")

    # Thiết lập mapping cho city
    city_dict = {0: "city_Hanoi", 1: "city_DaNang", 2: "city_HoChiMinh"}
    if city not in city_dict:
        raise ValueError("Lựa chọn thành phố không hợp lệ. Chọn 0, 1 hoặc 2.")
    city_col = city_dict[city]

    # Nếu target_date đã có trong dataset (cho thành phố đã chọn) thì dự báo trực tiếp
    row = df[(df["datetime"] == target_date) & (df[city_col] == 1)]
    if not row.empty:
        X_input = row[all_lag_cols]
        y_pred = model.predict(X_input)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        return y_pred[0]
    else:
        # Nếu target_date là ngày trong tương lai, tiến hành dự báo lặp trên 1 df mới chỉ với 7 dòng cuối
        current_last_date = df["datetime"].max()
        if target_date <= current_last_date:
            raise ValueError(f"Dữ liệu cho ngày {target_date.strftime('%d/%m/%Y')} không tồn tại.")

        # Khởi tạo new_df với 7 dòng cuối của df hiện có
        new_df = df.tail(7).copy().reset_index(drop=True)

        # Lặp dự báo cho đến khi new_df có dòng cuối cùng có datetime bằng target_date
        while new_df["datetime"].max() < target_date:
            # Tạo input cho mô hình: từ các lag columns của 7 dòng cuối
            last_7 = new_df.tail(7)
            new_X = {}
            for lag in range(1, HORIZON + 1):
                for col in base_cols:
                    new_X[f"{col}_lag{lag}"] = last_7[col].iloc[-lag]
            X_new = pd.DataFrame([new_X])
            # Dự báo cho ngày tiếp theo
            y_new = model.predict(X_new)[0]
            y_new = np.clip(y_new, a_min=0, a_max=None)
            # Tạo new row dự báo
            new_row = create_new_row(last_7, base_cols, all_lag_cols, target_cols, city, y_new)
            # Append new_row vào new_df
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

        # Khi vòng lặp kết thúc, dòng cuối của new_df ứng với target_date
        return np.array([new_df[target_cols].iloc[-1][col] for col in target_cols])


# -----------------------------
# Ví dụ gọi hàm forecast_web
# -----------------------------
if __name__ == "__main__":
    city = int(input("Chọn thành phố (0: Hà Nội/1: Đà Nẵng/2: Hồ Chí Minh): "))
    date = input("Chọn ngày (dd/mm/yyyy): ")
    try:
        result = predict(city, date)
        print("Dự báo [tempmax, tempmin, humidity, precip]:", [f"{v:.2f}" for v in result])
    except Exception as e:
        print("Error:", e)
