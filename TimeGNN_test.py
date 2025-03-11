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


def main():
    # Read experiment config from YAML file.
    try:
        with open('Experiment_config.yaml', 'r') as f:
            config = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]
    except Exception as e:
        print("Error reading config file:", e)
        sys.exit(1)

    # Load dataset (assuming 'custom' dataset, adjust if needed)
    # The training code uses "data_converted.csv" for the "custom" dataset.
    try:
        df = pd.read_csv(DATA)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)

    # Convert the 'datetime' column to datetime objects (dayfirst=True)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)

    city_input = input("Enter the city (1 for DaNang/2 for Hanoi/3 for HoChiMinh): ")
    # Map user input to column name
    valid_cities = {
        "1": "city_DaNang",
        "2": "city_Hanoi",
        "3": "city_HoChiMinh"
    }
    if city_input not in valid_cities:
        print("Invalid city. Choose city from 1,2 or 3.")
        sys.exit(1)
    city_col = valid_cities[city_input]
    df_city = df[df[city_col] == 1].copy()
    df_city_sorted = df_city.sort_values("datetime").reset_index(drop=True)
    # Ask the user for the target date.
    date_str = input("Enter the target day (dd/mm/yyyy): ").strip()
    try:
        target_date = datetime.datetime.strptime(date_str, "%d/%m/%Y")
    except ValueError:
        print("Incorrect date format. Please use dd/mm/yyyy.")
        sys.exit(1)

    # Find the row corresponding to the target date.
    target_rows = df_city_sorted[df_city_sorted["datetime"] == target_date]
    if target_rows.empty:
        print("The specified date was not found in the dataset for this city.")
        sys.exit(1)
    target_idx_city = target_rows.index[0]

    # Ensure we have at least 7 days of history before the target date.
    if target_idx_city < 7:
        print("Not enough historical data (need 7 days before the target date) for this city.")
        sys.exit(1)

    # Determine feature columns (all except the date column).
    feature_columns = df.columns[1:]
    # For prediction, we assume the targets [temp, humidity, precip] are the first three features.
    target_indices = [0, 3, 4]

    # Extract the 7-day history before the target day.
    input_seq_df = df_city_sorted.iloc[target_idx_city - 7: target_idx_city][feature_columns]
    # Also extract the true values from the target day.
    true_row = target_rows.iloc[0]

    # Convert to numpy array and ensure type is float32.
    input_seq = input_seq_df.values.astype(np.float32)  # Shape: (7, num_features)

    # Create and fit a scaler on the entire dataset features (consistent with training).
    scaler = StandardScaler()
    scaler.fit(df_city_sorted[feature_columns].values)
    input_seq_scaled = scaler.transform(input_seq)

    # Prepare the input tensor for the model with shape (batch_size, seq_len, input_dim).
    x = torch.tensor(input_seq_scaled).unsqueeze(0).to(device)  # Shape: (1, 7, input_dim)
    x = x.float()  # Ensure the input is float32.

    # Use the config to determine model parameters.
    # For our test, we override seq_len to 7 and output_dim to 3 (for [temp, humidity, precip]).
    input_dim = len(feature_columns)
    hidden_dim = 32  # This should match the training setting.
    output_dim = 3
    seq_len = 7
    batch_size = 1

    # Build the model with parameters from config and our test overrides.
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
    # Use masked_mae as loss function, as in training.
    model = TimeGNN(loss=masked_mae, **model_args).to(device)

    # Load the trained checkpoint.
    try:
        state_dict = torch.load(MODEL, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print("Error loading checkpoint:", e)
        sys.exit(1)

    # Ensure model and input are in float32.
    model = model.float()
    x = x.float()

    # Run prediction.
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        prediction = prediction.squeeze().cpu().numpy()  # Shape: (output_dim,)

    # Inverse-transform the predictions.
    # Create a temporary scaler that uses the same mean and std for the target features.
    scaler_target = StandardScaler(targets=target_indices)
    scaler_target.mean = scaler.mean  # Use full feature scaler parameters.
    scaler_target.std = scaler.std
    prediction_inv = scaler_target.inverse_transform(prediction)

    # Extract the true values for [temp, humidity, precip] from the target row.
    true_values = true_row[feature_columns].values.astype(np.float32)[target_indices]

    # Print the comparison.
    print("\nFor target date {}:".format(date_str))
    print("Predicted values [temp, humidity, precip]:", [f"{val:.2f}" for val in prediction_inv])
    print("True values      [temp, humidity, precip]:", [f"{val:.2f}" for val in true_values])


if __name__ == "__main__":
    main()