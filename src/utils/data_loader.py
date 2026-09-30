import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Cannot read Data file, Error: {e}")
        return (None, None)

    x_pandas = df.iloc[:, :-1]
    y_pandas = df.iloc[:, -1]
    return x_pandas.to_numpy(), y_pandas.to_numpy()
