import numpy as np
import pandas as pd


def summarize(raw):
    columns = ["mae_increase", "rmse_increase", "r2_decrease", "accuracy_drop_pp", "macro_f1_decrease",
               "mean_abs_g_change", "mean_abs_h_change", "mean_abs_ts_change"]
    rows = []
    for (name, level), group in raw.groupby(["feature", "level"], sort=False):
        row = dict(feature=name, level=level, n_repeats=len(group), varies=bool(group.varies.iloc[0]))
        for column in columns:
            values = group[column].to_numpy()
            row.update({column+"_mean": float(values.mean()), column+"_std": float(values.std(ddof=1)),
                        column+"_q025": float(np.quantile(values, .025)), column+"_q975": float(np.quantile(values, .975))})
        rows.append(row)
    return pd.DataFrame(rows)
