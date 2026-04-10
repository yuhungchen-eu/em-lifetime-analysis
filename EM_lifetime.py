import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import lognorm, norm

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "em_lifetime_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Stress condition
T_STRESS_C = 300.0
I_STRESS_MA_PER_UM2 = 0.2

# Use condition / spec
T_USE_C = 110.0
IAVG_SPEC_MA_PER_UM2 = 0.1
TARGET_HOURS = 100000.0
TARGET_CDF = 0.001   # 0.1% cumulative failure

# Black's equation assumptions for Cu EM
EA_EV = 0.90
N_CURRENT = 1.0
K_B_EV_PER_K = 8.617333262e-5

CONDITIONS = ["Before LOP", "After LOP"]

# ============================================================
# Helpers
# ============================================================

def celsius_to_kelvin(temp_c: float) -> float:
    return temp_c + 273.15

def benard_plotting_positions(n: int) -> np.ndarray:
    i = np.arange(1, n + 1)
    return (i - 0.3) / (n + 0.4)

def t_at_percentile_hours(mu: float, sigma: float, cumulative_failure: float) -> float:
    z = norm.ppf(cumulative_failure)
    return float(np.exp(mu + sigma * z))

def pass_fail(allowable_iavg: float, spec_iavg: float) -> str:
    return "PASS" if allowable_iavg >= spec_iavg else "FAIL"

def probability_ticks():
    prob_pct = np.array([
        0.1, 0.2, 0.5,
        1, 2, 5, 10, 20, 30, 50, 70, 80, 90, 95, 98, 99,
        99.5, 99.8, 99.9
    ])
    prob_frac = prob_pct / 100.0
    z_ticks = norm.ppf(prob_frac)
    tick_labels = [f"{p:g}" for p in prob_pct]
    return z_ticks, tick_labels

def temp_acceleration_term(t_stress_c: float, t_use_c: float, ea_ev: float) -> float:
    t_stress_k = celsius_to_kelvin(t_stress_c)
    t_use_k = celsius_to_kelvin(t_use_c)
    return math.exp((ea_ev / K_B_EV_PER_K) * (1.0 / t_use_k - 1.0 / t_stress_k))

# ============================================================
# Fitting methods
# ============================================================

def fit_lognormal_mle(ttf_hours: np.ndarray) -> dict:
    ttf_hours = np.asarray(ttf_hours, dtype=float)
    shape, loc, scale = lognorm.fit(ttf_hours, floc=0)
    sigma = float(shape)
    mu = float(np.log(scale))
    return {
        "method": "MLE",
        "mu": mu,
        "sigma": sigma,
    }

def fit_lognormal_lse(ttf_hours: np.ndarray) -> dict:
    x = np.sort(np.asarray(ttf_hours, dtype=float))
    f = benard_plotting_positions(len(x))
    z = norm.ppf(f)
    y = np.log(x)

    slope, intercept = np.polyfit(z, y, 1)
    sigma = float(slope)
    mu = float(intercept)

    return {
        "method": "LSE",
        "mu": mu,
        "sigma": sigma,
    }

# ============================================================
# Projection
# ============================================================

def allowable_iavg_from_stress_t_life(
    t_life_stress_hr: float,
    target_hours: float,
    i_stress: float,
    t_stress_c: float,
    t_use_c: float,
    n_current: float,
    ea_ev: float
) -> float:
    """
    Black's equation:
        life_use = life_stress
                   * (I_stress / I_use)^n
                   * exp(Ea/k * (1/T_use - 1/T_stress))

    Solve for allowable I_use at target life.
    """
    temp_term = temp_acceleration_term(t_stress_c, t_use_c, ea_ev)
    ratio = target_hours / (t_life_stress_hr * temp_term)
    i_use_allowable = i_stress / (ratio ** (1.0 / n_current))
    return float(i_use_allowable)

# ============================================================
# Export plots
# ============================================================

def export_probability_plot_by_method(method_name: str, condition_data_dict: dict, output_dir: str) -> str:
    """
    Probability plot with:
    - x-axis: log scale TTF
    - y-axis: normal probability scale shown as %
    - one chart for one fit method (LSE or MLE)
    """
    plt.figure(figsize=(8.8, 6.4))

    z_ticks, tick_labels = probability_ticks()
    z_min = norm.ppf(0.001)
    z_max = norm.ppf(0.999)

    global_x_min = None
    global_x_max = None

    for condition_name, payload in condition_data_dict.items():
        ttf = np.asarray(payload["ttf"], dtype=float)
        fit = payload["fit"]

        x_sorted = np.sort(ttf)
        n = len(x_sorted)
        F = benard_plotting_positions(n)
        z_emp = norm.ppf(F)

        # empirical data points
        plt.scatter(x_sorted, z_emp, s=35, label=f"{condition_name} data")

        # update x range
        x_min = max(np.min(x_sorted) * 0.7, 1e-6)
        x_max = np.max(x_sorted) * 1.5

        if global_x_min is None or x_min < global_x_min:
            global_x_min = x_min
        if global_x_max is None or x_max > global_x_max:
            global_x_max = x_max

        # fitted line across full probability range
        z_line = np.linspace(z_min, z_max, 400)
        x_line = np.exp(fit["mu"] + fit["sigma"] * z_line)
        plt.plot(x_line, z_line, linewidth=2, label=f"{condition_name} {method_name} fit")

    plt.xscale("log")

    # force multiple decades to show nicely
    x_min_plot = 10 ** np.floor(np.log10(global_x_min))
    x_max_plot = 10 ** np.ceil(np.log10(global_x_max * 2.0))
    plt.xlim(x_min_plot, x_max_plot)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

    plt.xlabel("Time to Failure, TTF (hours)")
    plt.ylabel("Cumulative Failure Probability (%)")
    plt.title(f"{method_name}: Lognormal Probability Plot (Before vs After LOP)")

    plt.yticks(z_ticks, tick_labels)
    plt.ylim(z_min, z_max)

    plt.grid(True, which="major", linestyle="-", alpha=0.5)
    plt.grid(True, which="minor", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f"{method_name.lower()}_probability_plot.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=220, bbox_inches="tight")
    plt.close()

    return filepath

def export_table_plot(df: pd.DataFrame, output_dir: str) -> str:
    df_plot = df.copy()

    numeric_cols = [
        "Sigma",
        "t50 (hr)",
        "t0.1 (hr)",
        "Imax (mA/um²)",
        "Ispec (mA/um²)",
    ]

    # ============================================================
    # Custom formatting per column
    # ============================================================

    for col in df_plot.columns:
        if col == "Sigma":
            df_plot[col] = df_plot[col].map(lambda x: f"{x:.2f}")
        elif col in ["t50 (hr)", "t0.1 (hr)"]:
            df_plot[col] = df_plot[col].map(lambda x: f"{x:.1f}")
        elif col in ["Imax (mA/um²)", "Ispec (mA/um²)"]:
            df_plot[col] = df_plot[col].map(lambda x: f"{x:.3f}")

    n_rows = len(df_plot)
    fig_height = max(2.5, 0.8 + 0.55 * (n_rows + 1))

    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        loc="upper center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    # ============================================================
    # Color Results column (PASS = blue, FAIL = red)
    # ============================================================

    # find column index of "Results"
    col_idx = df_plot.columns.get_loc("Results")

    for row in range(1, len(df_plot) + 1):  # skip header row (row=0)
        value = df_plot.iloc[row - 1, col_idx]

        cell = table[row, col_idx]

        if value == "PASS":
            cell.get_text().set_color("blue")
            cell.get_text().set_weight("bold")
        elif value == "FAIL":
            cell.get_text().set_color("red")
            cell.get_text().set_weight("bold")

    plt.title("EM Lifetime Summary Table", fontsize=12, pad=4)

    filepath = os.path.join(output_dir, "summary_table.png")
    plt.savefig(filepath, dpi=220, bbox_inches="tight")
    plt.close()

    return filepath

# ============================================================
# Main workflow
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading CSV from:")
    print(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    required_cols = {"Condition", "TTF_hr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    fit_results = {}
    summary_rows = []

    for condition in CONDITIONS:
        condition_df = df[df["Condition"] == condition].copy()

        if condition_df.empty:
            print(f"Warning: no data found for condition '{condition}'")
            continue

        ttf = condition_df["TTF_hr"].astype(float).values

        lse_fit = fit_lognormal_lse(ttf)
        mle_fit = fit_lognormal_mle(ttf)

        fit_results[condition] = {
            "ttf": ttf,
            "LSE": lse_fit,
            "MLE": mle_fit
        }

        for fit in [lse_fit, mle_fit]:
            t50_stress_hr = float(np.exp(fit["mu"]))
            t01_stress_hr = t_at_percentile_hours(
                mu=fit["mu"],
                sigma=fit["sigma"],
                cumulative_failure=TARGET_CDF
            )

            allowable_iavg = allowable_iavg_from_stress_t_life(
                t_life_stress_hr=t01_stress_hr,
                target_hours=TARGET_HOURS,
                i_stress=I_STRESS_MA_PER_UM2,
                t_stress_c=T_STRESS_C,
                t_use_c=T_USE_C,
                n_current=N_CURRENT,
                ea_ev=EA_EV
            )

            summary_rows.append({
                "Condition": condition,
                "Fit_Method": fit["method"],
                "Sigma": fit["sigma"],
                "t50 (hr)": t50_stress_hr,
                "t0.1 (hr)": t01_stress_hr,
                "Imax (mA/um²)": allowable_iavg,
                "Ispec (mA/um²)": IAVG_SPEC_MA_PER_UM2,
                "Results": pass_fail(allowable_iavg, IAVG_SPEC_MA_PER_UM2),
            })

    if not fit_results:
        raise ValueError("No valid condition data found. Check the CSV content.")

    lse_plot_path = export_probability_plot_by_method(
        method_name="LSE",
        condition_data_dict={
            condition: {
                "ttf": fit_results[condition]["ttf"],
                "fit": fit_results[condition]["LSE"]
            }
            for condition in CONDITIONS if condition in fit_results
        },
        output_dir=OUTPUT_DIR
    )

    mle_plot_path = export_probability_plot_by_method(
        method_name="MLE",
        condition_data_dict={
            condition: {
                "ttf": fit_results[condition]["ttf"],
                "fit": fit_results[condition]["MLE"]
            }
            for condition in CONDITIONS if condition in fit_results
        },
        output_dir=OUTPUT_DIR
    )

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df[
        [
            "Condition",
            "Fit_Method",
            "Sigma",
            "t50 (hr)",
            "t0.1 (hr)",
            "Imax (mA/um²)",
            "Ispec (mA/um²)",
            "Results",
        ]
    ]

    summary_df["Condition"] = pd.Categorical(
        summary_df["Condition"],
        categories=CONDITIONS,
        ordered=True
    )
    summary_df["Fit_Method"] = pd.Categorical(
        summary_df["Fit_Method"],
        categories=["LSE", "MLE"],
        ordered=True
    )
    summary_df = summary_df.sort_values(["Fit_Method", "Condition"]).reset_index(drop=True)

    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    table_plot_path = export_table_plot(summary_df, OUTPUT_DIR)

    print("\n=== Exported files ===")
    print("LSE plot:   ", lse_plot_path)
    print("MLE plot:   ", mle_plot_path)
    print("Table CSV:  ", summary_csv_path)
    print("Table PNG:  ", table_plot_path)

    print("\n=== Summary table ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()