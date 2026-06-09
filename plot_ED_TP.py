import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def exact_poly_vth(i_col, vl, degree, threshold=1e-6):
    mask = i_col > threshold
    if not mask.any(): return np.nan
    idx = np.argmax(mask)

    # Define exact nodes (using 4 points for degree=3)
    if degree == 1:
        indices = [idx - 1, idx]
    elif degree == 3:
        indices = [idx - 1, idx, idx + 1, idx + 2]
    else:
        return np.nan

    indices = [i for i in indices if 0 <= i < len(i_col)]
    if len(indices) != degree + 1:
        return np.nan

    vl_sub = vl[indices]
    i_sub = i_col[indices]

    # Exact fit mapping I to Vl
    coeffs = np.polyfit(i_sub, vl_sub, degree)
    poly = np.poly1d(coeffs)
    return poly(0)


def process_file(filename, dT=0.02):
    df = pd.read_csv(filename)
    vl = df['Vl (V)'].values
    results = []

    for col in df.columns:
        if 'Grad_' in col and '_I' in col:
            n = int(col.split('_')[1])
            i_col = df[col].values

            # Hybrid extraction
            v_th_linear = exact_poly_vth(i_col, vl, degree=1)
            v_th_cubic = exact_poly_vth(i_col, vl, degree=3)
            v_th_chosen = v_th_linear if n <= 2 else v_th_cubic

            results.append({
                'n': n,
                'n_times_0.02': n * 0.02,
                'Vth_Final': v_th_chosen
            })

    res_df = pd.DataFrame(results).sort_values('n').reset_index(drop=True)
    res_df['dVth'] = res_df['Vth_Final'].diff()
    res_df['S(T)'] = -res_df['dVth'] / dT
    return res_df


# Define files
files = {
    'SC': 'IV_data_all_grads_sc.csv',
    'Metal': 'IV_data_all_grads_m.csv'
}

data = {label: process_file(fname) for label, fname in files.items()}

# 1. Plot Vth comparison
plt.figure(figsize=(10, 6))
for label, df_res in data.items():
    plt.plot(df_res['n'], df_res['Vth_Final'], marker='o', label=label)

plt.xlabel('Gradient (n)')
plt.ylabel('Extrapolated Vth (V)')
plt.title('Vth vs Gradient: SC vs Metal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('vth_comparison_plot.png')

# 2. Plot S(T) comparison
plt.figure(figsize=(10, 6))
for label, df_res in data.items():
    plot_df = df_res.dropna(subset=['S(T)'])
    plt.plot(plot_df['n_times_0.02'], plot_df['S(T)'], marker='o', label=label)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Temperature Proxy (n * 0.02)')
plt.ylabel('Thermopower S(T) (V/K)')
plt.title('Thermopower S(T) Comparison: SC vs Metal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('thermopower_comparison_plot.png')

print("Calculations complete. Plots saved as vth_comparison_plot.png and thermopower_comparison_plot.png")