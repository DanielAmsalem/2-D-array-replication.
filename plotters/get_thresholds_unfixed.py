import pandas as pd
import numpy as np


def exact_poly_vth(i_col, vl, degree, threshold=1e-6):
    mask = i_col > threshold

    # Case 1: The current never crosses the threshold (Completely Blocked)
    if not mask.any():
        return np.nan

    # Case 2: The current is already above threshold at V=0 (No Blockade / Leaking)
    if mask[0]:
        return 0.0

    idx = np.argmax(mask)

    # Define exact nodes for interpolation (using 4 points for degree=3)
    if degree == 1:
        indices = [idx - 1, idx]
    elif degree == 3:
        indices = [idx - 1, idx, idx + 1, idx + 2]
    else:
        return np.nan

    # Filter out out-of-bounds indices at the tail end of the sweep
    indices = [i for i in indices if 0 <= i < len(i_col)]

    # Fallback to linear if we don't have enough points at the edge for a cubic fit
    if len(indices) != degree + 1:
        indices = [idx - 1, idx]
        degree = 1

    vl_sub = vl[indices]
    i_sub = i_col[indices]

    # Exact fit mapping I to Vl to find the zero-crossing (V at I=0)
    coeffs = np.polyfit(i_sub, vl_sub, degree)
    poly = np.poly1d(coeffs)
    return poly(0)


def extract_thresholds(filename):
    print(f"Reading data from {filename}...")
    df = pd.read_csv(filename)
    vl = df['Vl (V)'].values
    results = []

    for col in df.columns:
        if col.startswith('Grad_') and col.endswith('_I'):
            # Safely extract 'n' which can now be negative (e.g., from 'Grad_-19_I')
            n = int(col.split('_')[1])
            i_col = df[col].values

            # Hybrid extraction: cubic for smoother high-gradient curves, linear for sharp small-n corners
            v_th_linear = exact_poly_vth(i_col, vl, degree=1)
            v_th_cubic = exact_poly_vth(i_col, vl, degree=3)
            v_th_chosen = v_th_linear if abs(n) <= 2 else v_th_cubic

            results.append({
                'n': n,
                'Vth': v_th_chosen
            })

    # Sort mathematically from highly negative gradients to highly positive gradients
    res_df = pd.DataFrame(results).sort_values('n').reset_index(drop=True)

    # Export cleanly to CSV for your external plotting script
    output_csv = "Extracted_Vth_Thresholds.csv"
    res_df.to_csv(output_csv, index=False)
    print(f"Successfully processed {len(res_df)} gradients.")
    print(f"Threshold data saved to: {output_csv}")

    return res_df


if __name__ == '__main__':
    # Target the newly generated file containing both flip states
    extract_thresholds('NEW_IV_data_reintegrated_D2_0.csv')