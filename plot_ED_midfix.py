import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load the data
max_diff_list = [6,20]
for diff in max_diff_list:
    file_path = f'IV_data_all_grads_{diff}T0.CSV'
    df = pd.read_csv(file_path)

    # Create a new DataFrame containing only rows where Vl (V) <= 1
    df = df[df['Vl (V)'] <= 1].copy()
    df = df[df['Vl (V)'] >= 0].copy()
    # Extract Voltage column
    vl = df['Vl (V)']

    # Identify and sort Gradient columns numerically
    grad_cols = [col for col in df.columns if 'Grad_' in col and '_I' in col]
    grad_cols.sort(key=lambda x: int(x.split('_')[1]))

    # Setup plot
    plt.figure(figsize=(10, 6))

    # Define shift parameter
    shift = 0.0

    # Generate color map (Reds, fading with gradient index)
    # We use linspace(0.4, 1.0) to avoid overly light colors that are hard to see
    colors = cm.viridis(np.linspace(0, 1, len(grad_cols)))

    # Plotting each gradient curve
    for i, col in enumerate(grad_cols):
        plt.plot(vl + shift, df[col], color=colors[i], alpha=0.8, linewidth=1.5)

    # Formatting
    plt.xlabel('Left Voltage $V_l$ (V)')
    plt.ylabel('Steady-State Current $I$ (A)')
    plt.title('I-V Characteristics SC Island (Fading Yellow = Stronger Gradient)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'all_iv_curves_faded_maxdiff{diff}T0.png', dpi=300)
    print(f"Plot saved as all_iv_curves_faded_maxdiff{diff}T0.png")
