import warnings
from pathlib import Path

import numpy as np

''' 
THIS FILE FINDS A TABLE IN A GIVEN PATH AND RETURNS:
    neg_bound < dE < pos_bound 
    Ec = ? 
    T: list -> the temperature gradient for this table
    how many values are stored 
'''

path_to_check = Path(__file__).parent.parent / "export"


def check_table_triplets_file(
        triplets_file: Path,
) -> bool:
    if not triplets_file.exists():
        warnings.warn(
            f"table triplets file does not exist (path searched: {triplets_file})"
        )
        return False

    data = np.load(triplets_file.as_posix())
    table_val = data["val"]
    table_T = data["temp"]
    mu = data["mu"]

    T_in_file = np.unique(table_T)

    # Compare with the current T, Ec
    print(f"First val: {table_val[0]}, Last val: {table_val[-1]}")
    print("mu = " + str(mu))
    print(T_in_file)
    print(f"length is {len(table_val)}")


if __name__ == "__main__":
    check_table_triplets_file(
        path_to_check / "table_triplets_Tstd6_20.npz",
    )
