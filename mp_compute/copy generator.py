import shutil
from itertools import chain

for n in chain(range(1, 20, 2), [20]):
    original = ("table_computer_fixmiddle.py")
    new_file = f"table_computer_Tmid15_4_Tstd{n}_20_ratio1_Cg2.py"
    shutil.copy(original, new_file)
    new_file = f"table_computer_Tmid15_4_Tstd{n}_20_ratio1_Cg10.py"
    shutil.copy(original, new_file)

print("Copies created successfully!")
