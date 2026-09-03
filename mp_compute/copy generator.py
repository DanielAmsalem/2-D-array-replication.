import shutil
from itertools import chain

for n in range(550,600,50):
    original = ("table_computer_low_res_metal.py")
    new_file = f"table_computer_Tstd{n}_20_ratio1_Cg10.py"
    shutil.copy(original, new_file)

print("Copies created successfully!")

