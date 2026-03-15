import shutil

for n in range(2, 20):
    original = ("table_computer_Tstd1_20.py")
    new_file = f"table_computer_Tstd{n}_20.py"
    shutil.copy(original, new_file)

print("Copies created successfully!")
