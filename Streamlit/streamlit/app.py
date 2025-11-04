# redirect shim for Streamlit Cloud
from pathlib import Path
import os, runpy

# find the real app one level up
real_app = Path(__file__).resolve().parent.parent / "app.py"

if not real_app.exists():
    raise FileNotFoundError(f"Cannot find real app: {real_app}")

# change directory so local loads (joblib.load etc.) still work
os.chdir(real_app.parent)

# run the actual app
runpy.run_path(str(real_app), run_name="__main__")

