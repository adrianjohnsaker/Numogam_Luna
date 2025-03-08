import os
import importlib
import sys
import traceback

# Adjust this path dynamically if needed
MODULES_PATH = "/storage/emulated/0/Android/data/com.yourapp.luna/assets/"  

def load_modules():
    """
    Recursively loads Python modules from the assets folder, including subdirectories.
    """
    loaded_modules = []

    if not os.path.exists(MODULES_PATH):
        print(f"❌ Error
