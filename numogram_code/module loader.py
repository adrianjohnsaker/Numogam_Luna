import os
import importlib
import sys

MODULES_PATH = "/storage/emulated/0/Android/data/com.yourapp.luna/assets/"  # Adjust the path to your app's assets folder

def load_modules():
    """
    Recursively loads Python modules from the assets folder, including subdirectories.
    """
    loaded_modules = []

    if not os.path.exists(MODULES_PATH):
        print(f"Error: {MODULES_PATH} does not exist.")
        return

    sys.path.append(MODULES_PATH)  # Add assets to system path

    for root, _, files in os.walk(MODULES_PATH):
        for filename in files:
            if filename.endswith(".py") and filename != "module_loader.py":
                module_path = os.path.relpath(os.path.join(root, filename), MODULES_PATH)
                module_name = module_path.replace(os.sep, ".")[:-3]  # Convert path to module name

                try:
                    module = importlib.import_module(module_name)
                    loaded_modules.append(module_name)
                    print(f"✅ Loaded: {module_name}")
                except Exception as e:
                    print(f"❌ Failed: {module_name} - {e}")

    print(f"\n🔹 Modules Loaded: {loaded_modules}")

if __name__ == "__main__":
    load_modules()
