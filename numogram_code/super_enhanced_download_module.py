import os
import subprocess

class JavaBridge:
    def __init__(self):
        self.bridge_path = "assets/python_bridge.jar"  # Ensure correct path

    def send_message(self, message):
        try:
            # Call the Java JAR with the message
            result = subprocess.run(
                ["java", "-jar", self.bridge_path, message], 
                capture_output=True, text=True
            )
            return result.stdout.strip()  # Return Java's response
        except Exception as e:
            return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    bridge = JavaBridge()
    response = bridge.send_message("Hello from Python!")
    print("Java response:", response)

import os
import requests
import json

REPO_BASE_URL = "https://raw.githubusercontent.com/your-username/numogram-luna/main/"
MANIFEST_URL = f"{REPO_BASE_URL}modules/modules_manifest.json"

def download_content(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

class ModuleDownloader:
    def __init__(self):
        self.module_dir = os.path.join(os.getcwd(), "modules")
        os.makedirs(self.module_dir, exist_ok=True)
        self.download_modules()

    def download_modules(self):
        try:
            manifest_json = download_content(MANIFEST_URL)
            manifest = json.loads(manifest_json)
            modules = manifest["modules"]

            for module_info in modules:
                module_name = module_info["name"]
                version = module_info["version"]
                print(f"Downloading {module_name} v{version}")

                module_url = f"{REPO_BASE_URL}modules/{module_name}.py"
                module_content = download_content(module_url)

                module_file_path = os.path.join(self.module_dir, f"{module_name}.py")
                with open(module_file_path, "w") as module_file:
                    module_file.write(module_content)

                print(f"Installed {module_name}")

            print("All modules downloaded successfully")
        except Exception as e:
            print(f"Error: {e}")
            print("Failed to download all modules")

if __name__ == "__main__":
    ModuleDownloader()
