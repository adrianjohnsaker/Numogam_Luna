import requests
import json

class GitHubMemoryManager:
    def __init__(self, repo_owner, repo_name, access_token):
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def fetch_memory(self, file_path):
        """Fetch the existing memory JSON from GitHub."""
        response = requests.get(self.base_url + file_path, headers=self.headers)
        if response.status_code == 200:
            file_data = response.json()
            content = file_data["content"]
            return json.loads(content)
        else:
            print(f"Failed to fetch memory: {response.status_code}")
            return None

    def update_memory(self, file_path, new_memory_entry, commit_message):
        """Update the memory JSON file in GitHub."""
        current_memory = self.fetch_memory(file_path)
        
        if current_memory:
            # Append new memory entry
            current_memory["memory"].append(new_memory_entry)
            updated_content = json.dumps(current_memory, indent=2)

            # Get current SHA for the file
            response = requests.get(self.base_url + file_path, headers=self.headers)
            sha = response.json()["sha"]

            # Update file on GitHub
            update_data = {
                "message": commit_message,
                "content": updated_content.encode("utf-8").decode("ascii"),
                "sha": sha
            }
            response = requests.put(self.base_url + file_path, headers=self.headers, data=json.dumps(update_data))

            if response.status_code == 200:
                print("Memory updated successfully!")
            else:
                print(f"Failed to update memory: {response.status_code}")
        else:
            print("No existing memory to update!")

# Example usage
repo_owner = "YourGitHubUsername"
repo_name = "YourRepositoryName"
access_token = "YourGitHubAccessToken"
file_path = "path/to/memory.json"

github_manager = GitHubMemoryManager(repo_owner, repo_name, access_token)

# Example memory entry
new_memory = {
    "test_id": "new_test_003",
    "test_name": "Example Test",
    "date": "2025-03-04",
    "objectives": ["Test GitHub API integration with memory updates."],
    "outcomes": {
        "successes": ["Integration was successful."],
        "improvements": ["Optimize API response time."]
    },
    "key_learnings": ["External memory provides persistent knowledge."]
}

github_manager.update_memory(file_path, new_memory, "Added a new test memory entry.")
