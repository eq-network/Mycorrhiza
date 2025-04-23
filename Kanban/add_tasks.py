# add_tasks.py
import os
import requests
import re
from dotenv import load_dotenv

# Config (explicit is better than implicit)
load_dotenv("Kanban/.env")
TOKEN = os.getenv("GITHUB_TOKEN")
ORG_NAME = "eq-network"  # Organization name - hardcoded for clarity
PROJECT_NUMBER = int(os.getenv("PROJECT_NUMBER", "1"))
TASKS_FILE = "text/InitialKanban.md"

# Simple API request function
def github_request(query, variables=None):
    response = requests.post(
        "https://api.github.com/graphql",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"query": query, "variables": variables or {}}
    )
    data = response.json()
    if "errors" in data:
        raise Exception(f"API Error: {data['errors'][0]['message']}")
    return data

# Get organization project ID - explicitly use organization query
def get_project_id():
    query = """
    query($org: String!, $number: Int!) {
      organization(login: $org) {
        projectV2(number: $number) {
          id
        }
      }
    }
    """
    result = github_request(query, {"org": ORG_NAME, "number": PROJECT_NUMBER})
    return result["data"]["organization"]["projectV2"]["id"]

# Get project fields
def get_project_fields(project_id):
    query = """
    query($id: ID!) {
      node(id: $id) {
        ... on ProjectV2 {
          fields(first: 20) {
            nodes {
              ... on ProjectV2Field { id, name }
              ... on ProjectV2SingleSelectField {
                id, name, options { id, name }
              }
            }
          }
        }
      }
    }
    """
    result = github_request(query, {"id": project_id})
    
    fields = {}
    for field in result["data"]["node"]["fields"]["nodes"]:
        data = {"id": field["id"]}
        if "options" in field:
            data["options"] = {opt["name"]: opt["id"] for opt in field["options"]}
        fields[field["name"]] = data
    
    return fields

# Add task and set fields
def add_task(project_id, fields, task):
    # Create item
    create_query = """
    mutation($projectId: ID!, $title: String!, $body: String) {
      addProjectV2DraftItem(input: {
        projectId: $projectId, title: $title, body: $body
      }) {
        projectItem { id }
      }
    }
    """
    result = github_request(create_query, {
        "projectId": project_id,
        "title": task["title"],
        "body": task["description"]
    })
    item_id = result["data"]["addProjectV2DraftItem"]["projectItem"]["id"]
    
    # Set priority field
    if "Priority" in fields and task["priority"] in fields["Priority"].get("options", {}):
        set_field(project_id, item_id, fields["Priority"]["id"], 
                 fields["Priority"]["options"][task["priority"]])
    
    # Set size field
    if "Size" in fields and task["size"] in fields["Size"].get("options", {}):
        set_field(project_id, item_id, fields["Size"]["id"], 
                 fields["Size"]["options"][task["size"]])
    
    return item_id

# Set a field value
def set_field(project_id, item_id, field_id, option_id):
    query = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId, itemId: $itemId, fieldId: $fieldId,
        value: { singleSelectOptionId: $optionId }
      }) {
        clientMutationId
      }
    }
    """
    github_request(query, {
        "projectId": project_id,
        "itemId": item_id,
        "fieldId": field_id,
        "optionId": option_id
    })

# Parse tasks from markdown
def extract_tasks():
    # Read file
    try:
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise Exception(f"Task file not found: {TASKS_FILE}")
    
    # Find all tasks
    tasks = []
    task_blocks = re.split(r'### \[(HIGH|MED|LOW)\] ', content)[1:]  # Skip first empty item
    
    # Process each task block
    for i in range(0, len(task_blocks), 2):
        if i+1 >= len(task_blocks):
            break
            
        priority = task_blocks[i]
        content_block = task_blocks[i+1]
        
        # Get title
        title_match = re.match(r'(.*?)(?:\n|$)', content_block)
        title = title_match.group(1).strip() if title_match else "Untitled Task"
        
        # Get description
        desc_match = re.search(r'\*\*Description:\*\*(.*?)(?=\*\*|\Z)', content_block, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Get size
        size_match = re.search(r'\*\*Estimated Effort:\*\*\s*([SML])', content_block)
        size = {"S": "Small", "M": "Medium", "L": "Large"}[size_match.group(1)] if size_match else "Medium"
        
        tasks.append({
            "title": f"[{priority}] {title}",
            "description": description,
            "priority": {"HIGH": "High", "MED": "Medium", "LOW": "Low"}[priority],
            "size": size
        })
    
    return tasks

# Main function
def main():
    try:
        print(f"Getting project from organization: {ORG_NAME}...")
        project_id = get_project_id()
        print(f"Project ID: {project_id}")
        
        fields = get_project_fields(project_id)
        print(f"Found fields: {', '.join(fields.keys())}")
        
        tasks = extract_tasks()
        print(f"Found {len(tasks)} tasks to add")
        
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] Adding: {task['title']}")
            item_id = add_task(project_id, fields, task)
            print(f"Added successfully (ID: {item_id[:8]}...)")
        
        print("✓ All tasks added to project")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()