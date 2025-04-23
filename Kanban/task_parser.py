# process_tasks.py
import re
from github_project import GithubProject

def parse_tasks(markdown_file):
    """Extract tasks from the kanban markdown file"""
    with open(markdown_file, 'r') as f:
        content = f.read()
    
    # Find all task cards with headers like "### [HIGH] Task Title"
    card_pattern = r'### \[(HIGH|MED|LOW)\] (.*?)\n(.*?)(?=### \[|\Z)'
    cards = re.findall(card_pattern, content, re.DOTALL)
    
    tasks = []
    for priority_code, title, body in cards:
        # Map priority codes to actual values in project
        priority_map = {"HIGH": "High", "MED": "Medium", "LOW": "Low"}
        
        task = {
            "title": f"[{priority_code}] {title.strip()}",
            "description": body.strip(),
            "priority": priority_map[priority_code],
            "status": "To do"  # Default status
        }
        
        # Extract size if present
        size_match = re.search(r'Estimated Effort: ([SML])', body)
        if size_match:
            size_map = {"S": "Small", "M": "Medium", "L": "Large"}
            task["size"] = size_map[size_match.group(1)]
        
        tasks.append(task)
    
    return tasks

def main():
    # Parse tasks from markdown
    tasks = parse_tasks("democratic_mechanism_simulation_kanban.md")
    print(f"Found {len(tasks)} tasks to add")
    
    # Create client and add tasks
    client = GithubProject()
    
    for task in tasks:
        print(f"Adding task: {task['title']}")
        try:
            item_id = client.create_task(task)
            print(f"✓ Added successfully")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")

if __name__ == "__main__":
    main()