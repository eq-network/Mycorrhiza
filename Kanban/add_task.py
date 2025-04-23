# create_tasks.py
from github_client import GitHubProjectClient

# Task definitions (you can move these to a separate file)
tasks = [
    {
        "title": "[HIGH] Implement Core Graph-Based Representation System",
        "priority": "High",
        "size": "Medium",
        "start_date": "2025-04-25",
        "due_date": "2025-04-28",
        "description": "Implement the fundamental graph structures that will represent agent networks, information state, and delegation relationships."
    },
    {
        "title": "[HIGH] Implement Core Transformation Compositions",
        "priority": "High",
        "size": "Medium",
        "start_date": "2025-04-28",
        "due_date": "2025-05-01",
        "description": "Create the composition mechanisms that allow transformations to be combined and sequenced."
    },
    # Add more tasks...
]

def main():
    client = GitHubProjectClient()
    
    for task in tasks:
        print(f"Creating task: {task['title']}")
        
        # Create the draft item
        item_id = client.create_draft_item(task['title'])
        
        # Set the priority
        client.set_select_field(item_id, "Priority", task['priority'])
        
        # Set the size
        client.set_select_field(item_id, "Size", task['size'])
        
        # Set dates
        if 'start_date' in task:
            client.set_date_field(item_id, "Start date", task['start_date'])
        if 'due_date' in task:
            client.set_date_field(item_id, "Due date", task['due_date'])
        
        print(f"Successfully created task: {task['title']}")

if __name__ == "__main__":
    main()