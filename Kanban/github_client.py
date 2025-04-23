# github_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GithubProject:  # Changed the name to match your import
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.owner = os.getenv("GITHUB_OWNER")
        self.project_number = int(os.getenv("PROJECT_NUMBER"))
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        # Get project ID and fields on initialization
        self.project_id = self._get_project_id()
        self.fields = self._get_fields()
    
    def _request(self, query, variables=None):
        """Send a GraphQL request to GitHub API"""
        response = requests.post(
            "https://api.github.com/graphql",
            headers=self.headers,
            json={"query": query, "variables": variables or {}}
        )
        response.raise_for_status()
        return response.json()
    
    def _get_project_id(self):
        """Get project ID from project number"""
        query = """
        query($owner: String!, $number: Int!) {
          user(login: $owner) {
            projectV2(number: $number) {
              id
            }
          }
        }
        """
        result = self._request(query, {
            "owner": self.owner,
            "number": self.project_number
        })
        return result["data"]["user"]["projectV2"]["id"]
    
    def _get_fields(self):
        """Get field IDs and options"""
        query = """
        query($projectId: ID!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              fields(first: 20) {
                nodes {
                  ... on ProjectV2Field {
                    id
                    name
                  }
                  ... on ProjectV2SingleSelectField {
                    id
                    name
                    options {
                      id
                      name
                    }
                  }
                  ... on ProjectV2DateField {
                    id
                    name
                  }
                }
              }
            }
          }
        }
        """
        result = self._request(query, {"projectId": self.project_id})
        
        fields = {}
        for field in result["data"]["node"]["fields"]["nodes"]:
            field_data = {"id": field["id"]}
            if "options" in field:
                field_data["options"] = {opt["name"]: opt["id"] for opt in field["options"]}
            fields[field["name"]] = field_data
        return fields
    
    # Added create_task method to match what's called in task_parser.py
    def create_task(self, task):
        """Add a task to GitHub Project board"""
        # Create the item
        item_id = self._create_item(task.get("title", "Untitled Task"), task.get("description"))
        
        # Set status field
        if "Status" in self.fields and "options" in self.fields["Status"]:
            if task.get("status") in self.fields["Status"]["options"]:
                self._set_select(item_id, "Status", task["status"])
        
        # Set priority field
        if "Priority" in self.fields and "options" in self.fields["Priority"] and "priority" in task:
            self._set_select(item_id, "Priority", task["priority"])
        
        # Set size field
        if "Size" in self.fields and "options" in self.fields["Size"] and "size" in task:
            self._set_select(item_id, "Size", task["size"])
        
        # Set due date field
        if "Due date" in self.fields and "due_date" in task:
            self._set_date(item_id, "Due date", task["due_date"])
        
        return item_id
    
    def _create_item(self, title, body=None):
        """Create a draft item"""
        mutation = """
        mutation($projectId: ID!, $title: String!, $body: String) {
          addProjectV2DraftItem(input: {
            projectId: $projectId
            title: $title
            body: $body
          }) {
            projectItem {
              id
            }
          }
        }
        """
        result = self._request(mutation, {
            "projectId": self.project_id,
            "title": title,
            "body": body
        })
        return result["data"]["addProjectV2DraftItem"]["projectItem"]["id"]
    
    def _set_select(self, item_id, field_name, option_value):
        """Set a select field value"""
        field_id = self.fields[field_name]["id"]
        option_id = self.fields[field_name]["options"][option_value]
        
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: { singleSelectOptionId: $optionId }
          }) {
            projectItem { id }
          }
        }
        """
        self._request(mutation, {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "optionId": option_id
        })
    
    def _set_date(self, item_id, field_name, date_string):
        """Set a date field"""
        field_id = self.fields[field_name]["id"]
        
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $date: Date!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: { date: $date }
          }) {
            projectItem { id }
          }
        }
        """
        self._request(mutation, {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "date": date_string
        })