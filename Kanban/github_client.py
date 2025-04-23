# github_project.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class GithubProject:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.owner = os.getenv("GITHUB_OWNER")
        self.project_number = int(os.getenv("PROJECT_NUMBER"))
        self.api_url = "https://api.github.com/graphql"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        # Get the essential IDs we need
        self.project_id = self._get_project_id()
        self.fields = self._get_fields()
    
    def _request(self, query, variables=None):
        """Send GraphQL request and return response data"""
        if variables is None:
            variables = {}
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"query": query, "variables": variables}
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
        """Get all fields and their options"""
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
            name = field["name"]
            field_data = {"id": field["id"]}
            
            # Add options if this is a select field
            if "options" in field:
                field_data["options"] = {opt["name"]: opt["id"] for opt in field["options"]}
                
            fields[name] = field_data
            
        return fields
    
    def create_task(self, task):
        """Create a task with all provided fields"""
        # 1. Create the item
        item_id = self._create_item(task["title"], task.get("description"))
        
        # 2. Set the status if present in task and fields
        if "status" in task and "Status" in self.fields and "options" in self.fields["Status"]:
            self._update_select_field(item_id, "Status", task["status"])
        
        # 3. Set priority if present
        if "priority" in task and "Priority" in self.fields and "options" in self.fields["Priority"]:
            self._update_select_field(item_id, "Priority", task["priority"])
        
        # 4. Set size if present
        if "size" in task and "Size" in self.fields and "options" in self.fields["Size"]:
            self._update_select_field(item_id, "Size", task["size"])
        
        # 5. Set dates if present
        if "start_date" in task and "Start date" in self.fields:
            self._update_date_field(item_id, "Start date", task["start_date"])
            
        if "due_date" in task and "Due date" in self.fields:
            self._update_date_field(item_id, "Due date", task["due_date"])
            
        return item_id
    
    def _create_item(self, title, description=None):
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
            "body": description
        })
        return result["data"]["addProjectV2DraftItem"]["projectItem"]["id"]
    
    def _update_select_field(self, item_id, field_name, option_value):
        """Update a select field with the given option"""
        field_id = self.fields[field_name]["id"]
        option_id = self.fields[field_name]["options"][option_value]
        
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: { 
              singleSelectOptionId: $optionId
            }
          }) {
            projectItem {
              id
            }
          }
        }
        """
        self._request(mutation, {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "optionId": option_id
        })
    
    def _update_date_field(self, item_id, field_name, date_string):
        """Update a date field (format: YYYY-MM-DD)"""
        field_id = self.fields[field_name]["id"]
        
        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $date: Date!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: { 
              date: $date
            }
          }) {
            projectItem {
              id
            }
          }
        }
        """
        self._request(mutation, {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "date": date_string
        })