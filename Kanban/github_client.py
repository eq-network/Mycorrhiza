# github_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class GitHubProjectClient:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.owner = os.getenv("GITHUB_OWNER")
        self.project_number = int(os.getenv("PROJECT_NUMBER"))
        self.url = "https://api.github.com/graphql"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.project_id = self._get_project_id()
        self.field_info = self._get_field_info()
        
    def _get_project_id(self):
        # Query to get the project ID
        query = """
        query($owner: String!, $number: Int!) {
          user(login: $owner) {
            projectV2(number: $number) {
              id
            }
          }
        }
        """
        variables = {
            "owner": self.owner,
            "number": self.project_number
        }
        
        response = self._execute_query(query, variables)
        return response["data"]["user"]["projectV2"]["id"]
    
    def _get_field_info(self):
        # Get all field IDs and their options
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
        
        variables = {
            "projectId": self.project_id
        }
        
        response = self._execute_query(query, variables)
        fields = response["data"]["node"]["fields"]["nodes"]
        
        # Transform into a more usable structure
        field_info = {}
        for field in fields:
            field_info[field["name"]] = {
                "id": field["id"],
                "options": {opt["name"]: opt["id"] for opt in field.get("options", [])} if "options" in field else None
            }
            
        return field_info
    
    def _execute_query(self, query, variables):
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"query": query, "variables": variables}
        )
        
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    
    def create_draft_item(self, title, description=None):
        mutation = """
        mutation($projectId: ID!, $title: String!) {
          addProjectV2DraftItem(input: {
            projectId: $projectId
            title: $title
          }) {
            projectItem {
              id
            }
          }
        }
        """
        
        variables = {
            "projectId": self.project_id,
            "title": title
        }
        
        response = self._execute_query(mutation, variables)
        return response["data"]["addProjectV2DraftItem"]["projectItem"]["id"]
    
    def set_select_field(self, item_id, field_name, option_value):
        field_id = self.field_info[field_name]["id"]
        option_id = self.field_info[field_name]["options"][option_value]
        
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
        
        variables = {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "optionId": option_id
        }
        
        return self._execute_query(mutation, variables)
    
    def set_date_field(self, item_id, field_name, date_string):
        field_id = self.field_info[field_name]["id"]
        
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
        
        variables = {
            "projectId": self.project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "date": date_string  # Format: YYYY-MM-DD
        }
        
        return self._execute_query(mutation, variables)