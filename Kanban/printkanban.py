import requests

headers = {
    'Authorization': f'bearer YOUR_GITHUB_TOKEN',
    'Content-Type': 'application/json',
}

# Example GraphQL query for projects in an organization
query = """
query {
  organization(login: "your-organization") {
    projectsV2(first: 10) {
      nodes {
        id
        title
        number
        items(first: 100) {
          nodes {
            id
            content {
              ... on Issue {
                title
                number
              }
              ... on PullRequest {
                title
                number
              }
            }
            fieldValues(first: 20) {
              nodes {
                ... on ProjectV2ItemFieldTextValue {
                  text
                  field {
                    name
                  }
                }
                ... on ProjectV2ItemFieldSingleSelectValue {
                  name
                  field {
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

data = {'query': query}
response = requests.post('https://api.github.com/graphql', json=data, headers=headers)
result = response.json()

# Process the results
for project in result['data']['organization']['projectsV2']['nodes']:
    print(f"Project: {project['title']} (#{project['number']})")
    
    for item in project['items']['nodes']:
        # Get the item title and number
        if item['content']:
            title = item['content'].get('title', 'Untitled')
            number = item['content'].get('number', 'N/A')
            print(f"  - {title} (#{number})")
            
            # Display field values (including status columns)
            for field_value in item['fieldValues']['nodes']:
                if 'field' in field_value and 'name' in field_value['field']:
                    field_name = field_value['field']['name']
                    if 'text' in field_value:
                        print(f"    - {field_name}: {field_value['text']}")
                    elif 'name' in field_value:
                        print(f"    - {field_name}: {field_value['name']}")