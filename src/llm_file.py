import requests
import json
import ast

def call_server(input_string):
    url = 'http://172.27.15.38:5015/process'
    headers = {'Content-Type': 'application/json'}
    data = {'input_string': input_string}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        return result['result']
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None

# extract the json string from the response
def extract_between_braces(input_string):
    start = input_string.find('{')
    end = input_string.find('}', start) + 1
    if start > 0 and end > start:
        return input_string[start:end]
    else:
        return None
   

if __name__ == '__main__':
    # Example usage
    while True:
        user_input = input("User: ")
        
        result = call_server(user_input)

        if result:
            result = extract_between_braces(result)
            print(result)
            print(type(result))
            result = ast.literal_eval(result)
            print(result["function"])
            print(result["response"])
