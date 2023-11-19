import requests

api_url = 'http://127.0.0.1:5000/predict?sepal_length=5&sepal_width=1&petal_length=1&petal_width=1'
response = requests.get(api_url)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")