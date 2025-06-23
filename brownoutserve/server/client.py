import requests
import threading

prompt = 'What is Huawei?'
response = requests.get("http://127.0.0.1:5000/generate", params={"prompt": prompt})
print(response.status_code)  # HTTP status code
print(prompt)
print(response.json()['result'])  # Response content
print(response.json()['latency'])  # Response content


