import json

with open('A2/models/results.json', 'r') as file:
    res = json.load(file)

print(res)