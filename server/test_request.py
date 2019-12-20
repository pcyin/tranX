import requests
import urllib.parse

url = 'http://localhost:8081/parse/conala/'
utterance = input()
print(requests.get(url + urllib.parse.quote(utterance)).json())
