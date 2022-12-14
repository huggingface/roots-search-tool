import json
import os
import pprint

import requests

pp = pprint.PrettyPrinter(indent=4)

query = "hello world"

# run the following in commandline, replace address with your server address:
# export address="http://12.345.678.910:8080"

output = requests.post(
    os.environ.get("address"),
    headers={"Content-type": "application/json"},
    data=json.dumps({"query": query}),
    timeout=60,
)

results = json.loads(output.text)

pp.pprint(results)
