import functions_framework
import io
import json 
import torch
import requests
from io import BytesIO
import base64
import torch.nn as nn

# A method to fetch the encoded model from IPFS
def fetch_model_from_ipfs(cid): 
    # Call IPFS lambda handler to fetch the model
    url = "https://emviofaj63.execute-api.us-east-1.amazonaws.com/default/ipfs-handler"
    payload = {
        "action": "1",
        "cid": cid
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200: 
        response_data = response.json()
        file_content_base64 = (response_data.get("fileContent"))

        return file_content_base64
        
    else:
        raise Exception(f"Failed to fetch model from IPFS. Status code: {response.status_code}, Response: {response.content}")

# A method to fetch the hash from the blockchain
def fetch_hash_from_blockchain(indexStr): 
    # Call blockchain lambda handler to fetch the hash
    url = "https://r3h9ia9po3.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
    payload = {
        "action": "1",
        "clientIndex": indexStr
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(response.content)
        return response.json()["hash"]
    else:
        raise Exception(f"Failed to fetch model from blockchain. Status code: {response.status_code}, Response: {response.content}")

# Th real model used
class Model(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)
        
    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return pred.squeeze()


# A method to display the weights of the model
def display_modle_weights(loaded_model):
    for name, param in loaded_model.items():
            print(f"{name}: {param}")

# Lambda handler
@functions_framework.http
def lambda_handler(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    clientIndex = str(request_json.get('clientIndex'))  

    # Get hash from blockchain
    cid = fetch_hash_from_blockchain(clientIndex)

    # Get encoded models from IPFS
    base64_model = fetch_model_from_ipfs(cid)
    
    return base64_model