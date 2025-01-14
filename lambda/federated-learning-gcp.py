import functions_framework
import io
import json 
import torch
import requests
from io import BytesIO
import base64
import torch.nn as nn

# A method to fetch the model from IPFS
def fetch_model_from_ipfs(index, cid): 
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

        # Decode the Base64-encoded content back into binary data 
        file_bytes = base64.b64decode(file_content_base64) 
        
        with open(f"model{index}.pth", "wb") as f:
            f.write(file_bytes)
        print(f"File saved as model{index}.pth for inspection.")  
        
    else:
        raise Exception(f"Failed to fetch model from IPFS. Status code: {response.status_code}, Response: {response.content}")

# A method to upload the model to IPFS
def upload_model_to_ipfs(index, model):
    # Save the model to a buffer
    buffer = BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    
    # Encode the model to Base64
    model_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    #print(model_base64)
    
    # Call IPFS lambda handler to upload the model
    url = "https://emviofaj63.execute-api.us-east-1.amazonaws.com/default/ipfs-handler"
    payload = {
        "action": "0",
        "fileName": f"aggregated_model{index}.pth",
        "fileContent": model_base64
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Model uploaded to IPFS with CID: {response.json()['cid']}")
    else:
        raise Exception(f"Failed to upload model to IPFS. Status code: {response.status_code}, Response: {response.content}")
    
    return response.json()['cid']

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

# A method to upload the hash to the blockchain
def upload_hash_to_blockchain(index, hashToStore):
    # Call blockchain lambda handler to upload the hash
    index = str(index) 
    hashToStore = str(hashToStore)
    url = "https://r3h9ia9po3.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
    payload = {
        "action": "0",
        "clientIndex": index,
        "content": hashToStore
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Hash uploaded to blockchain for client {index}")
    else:
        raise Exception(f"Failed to upload hash to blockchain. Status code: {response.status_code}, Response: {response.content}")

# A simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)
    
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


# Function to aggregate models
def aggregate_model(index, updated_model, to_be_updated):  
    models = []

    #Create model container 1
    model = Model()
    model.load_state_dict(to_be_updated)
    models.append(model)

    #Create model container 2
    model = Model()
    model.load_state_dict(updated_model)
    models.append(model)

    # Simple averaging of models
    aggregated_state_dict = models[0].state_dict()
    for key in aggregated_state_dict.keys():
        for model in models[1:]:
            aggregated_state_dict[key] += model.state_dict()[key]
        aggregated_state_dict[key] /= len(models)

    # Save the aggregated model
    aggregated_model = Model()
    aggregated_model.load_state_dict(aggregated_state_dict)
    torch.save(aggregated_model.state_dict(), f'aggregated_model{index}.pth')

    return aggregated_model

# A method to display the weights of the model
def display_modle_weights(loaded_model):
    for name, param in loaded_model.items():
            print(f"{name}: {param}")

# Lambda handler
@functions_framework.http
def lambda_handler(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    updatedClientIndex = int(request_json.get('updatedClientIndex'))
    blockchainEnabled = int(request_json.get('blockchainEnabled'))
    client_model_cids = []

    # Get hashes from blockchain
    for i in range(3): 
        client_model_cids.append(fetch_hash_from_blockchain(str(i))) 

    # Get models from IPFS
    client_model = [] 
    for i in range(3):
        print(i)
        fetch_model_from_ipfs(i, client_model_cids[i])
    
    # Identify the updated model
    for i in range(3):
        if i == updatedClientIndex:
            updated_model = torch.load(f"model{i}.pth")
        else:
            client_model.append(torch.load(f"model{i}.pth"))

    #Print before aggregation
    print("Before aggregation")
    display_modle_weights(updated_model)
    display_modle_weights(client_model[0])
    display_modle_weights(client_model[1])

    # Perform aggregation
    aggregated_models = [] 
    aggregated_model_1 = aggregate_model(0, updated_model, client_model[0])
    aggregated_model_2 = aggregate_model(1, updated_model, client_model[1])

    aggregated_models.append(torch.load("aggregated_model0.pth"))
    aggregated_models.append(torch.load("aggregated_model1.pth"))

    #print the model weights
    print("After aggregation")
    display_modle_weights(updated_model)
    display_modle_weights(aggregated_models[0])
    display_modle_weights(aggregated_models[1]) 
    
    # Upload the updated models to IPFS
    count = 0
    hashes = []
    for i in range(3):
        if i != updatedClientIndex:
            thisHash = upload_model_to_ipfs(i, aggregated_models[count])
            hashes.append(thisHash)
            count += 1
    
    # Upload the hash to blockchain
    count = 0
    if blockchainEnabled: 
      for i in range(3):
          if i != updatedClientIndex:
              upload_hash_to_blockchain(i, hashes[count])
              #print(hashes[count])
              count += 1

      print("Uploaded to blockchain")

    
    return "true"
    
'''
{
    "updatedClientIndex": "0",
    "blockchainEnabled": "0"
}
'''