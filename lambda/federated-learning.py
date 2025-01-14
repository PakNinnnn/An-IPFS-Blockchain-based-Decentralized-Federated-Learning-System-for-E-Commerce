import io
import json 
import torch
import requests
from io import BytesIO
import base64
import torch.nn as nn

def fetch_model_from_ipfs(index, cid):
    url = "https://2zk0vq0ll6.execute-api.us-east-1.amazonaws.com/default/ipfs-file-streaming"
    payload = {
        "action": "1",
        "cid": cid
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        
        response_data = response.json()
        file_content_base64 = response_data.get("fileContent") 

        # Decode the Base64-encoded content back into binary data
        try:
            file_bytes = base64.b64decode(file_content_base64)
        except Exception as e:
            raise Exception(f"Failed to decode file content: {e}")
        
        with open(f"model{index}.pth", "wb") as f:
            f.write(file_bytes)
        print(f"File saved as model{index}.pth for inspection.")

        # Load the PyTorch model directly from the decoded binary data
        try:
            model = torch.load(io.BytesIO(file_bytes), map_location=torch.device('cpu'))  # Map to CPU for safety
        except Exception as e:
            raise Exception(f"Failed to load PyTorch model: {e}")
        
        return model
    else:
        raise Exception(f"Failed to fetch model from IPFS. Status code: {response.status_code}, Response: {response.content}")

def upload_model_to_ipfs(index, model):
    # Save the model to a buffer
    buffer = BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    
    # Encode the model to Base64
    model_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    #print(model_base64)
    
    # Upload the model to IPFS
    url = "https://2zk0vq0ll6.execute-api.us-east-1.amazonaws.com/default/ipfs-file-streaming"
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

def fetch_hash_from_blockchain(indexStr):
    url = "https://82o2i4mwfc.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
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

def upload_hash_to_blockchain(index, hash):
    url = "https://82o2i4mwfc.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
    payload = {
        "action": "0",
        "clientIndex": index,
        "hash": hash
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Hash uploaded to blockchain for client {index}")
    else:
        raise Exception(f"Failed to upload hash to blockchain. Status code: {response.status_code}, Response: {response.content}")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Function to aggregate models
def aggregate_model(index, updated_model, to_be_updated):  
    models = []

    model = SimpleModel()
    model.load_state_dict(to_be_updated)
    models.append(model)

    model = SimpleModel()
    model.load_state_dict(updated_model)
    models.append(model)

    # Simple averaging of models
    aggregated_state_dict = models[0].state_dict()
    for key in aggregated_state_dict.keys():
        for model in models[1:]:
            aggregated_state_dict[key] += model.state_dict()[key]
        aggregated_state_dict[key] /= len(models)

    aggregated_model = SimpleModel()
    aggregated_model.load_state_dict(aggregated_state_dict)
    torch.save(aggregated_model.state_dict(), f'aggregated_model{index}.pth')

    return aggregated_model

def display_modle_weights(loaded_model):
    for name, param in loaded_model.items():
            print(f"{name}: {param}")

# Lambda handler
#def lambda_handler(event, context):
#Initiate client should upload hash to blockchain by itself first
def lambda_handler(updatedClientIndex): 
    
    client_model_cids = []
    for i in range(3):
        #call another api to get hash from blockchain
        client_model_cids.append(fetch_hash_from_blockchain(str(i)))

    client_model_cids[0] = "QmdeCEbBww93j2CAoDV1qjrmaWvW6S73dRETiVHdsCxva9"
    #updated_model_cid = client_model_cids[updatedClientIndex];
    #client_model_cids = [cid for i, cid in enumerate(client_model_cids) if i != updatedClientIndex]
    
    client_model = []
    
    for i in range(3):
        print(i)
        fetch_model_from_ipfs(i, client_model_cids[i])
    
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

    aggregated_models = []
    
    # Perform aggregation
    aggregated_model_1 = aggregate_model(0, updated_model, client_model[0])
    aggregated_model_2 = aggregate_model(1, updated_model, client_model[1])

    aggregated_models.append(torch.load("aggregated_model0.pth"))
    aggregated_models.append(torch.load("aggregated_model1.pth"))

    #print the model weights
    print("After aggregation")
    display_modle_weights(updated_model)
    display_modle_weights(aggregated_models[0])
    display_modle_weights(aggregated_models[1]) 
    
    # Save the models as .pth files
    #torch.save(updated_model.state_dict(), "aggregated_model_1.pth")
    #torch.save(aggregated_model_1.state_dict(), "aggregated_model_2.pth")
    
    # Upload the models to IPFS
    count = 0
    for i in range(3):
        if i != updatedClientIndex:
            upload_model_to_ipfs(i, aggregated_models[count])
            count += 1
    
    # Upload the hash to blockchain
    '''
    for i in range(3):
        if i != updatedClientIndex:
            upload_hash_to_blockchain(i, f"aggregated_model{i}.pth")
    '''

# Test the lambda handler
lambda_handler(1) 
 