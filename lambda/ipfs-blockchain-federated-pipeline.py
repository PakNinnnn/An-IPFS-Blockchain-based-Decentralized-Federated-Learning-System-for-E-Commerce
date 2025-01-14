import json
import requests
import base64
import io
from io import BytesIO

def upload_model_to_ipfs(index, model_base64):
    url = "https://emviofaj63.execute-api.us-east-1.amazonaws.com/default/ipfs-handler"
    payload = {
        "action": "0",
        "fileName": f"updated_model{index}.pth",
        "fileContent": model_base64
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Model uploaded to IPFS with CID: {response.json()['cid']}")
    else:
        raise Exception(f"Failed to upload model to IPFS. Status code: {response.status_code}, Response: {response.content}")
    
    return response.json()['cid']

def upload_hash_to_blockchain(index, hashToStore):
    index = str(index)
    hashToStore = str(hashToStore)
    #url = "https://82o2i4mwfc.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
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
 
def lambda_handler(event, context): 

    body = event.get('body', '{}') 
    parsed_body = json.loads(body)
     
    updatedClientIndex = str(parsed_body.get('updatedClientIndex'))
    blockchainEnabled = str(parsed_body.get('blockchainEnabled'))
    modelEncoded = str(parsed_body.get('modelEncoded'))

    newHash = upload_model_to_ipfs(updatedClientIndex, modelEncoded)

    if blockchainEnabled == "1":
        upload_hash_to_blockchain(updatedClientIndex, newHash)

    url = "https://federated-learning-473375422539.us-central1.run.app"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "updatedClientIndex": updatedClientIndex,
        "blockchainEnabled": blockchainEnabled
    }

    try:
        print("sending request")
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            return {
                "statusCode": 200,
                "body": json.dumps(response_data)
            }
        else:
            return {
                "statusCode": response.status_code,
                "body": f"Error: {response.text}"
            }
    except Exception as e:
        # Handle exceptions (e.g., connection issues)
        return {
            "statusCode": 500,
            "body": f"An error occurred: {str(e)}"
        } 
