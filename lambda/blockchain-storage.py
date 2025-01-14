import json
from web3 import Web3
from web3.gas_strategies.rpc import rpc_gas_price_strategy 

# Add the Web3 Provider
RPC_PROVIDER_APIKEY  = "438da484051744a0bfc3cf924110bdc1"
RPC_PROVIDER_URL = "https://sepolia.infura.io/v3/438da484051744a0bfc3cf924110bdc1"

#To be replaced by client input
ACCOUNT = "0x79b12BcAF4F9Bf43dc7B2D370Da95C2668303A2E"
ACCOUNT_PRIVATE_KEY = "172e6450b8be0e8d15c235eefc9fda04ac803816ffaaae1cf05d59092a3f6e8e" 
ACCOUNT_LIST = [
                    "0x79b12BcAF4F9Bf43dc7B2D370Da95C2668303A2E", 
                    "0x64bf668Aa38d4892CC0041340ED4D7C3FA2238e7", 
                    "0x5b162FB15dF23bf390eE49e4F223779c0e3587aC"
                ]
ACCOUNT_PKEY_LIST = [
                        "172e6450b8be0e8d15c235eefc9fda04ac803816ffaaae1cf05d59092a3f6e8e", 
                        "c6c5abb90237ef79fb7a5d88b6334a0a09b050608df78cb037b5714ccae7fc65", 
                        "c49687851e2a9c6254b1095ed375fc795843b939817a19a43699275996341553"
                    ]

web3Handler = Web3(Web3.HTTPProvider(RPC_PROVIDER_URL))

account_from = {
    'private_key': ACCOUNT_PRIVATE_KEY,
    'address': web3Handler.eth.account.from_key(ACCOUNT_PRIVATE_KEY).address
}

#print("Attempting to deploy from account: ", account_from['address'])
#print("Balance: ", web3Handler.eth.get_balance(account_from['address']))

#Declare contract
contract_address = '0x980B6A9D39AdbDA1435b8498C429A19e04466237'
contract_abi = json.loads('[{"anonymous":false,"inputs":[{"indexed":false,"internalType":"string","name":"ipfsHash","type":"string"},{"indexed":false,"internalType":"address","name":"client","type":"address"},{"indexed":false,"internalType":"uint256","name":"timestamp","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"index","type":"uint256"}],"name":"ModelUploaded","type":"event"},{"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"getModel","outputs":[{"internalType":"string","name":"","type":"string"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"models","outputs":[{"internalType":"string","name":"ipfsHash","type":"string"},{"internalType":"address","name":"client","type":"address"},{"internalType":"uint256","name":"timestamp","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"string","name":"_ipfsHash","type":"string"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"uploadModel","outputs":[],"stateMutability":"nonpayable","type":"function"}]')
contract = web3Handler.eth.contract(address=contract_address, abi=contract_abi) 

#Set rule for gas price
web3Handler.eth.set_gas_price_strategy(rpc_gas_price_strategy)

def get_hash_from_blockchain(index): 
    call_result = contract.functions.getModel(index).call()
    hash = call_result[0] 

    return hash

def store_hash_on_blockchain(ipfs_hash, index): 
    gas_price = web3Handler.eth.generate_gas_price()
    print("Gas Price: ", gas_price)

    # Build transaction  
    transaction = contract.functions.uploadModel(ipfs_hash, index).build_transaction({
        'from': account_from['address'], 
        'nonce': web3Handler.eth.get_transaction_count(account_from['address']),
        'gasPrice': gas_price
    })

    # Sign transaction
    signed_txn = web3Handler.eth.account.sign_transaction(transaction, account_from['private_key']) 
    tx_hash = web3Handler.eth.send_raw_transaction(signed_txn.raw_transaction)   
    receipt = web3Handler.eth.wait_for_transaction_receipt(tx_hash)

    return receipt.transactionHash.hex(), receipt.contractAddress

def lambda_handler(event, context):
    #action = event.get('action')
    #clientIndex= event.get('clientIndex')
    #content = event.get('content')
    body = event.get('body', '{}') 
    parsed_body = json.loads(body)

    action = parsed_body.get('action')
    clientIndex = parsed_body.get('clientIndex')
    content = parsed_body.get('content')
    
    #0: write; 1: read

    if action is None:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': "Invalid input. 'action' is required."
            })
        }

    if clientIndex is None:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': "Invalid input. 'clientIndex' is required."
            })
        }

    clientIndex = int(clientIndex)

    if action == "0": #Write
        if content is None:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': "Invalid input. 'content' is required."
                })
            }

        ACCOUNT = ACCOUNT_LIST[clientIndex]
        ACCOUNT_PRIVATE_KEY = ACCOUNT_PKEY_LIST[clientIndex]

        #print(ACCOUNT, ACCOUNT_PRIVATE_KEY)

        #Upload to smart contract
        store_hash_on_blockchain(content, clientIndex)

        # TODO implement
        return {
            'statusCode': 200,
            'body': json.dumps({
                        'status': 'true',
                    }),
        }
    elif action == "1": #Read 
        hash = get_hash_from_blockchain(clientIndex)
        if hash is not None:
            return {
                'statusCode': 200,
                'body': json.dumps({
                            'status': 'true',
                            'hash': hash,
                        }),
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                            'error': "Failed retrieving content."
                        }),
            }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({
                        'error': "Invalid action."
                    }),
            }