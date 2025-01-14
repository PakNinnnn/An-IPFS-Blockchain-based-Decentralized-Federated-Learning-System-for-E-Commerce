// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    struct Model {
        string ipfsHash;
        address client;
        uint256 timestamp;
    }

    Model[3] public models;  // Fixed array for 3 clients
    //mapping(address => uint256) public clientIndices;  // Map clients to their indices
    //address[] public clients;  // List of client addresses

    //event ModelUploaded(string ipfsHash, address client, uint256 timestamp, uint256 index);
    event ModelUploaded(string ipfsHash, address client, uint256 timestamp, uint256 index);
    
    /*
    constructor() {  
        clientIndices[0x79b12BcAF4F9Bf43dc7B2D370Da95C2668303A2E] = 0;
        clientIndices[0x64bf668Aa38d4892CC0041340ED4D7C3FA2238e7] = 1;
        clientIndices[0x5b162FB15dF23bf390eE49e4F223779c0e3587aC] = 2; 
    }
    */

    function uploadModel(string memory _ipfsHash, uint256 index) public {
        //uint256 index = clientIndices[msg.sender];
        //require(clients[index] == msg.sender, "Unauthorized client");
         require(index < models.length, "Model index out of bounds");
        
        models[index] = Model(_ipfsHash, msg.sender, block.timestamp);
        emit ModelUploaded(_ipfsHash, msg.sender, block.timestamp, index);
    }

    function getModel(uint256 index) public view returns (string memory, address, uint256) {
        require(index < models.length, "Model index out of bounds");
        Model memory model = models[index];
        return (model.ipfsHash, model.client, model.timestamp);
    }
}