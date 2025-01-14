import { create } from 'ipfs-http-client';
import fs from 'fs';

const client = create({ url: "http://35.168.133.141:5001/api/v0" });

export const handler = async (event) => {
   
  try {  
    // Check if the body exists
    if (!event.body) {
      throw new Error("Request body is missing.");
    }
    
    let body;
    try {
      body = JSON.parse(event.body);
    } catch (parseError) {
      throw new Error("Invalid JSON in request body.");
    }

    let action = body.action;

    if(action == "0"){
      //return;
 


      // Validate required fields
      if (!body.fileName || !body.fileContent) {
        return {
          statusCode: 400,
          body: JSON.stringify({
            error: "Request must include 'fileName' and 'fileContent'.",
          }),
        };
      }

      const { fileName, fileContent } = body;

      // Decode the Base64 content
      const fileBuffer = Buffer.from(fileContent, 'base64');

      // Upload the file to IPFS
      const result = await client.add({
        path: fileName,
        content: fileBuffer,
      });

      console.log('IPFS Upload Result:', result);

      // Return the IPFS CID
      return {
        statusCode: 200,
        body: JSON.stringify({
          status: 'true',
          cid: result.cid.toString(),
        }),
      };
    }
    else if(action == "1"){  
      // Validate required fields
      if (!body.cid) {
        return {
          statusCode: 400,
          body: JSON.stringify({
            error: "Request must include 'cid'.",
          }),
        };
      }

      const { cid } = body;

      // Retrieve the file from IPFS
      try {
        const chunks = [];
        for await (const chunk of client.cat(cid)) {
          chunks.push(chunk);
        }

        const fileContent = Buffer.concat(chunks).toString('base64');

        console.log(`File retrieved from IPFS with CID: ${cid}`);

        // Return the retrieved file content
        return {
          statusCode: 200,
          body: JSON.stringify({
            status: 'true',
            cid: cid,
            fileContent: fileContent,
          }),
        };
      } catch (err) {
        console.error('Error retrieving file from IPFS:', err.message);
        return {
          statusCode: 500,
          body: JSON.stringify({
            error: `Failed to retrieve file from IPFS with CID: ${cid}`,
          }),
        };
      }
    }
    else{
      return {
        statusCode: 500,
        body: JSON.stringify({
          error: `Invalid action`,
        }),
      };
    }
    
    
  } catch (error) {
    console.error('Error uploading file to IPFS:', error.message);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Failed to upload file to IPFS.' }),
    };
  }
};
