# Inference API and Pipeline


## Inference Pipeline

We will be using API gateway and Lambda function to interact with the model we deployed. Following is the architecture diagram of the process.

`<Placeholder for the architecture diagram>`


### Create a Lambda function

We are going to use AWS Lambda functions as our backend to interact with the sagemaker endpoint we deployed. 
1. Navigate to https://nasa-impact.awsapps.com/start 
2. Login using the credentials used to setup your account
3. Click on `AWS accounts`
4. Click on `summerSchool`
5. Click on `Management Console`
6. Search for `Lambda function` (Make sure we are in `us-west-2` region at the top right)
7. Click on `Create function` button
8. Select `Author from Scratch`
9. Provide a name for the function using the following template `<your name>-api-backend`
10. Change `Runtime` to `Python 3.8`
11. Click on `Change default execution role`
12. Select `Use an existing role`
13. Select `igarss-sagemaker-role` from the list
14. Click on `Create function`
15. Once on the lambda fucntion page, click on `Layers`, and click on `Add a layer`. We will be adding `scipy` packages for our usecase.
16. Choose `AWS layers`
17. Choose `AWSLambda-Python38-SciPy1x` and latest version
18. Click `Add`
19. On the code section of the lambda function page, paste the following code.


```
import os
import io
import boto3
import json
import csv
import numpy as np

# grab environment variables
ENDPOINT_NAME = "<Your endpoint name>"
runtime= boto3.client('runtime.sagemaker')

def download_and_load_data(payload):
    client = boto3.client('s3')
    client.download_file(
        payload['bucket'], 
        payload['file_name'], 
        '/tmp/test.npy'
    )
    print(os.path.exists('/tmp/test.npy'))
    return np.load('/tmp/test.npy')
    

def lambda_handler(event, context):
    data = json.loads(json.dumps(event))
    payload = json.loads(data['body'])
    print(payload)
    
    data = download_and_load_data(payload)
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=json.dumps(data.tolist()))
    os.remove('/tmp/test.npy')
    result = json.loads(response['Body'].read().decode())
    print(result)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(result)
    } 
    
```


20. Update the `ENDPOINT_NAME` to reflect your endpoint from before.
21. Deploy the changes.

### Add API endpoint

Once the lambda function is setup, we can tie it together with API gateway.
1. In the search bar, search for `API gateway`
2. Click `Create API`
3. In the first `REST API` option, click `Build`
4. Provide an API name, format: `<your-name>-inference-api`
5. Click `Create API`
6. In the API page, click on `Action` and select `Create Method`
7. Select `POST` from the list and click on the tick button beside it.
8. In the lambda function, type the lambda function name you created on the previous step
9. Click `Ok` on the dialog box.


### Infer
Because of limitations on AWS API endpoints, we are not allowed to upload files more than 6 MB. To circumvent this issue, we will leverage S3 bucket to host the test data for us. We will follow the following steps for inferencing purposes:
1. Upload test data into S3 bucket that the API has access to.
2. Call inference API with the file we uploaded to the S3 bucket.
3. Visualize the results.

#### 1. Upload tet data into S3 bucket that the API has access to

##### To upload the files into the S3 bucket, we can take two different approaches:
1. Directly upload the files into the S3 bucket using the AWS console
2. Use console to upload the files.


##### Direct upload to S3 Bucket:
1. Navigate to https://nasa-impact.awsapps.com/start 
2. Login using the credentials used to setup your account
3. Click on `AWS accounts`
4. Click on `summerSchool`
5. Click on `Management Console`
6. From the search bar in the AWS console, search for `S3` and click it.
7. Click on `smoke-dataset-bucket`
8. Create a folder with your name (if not already there)
9. Click on the folder you created.
10. Upload the file you want to test to the folder (drag drop or select the file using the UI)

##### Use signed URL:
1. Navigate to https://nasa-impact.awsapps.com/start 
2. Login using the credentials used to setup your account
3. Click on `AWS accounts`
4. Click on `summerSchool`
5. Click on `Command Line or Programmatic Access`
6. Copy the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`
7. Update the corresponding variables in the notebook cell.
8. Run the cell with `generate_signed_url` method definition
9. Set the variable `test_filename` with `<your name>/<your test filename>`
9. Call `generate_signed_url` method with `test_filename` as the parameter: Eg: `signed_url = generate_signed_url("iksha/test.tif")`
10. Upload corresponding file from your local machine using curl. `!curl --request PUT --upload-file <file path to upload> "{signed_url}"`


#### 2. Call inference API with the file we uploaded to the S3 bucket

Once the files are uploaded we infer using the API.
1. Run cell with `infer` method definition.
2. Call `infer` method with the `test_filename` and assign to a variable. Eg: `predictions = infer(test_filename)`

#### 3. Visualize the results

Visualize the results using `visualize` method. Pass the `test_filename` and `predictions` to the method to view results.