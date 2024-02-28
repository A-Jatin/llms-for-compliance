# llms-for-compliace

## Description
This service checks the webpage copy against the compliance policy and reports the findings.
This is currently powered by AZURE OPENAI API which is using GPT-4 model to generate the compliance report.

## Steps to run the service
1. Clone the repository

#### Docker run
2. Run the following command to build the docker image
```docker build -t llms-for-compliance .```
3. Run the following command to run the docker container
```docker run -p 80:80 llms-for-compliance```
4. The service will be running on http://localhost:80

#### Local run
2. Run the following command to install the dependencies
```pip install -r requirements.txt```
3. Run the following command to start the service
```python main.py```
4. The service will be running on http://localhost:80

### API Endpoints
1. POST /compliance-check
    - Request Body: 
    ```
    {
        "web_page_url": "The webpage url will be here",
        "compliance_policy_url": "The compliance policy will be here"
    }
    ```
    - Response:
    ```
    {
        "status": "The status of the api will be here",
        "compliant": "The compliance status will be here",
        "findings": "The findings(non-compliant results) will be here"
    }
    ```

### Architecture
We majorly have 3 modules in the architecture:
1. Configurations: This module is responsible for setting the configurations for the service. We can set the configurations for the service in the configurations.py file.
2. Extract: This module is responsible for extracting any webpage url given to it. It uses Goose to extract the webpage content. We can later on include any kind of media in this module using builder design used in llm module.
3. LLM: This module is responsible for checking the compliance of the webpage content against the compliance policy. This module also includes a builder to plug and use any kind of llm for the compliance check. Currently, we are using Azure OpenAI API to check the compliance.