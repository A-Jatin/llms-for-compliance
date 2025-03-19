from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from configurations import (
    AZURE_OPENAI_DEPLOYMENT_NAME,
    OPENAI_MODEL_NAME,
)


class LLMBuilder:

    @staticmethod
    async def build_llm(llm_type: str):
        """This function takes in the type of LLM and returns the corresponding LLM class."""
        if llm_type == 'azure_openai':
            return AzureChatOpenAI
        elif llm_type == 'openai':
            return ChatOpenAI
        else:
            raise ValueError("Invalid LLM type. Supported types: 'azure_openai', 'openai'")

    @staticmethod
    async def build_llm_params(llm_type: str):
        """This function takes in the type of LLM and returns the corresponding LLM parameters."""
        if llm_type == 'azure_openai':
            return {
                "deployment_name": AZURE_OPENAI_DEPLOYMENT_NAME,
                "temperature": 0.1,
            }
        elif llm_type == 'openai':
            return {
                "model_name": OPENAI_MODEL_NAME,
                "temperature": 0.1,
            }
        else:
            raise ValueError("Invalid LLM type. Supported types: 'azure_openai', 'openai'")
