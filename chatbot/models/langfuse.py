from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain.prompts import ChatPromptTemplate
import os


class LangfusePromptManager:
    """
    A manager class for handling prompts using Langfuse integration.
    
    This class provides functionality to create, store, and retrieve prompts using Langfuse,
    with support for both chat and regular prompts. It handles authentication and provides
    methods to interact with the Langfuse API.

    Attributes:
        langfuse (Langfuse): The main Langfuse client instance
        langfuse_callback_handler (CallbackHandler): Callback handler for Langfuse events

    Environment Variables Used (if parameters not provided):
        LANGFUSE_SECRET_KEY: Secret key for Langfuse authentication
        LANGFUSE_PUBLIC_KEY: Public key for Langfuse authentication
        LANGFUSE_HOST: Host URL for Langfuse service
    """
    
    def __init__(self, secret_key=None, public_key=None, host=None):
        """
        Initialize the LangfusePromptManager with authentication credentials.
        
        Args:
            secret_key (str, optional): Secret key for Langfuse authentication. 
                                       Defaults to LANGFUSE_SECRET_KEY environment variable.
            public_key (str, optional): Public key for Langfuse authentication.
                                       Defaults to LANGFUSE_PUBLIC_KEY environment variable.
            host (str, optional): Host URL for Langfuse service.
                                 Defaults to LANGFUSE_HOST environment variable.
        
        Raises:
            AssertionError: If authentication check fails for either the main client
                           or callback handler
        """
        # Use provided parameters or fall back to environment variables
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        host = host or os.getenv("LANGFUSE_HOST")
        
        self.langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        self.langfuse_callback_handler = CallbackHandler(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )

        assert self.langfuse.auth_check()
        assert self.langfuse_callback_handler.auth_check()

    def add(
        self,
        prompt: str,
        name: str,
        config: dict = {},
        labels: list[str] = ["production"],
    ):
        """
        Add a new prompt to Langfuse.

        Args:
            prompt (str): The prompt text or template to store
            name (str): Unique identifier for the prompt
            config (dict, optional): Configuration parameters for the prompt. Defaults to {}.
            labels (list[str], optional): List of labels to tag the prompt. Defaults to ["production"].

        Returns:
            None
        """
        self.langfuse.create_prompt(name=name, prompt=prompt, config=config, labels=labels)

    def get_prompt(self, prompt_name: str, version: int = None):
        """
        Retrieve a prompt from Langfuse and convert it to a LangChain prompt.

        Args:
            prompt_name (str): Name of the prompt to retrieve
            version (int, optional): Specific version of the prompt. Defaults to None.

        Returns:
            Any: LangChain-compatible prompt object
        """
        return self.langfuse.get_prompt(prompt_name, version=version).get_langchain_prompt()
    
    def get(self, prompt_name: str, is_chat: bool = True, version: int = None):
        """
        Get a prompt and its configuration from Langfuse.

        This method retrieves a prompt and converts it to either a chat prompt template
        or a regular prompt template based on the is_chat parameter.

        Args:
            prompt_name (str): Name of the prompt to retrieve
            is_chat (bool, optional): Whether to return a chat prompt template. Defaults to True.
            version (int, optional): Specific version of the prompt. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - ChatPromptTemplate: The prompt template (either chat or regular)
                - dict: The prompt configuration
        """
        langfuse_prompt = self.langfuse.get_prompt(prompt_name, version=version)
        if is_chat:
            prompt = ChatPromptTemplate.from_messages(
                langfuse_prompt.get_langchain_prompt(),
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                langfuse_prompt.get_langchain_prompt(),
                metadata={"langfuse_prompt": langfuse_prompt},
            )
        return prompt, langfuse_prompt.config
