from typing import Optional, Any, List, Dict
import pydantic

class BaseProvider(pydantic.BaseModel):
    provider: str = "DefaultProvider"
    client: Optional[Any] = None
    api_key: Optional[str] = None
    model: str = "default-model"
    output_json_format: Optional[Any] = None
    last_response: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def create_client(self) -> Any:
        """
        Create and initialize the client for the provider.
        """
        raise NotImplementedError("Subclasses must implement `create_client`")

    async def get_response(
        self, 
        messages: str, 
        message_list: Optional[List[str]] = None, 
        system_message: Optional[str] = None
    ) -> Any:
        """
        Get a response from the provider based on input messages.

        Args:
            messages: The main input message as a string.
            message_list: A list of additional messages for context.
            system_message: An optional system message for context.

        Returns:
            The provider's response.
        """
        raise NotImplementedError("Subclasses must implement `get_response`")

    async def get_json_response(
        self, 
        messages: str, 
        message_list: Optional[List[str]] = None, 
        system_message: Optional[str] = None
    ) -> Any:
        """
        Get a JSON-formatted response from the provider.

        Args:
            messages: The main input message as a string.
            message_list: A list of additional messages for context.
            system_message: An optional system message for context.

        Returns:
            A JSON-formatted response.
        """
        raise NotImplementedError("Subclasses must implement `get_json_response`")

    def _prepare_message_list(
        self, 
        message: str, 
        message_list: Optional[List[str]] = None, 
        system_message: Optional[str] = None
    ) -> List[str]:
        """
        Prepare the list of messages to be sent to the provider.

        Args:
            message: The main input message.
            message_list: A list of additional messages for context.
            system_message: An optional system message for context.

        Returns:
            A complete list of messages.
        """
        raise NotImplementedError("Subclasses must implement `_prepare_message_list`")

    async def _fetch_response(
        self, 
        message_list: List[str], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Fetch the raw response from the provider.

        Args:
            message_list: A list of prepared messages.
            kwargs: Additional arguments for the request.

        Returns:
            The raw response.
        """
        raise NotImplementedError("Subclasses must implement `_fetch_response`")

    async def _fetch_json_response(
        self, 
        message_list: List[str], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Fetch the JSON-formatted response from the provider.

        Args:
            message_list: A list of prepared messages.
            kwargs: Additional arguments for the request.

        Returns:
            The JSON-formatted response.
        """
        raise NotImplementedError("Subclasses must implement `_fetch_json_response`")

    def _extract_content(self, response: Any) -> Any:
        """
        Extract content from the provider's response.

        Args:
            response: The provider's raw response.

        Returns:
            Extracted content.
        """
        raise NotImplementedError("Subclasses must implement `_extract_content`")