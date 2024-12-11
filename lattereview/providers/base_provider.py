from typing import Optional, Any, List, Dict
from tokencost import calculate_prompt_cost, calculate_completion_cost
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
        """
        raise NotImplementedError("Subclasses must implement `_prepare_message_list`")

    async def _fetch_response(
        self, 
        message_list: List[str], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Fetch the raw response from the provider.
        """
        raise NotImplementedError("Subclasses must implement `_fetch_response`")

    async def _fetch_json_response(
        self, 
        message_list: List[str], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Fetch the JSON-formatted response from the provider.
        """
        raise NotImplementedError("Subclasses must implement `_fetch_json_response`")

    def _extract_content(self, response: Any) -> Any:
        """
        Extract content from the provider's response.
        """
        raise NotImplementedError("Subclasses must implement `_extract_content`")
    
    def _get_cost(self, input_messages: list, completion_text: str):
        """
        Calculate the cost of a prompt completion.
        """
        input_cost = calculate_prompt_cost(input_messages, self.model)
        output_cost = calculate_completion_cost(completion_text, self.model)
        return {"input_cost": float(input_cost), "output_cost": float(output_cost), "total_cost": float(input_cost + output_cost)}   