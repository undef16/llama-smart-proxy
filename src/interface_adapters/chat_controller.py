from fastapi import HTTPException

from src.shared.error_utils import ErrorUtils
from src.use_cases.process_chat_completion import ProcessChatCompletion


class ChatController:
    def __init__(self, process_chat_completion_use_case: ProcessChatCompletion):
        self.process_chat_completion_use_case = process_chat_completion_use_case

    async def chat_completions(self, request: dict) -> dict:
        # Validate and sanitize request
        self._validate_chat_request(request)

        try:
            # Call use case
            response = await self.process_chat_completion_use_case.execute(request)
            return response
        except Exception as e:
            # Return a proper error response instead of letting FastAPI handle it
            return ErrorUtils.format_error_response(f"Internal server error: {str(e)}", "internal_error")


    def _validate_chat_request(self, request: dict) -> None:
        """Validate the chat completion request."""
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="Request must be a JSON object")

        model = request.get("model")
        if not model or not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code=400, detail="Model must be a non-empty string")
        request["model"] = model.strip()  # Sanitize

        messages = request.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="Messages must be a non-empty list")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise HTTPException(status_code=400, detail=f"Message {i} must be a JSON object")
            role = msg.get("role")
            if role not in ["user", "assistant", "system"]:
                raise HTTPException(status_code=400, detail=f"Message {i} role must be 'user', 'assistant', or 'system'")
            content = msg.get("content")
            if not isinstance(content, str):
                raise HTTPException(status_code=400, detail=f"Message {i} content must be a string")
            msg["content"] = content.strip()  # Sanitize

