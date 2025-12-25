class ErrorUtils:
    @staticmethod
    def format_error_response(message: str, error_type: str) -> dict:
        """
        Formats a consistent error response dictionary.

        Args:
            message: The error message to include in the response.
            error_type: The type of error (e.g., "internal_error", "health_check_error").

        Returns:
            A dictionary with the error details.
        """
        return {
            "error": {
                "message": message,
                "type": error_type
            }
        }