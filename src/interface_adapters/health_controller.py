from src.shared.error_utils import ErrorUtils
from src.use_cases.get_health import GetHealth


class HealthController:
    def __init__(self, get_health_use_case: GetHealth):
        self.get_health_use_case = get_health_use_case

    def health(self):
        try:
            return self.get_health_use_case.execute()
        except Exception as e:
            return ErrorUtils.format_error_response(f"Health check failed: {str(e)}", "health_check_error")
