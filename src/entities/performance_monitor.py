class PerformanceMonitor:
    """Monitor performance of GPU allocation decisions."""

    def __init__(self):
        self.allocation_times = []
        self.allocation_success_count = 0
        self.allocation_failure_count = 0

    def record_allocation_time(self, time_ms: float, success: bool):
        """Record the time taken for an allocation decision."""
        self.allocation_times.append(time_ms)
        if success:
            self.allocation_success_count += 1
        else:
            self.allocation_failure_count += 1

    def get_average_allocation_time(self) -> float:
        """Get the average allocation time in milliseconds."""
        if not self.allocation_times:
            return 0.0
        return sum(self.allocation_times) / len(self.allocation_times)

    def get_recent_allocation_time(self, n: int = 10) -> float:
        """Get the average allocation time for the last n allocations."""
        recent_times = self.allocation_times[-n:] if len(self.allocation_times) >= n else self.allocation_times
        if not recent_times:
            return 0.0
        return sum(recent_times) / len(recent_times)