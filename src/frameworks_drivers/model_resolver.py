class ModelResolver:
    """
    Resolves model identifiers from client requests into repository ID and filename pattern for GGUF files.
    """

    DEFAULT_VARIANT = "Q4_K_M"
    _cache: dict[str, tuple[str, str]] = {}

    @staticmethod
    def resolve(model_identifier: str) -> tuple[str, str]:
        """
        Parse the model identifier and resolve to repository ID and filename pattern.

        Supported formats:
        - 'repo_id:variant' (e.g., 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M')
        - 'repo_id' (uses default variant)

        Args:
            model_identifier: The model string from the request.

        Returns:
            A tuple of (repository_id, filename_pattern)

        Raises:
            ValueError: If the model identifier is invalid.
        """
        if model_identifier in ModelResolver._cache:
            return ModelResolver._cache[model_identifier]

        if not model_identifier or not model_identifier.strip():
            raise ValueError("Model identifier cannot be empty")

        model_identifier = model_identifier.strip()

        if ":" in model_identifier:
            parts = model_identifier.split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid model identifier format: {model_identifier}")
            repo_id, variant = parts
            if not repo_id or not variant or ":" in variant:
                raise ValueError(f"Invalid model identifier format: {model_identifier}")
        else:
            repo_id = model_identifier
            variant = ModelResolver.DEFAULT_VARIANT

        # Create filename pattern for GGUF files
        if variant.endswith(".gguf"):
            filename_pattern = variant
        else:
            filename_pattern = f"*{variant}.gguf"

        result = repo_id, filename_pattern
        ModelResolver._cache[model_identifier] = result
        return result