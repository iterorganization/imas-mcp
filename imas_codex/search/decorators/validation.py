"""
Input validation decorator for tools.

Provides Pydantic-based input validation with standardized error responses.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

F = TypeVar("F", bound=Callable[..., Any])


def validate_input(schema: type[BaseModel]) -> Callable[[F], F]:
    """
    Decorator to validate input parameters using Pydantic schema.

    Args:
        schema: Pydantic model class for validation

    Returns:
        Decorated function with input validation
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract 'self' instance and context
            instance = args[0] if args else None
            ctx = kwargs.pop("ctx", None)

            # Map positional arguments to schema fields
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            # Skip 'self' parameter
            if param_names and param_names[0] == "self":
                param_names = param_names[1:]

            # Map positional args (excluding self) to parameter names
            request_data = {}
            for i, arg in enumerate(args[1:]):  # Skip self
                if i < len(param_names) and param_names[i] != "ctx":
                    request_data[param_names[i]] = arg

            # Add kwargs (excluding ctx)
            request_data.update({k: v for k, v in kwargs.items() if k != "ctx"})

            try:
                # Validate input using schema
                validated = schema(**request_data)

                # Replace kwargs with validated data
                validated_kwargs = validated.model_dump()
                if ctx is not None:
                    validated_kwargs["ctx"] = ctx

                # Call function with validated parameters
                return await func(instance, **validated_kwargs)

            except ValidationError as e:
                # Create standardized error response
                error_details = []
                for error in e.errors():
                    field = ".".join(str(loc) for loc in error["loc"])
                    message = error["msg"]
                    error_details.append(f"{field}: {message}")

                error_message = f"Validation error: {'; '.join(error_details)}"

                # Return error response using instance method if available
                if instance and hasattr(instance, "_create_error_response"):
                    query = request_data.get("query", "")
                    return instance._create_error_response(error_message, query)

                # Fallback error response
                return {
                    "error": error_message,
                    "query": request_data.get("query", ""),
                    "validation_errors": error_details,
                }

        return wrapper

    return decorator


def create_validation_schema(
    required_fields: dict[str, type],
    optional_fields: dict[str, Any] | None = None,
    validators: dict[str, Callable] | None = None,
) -> type[BaseModel]:
    """
    Helper to create validation schemas dynamically.

    Args:
        required_fields: Dictionary of required field names and types
        optional_fields: Dictionary of optional field names and default values
        validators: Dictionary of field validators

    Returns:
        Pydantic model class
    """
    if optional_fields is None:
        optional_fields = {}
    if validators is None:
        validators = {}

    # Build field definitions
    annotations = {}
    defaults = {}

    # Add required fields
    for name, field_type in required_fields.items():
        annotations[name] = field_type

    # Add optional fields
    for name, default_value in optional_fields.items():
        field_type = type(default_value) if default_value is not None else Any
        annotations[name] = field_type
        defaults[name] = default_value

    # Create model class
    class DynamicSchema(BaseModel):
        pass

    # Set annotations
    DynamicSchema.__annotations__ = annotations

    # Set defaults
    for name, default_value in defaults.items():
        setattr(DynamicSchema, name, default_value)

    # Add validators
    for field_name, validator_func in validators.items():
        setattr(DynamicSchema, f"validate_{field_name}", validator_func)

    return DynamicSchema
