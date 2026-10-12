try:
    import pydantic  # type: ignore
except ImportError as e:
    raise ImportError(
        "camel-converter must be installed with the pydantic extra to use this class"
    ) from e

from camel_converter import to_camel


class CamelBase(pydantic.BaseModel):
    """A Pydantic model that provides a base configuration for converting between camel and snake case.

    If another Pydantic model inherit from this class it will get the ability to do this conversion
    between camel and snake case without having to add the configuration to the new model.
    """

    model_config = pydantic.ConfigDict(alias_generator=to_camel, populate_by_name=True)  # type: ignore[attr-defined]
