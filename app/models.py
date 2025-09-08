from typing import Optional

from pydantic import BaseModel, Field

try:  # Pydantic v2
    from pydantic import ConfigDict  # type: ignore
except Exception:  # Pydantic v1 fallback
    ConfigDict = dict  # type: ignore


class Property(BaseModel):
    """Structured representation of a property record from the dataset."""

    address: str = Field(alias="Address")
    suburb: str = Field(alias="Suburb")
    bedrooms: int = Field(alias="Rooms")
    property_type: str = Field(alias="Type")
    price: float = Field(alias="Price")
    bathrooms: Optional[int] = Field(default=None, alias="Bathroom")
    land_size: Optional[float] = Field(default=None, alias="Landsize")
    year_built: Optional[int] = Field(default=None, alias="YearBuilt")

    # Allow population by field name in addition to aliases (Pydantic v2)
    model_config = ConfigDict(populate_by_name=True)  # type: ignore


class SuburbTrends(BaseModel):
    """Analytical summary stats for a suburb."""

    suburb: str
    median_price: float
    property_count: int
    average_land_size: float


