from typing import List, Optional

import pandas as pd

from app.models import Property, SuburbTrends


class RealEstateDataProvider:
    """Provides access and simple analytics over a real estate CSV dataset."""

    REQUIRED_COLUMNS = [
        "Address",
        "Suburb",
        "Rooms",
        "Type",
        "Price",
        "Bathroom",
        "Landsize",
        "YearBuilt",
    ]

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.DataFrame()
        self._load_data()

    def _load_data(self) -> None:
        """Load and clean the CSV data into a DataFrame.

        - Handles file not found by keeping an empty DataFrame
        - Drops rows with missing prices
        - Fills missing Bathroom and Landsize values with 0
        """
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
            return

        # Ensure expected columns exist; add any missing with default NA
        for column_name in self.REQUIRED_COLUMNS:
            if column_name not in df.columns:
                df[column_name] = pd.NA

        # Basic cleaning
        df = df.dropna(subset=["Price"])  # must have a price
        df["Bathroom"] = df["Bathroom"].fillna(0)
        df["Landsize"] = df["Landsize"].fillna(0)

        # Normalize types where sensible
        with pd.option_context("mode.chained_assignment", None):
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
            df["Bathroom"] = pd.to_numeric(df["Bathroom"], errors="coerce").fillna(0).astype(int)
            df["Landsize"] = pd.to_numeric(df["Landsize"], errors="coerce").fillna(0.0)
            df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")
            df["YearBuilt"] = pd.to_numeric(df["YearBuilt"], errors="coerce")

        # Re-drop any rows that became NaN for essential numeric fields after coercion
        df = df.dropna(subset=["Price", "Rooms"])  # require price and rooms

        self.df = df

    def _row_to_property(self, row: pd.Series) -> Property:
        """Convert a DataFrame row into a Property model using alias keys."""
        data = {
            "Address": row.get("Address"),
            "Suburb": row.get("Suburb"),
            "Rooms": int(row.get("Rooms")) if pd.notna(row.get("Rooms")) else None,
            "Type": row.get("Type"),
            "Price": float(row.get("Price")) if pd.notna(row.get("Price")) else None,
            "Bathroom": int(row.get("Bathroom")) if pd.notna(row.get("Bathroom")) else 0,
            "Landsize": float(row.get("Landsize")) if pd.notna(row.get("Landsize")) else 0.0,
            "YearBuilt": int(row.get("YearBuilt")) if pd.notna(row.get("YearBuilt")) else None,
        }

        validator = getattr(Property, "model_validate", None)
        if callable(validator):
            return validator(data)
        # Pydantic v1
        return Property.parse_obj(data)

    def find_property_by_address(self, address: str) -> Optional[Property]:
        """Find a single property by exact address (case-insensitive)."""
        if self.df.empty or not address:
            return None

        address_norm = address.casefold()
        df = self.df.copy()
        mask = df["Address"].astype(str).str.casefold() == address_norm
        matches = df[mask]
        if matches.empty:
            return None
        # If multiple match, take the first
        first_row = matches.iloc[0]
        return self._row_to_property(first_row)

    def calculate_suburb_trends(self, suburb: str) -> Optional[SuburbTrends]:
        """Calculate median price, count, and average land size for a suburb (case-insensitive)."""
        if self.df.empty or not suburb:
            return None

        suburb_norm = suburb.casefold()
        df = self.df.copy()
        mask = df["Suburb"].astype(str).str.casefold() == suburb_norm
        sdf = df[mask]
        if sdf.empty:
            return None

        median_price = float(sdf["Price"].median()) if not sdf["Price"].empty else 0.0
        property_count = int(len(sdf))
        average_land_size = float(sdf["Landsize"].mean()) if not sdf["Landsize"].empty else 0.0

        # Use the canonical suburb name as seen in data for output consistency
        canonical_suburb = str(sdf.iloc[0]["Suburb"]) if pd.notna(sdf.iloc[0]["Suburb"]) else suburb

        return SuburbTrends(
            suburb=canonical_suburb,
            median_price=median_price,
            property_count=property_count,
            average_land_size=average_land_size,
        )

# ----------------------------
# LangChain tool integrations
# ----------------------------
from langchain.tools import tool
from pydantic import BaseModel as PydanticModel, Field

# Use the existing dataset in the repository (adjust if you relocate the file)
_DATASET_PATH = "/Users/muhammad/Documents/mySelf/projects/Australian Real Estate AI Agent/dataset/Melbourne_housing_FULL.csv"
_provider = RealEstateDataProvider(_DATASET_PATH)


class PropertySearchInput(PydanticModel):
    address: str = Field(..., description="Full street address to look up (case-insensitive)")


@tool(name="get_property_details", args_schema=PropertySearchInput)
def get_property_details(address: str):
    """Searches for and retrieves the details of a specific property by its address."""
    result = _provider.find_property_by_address(address)
    if result is None:
        return None
    # Return a JSON-serializable dict using original dataset column names via aliases
    dumper = getattr(result, "model_dump", None)
    if callable(dumper):
        return dumper(by_alias=True)
    return result.dict(by_alias=True)


class SuburbTrendsInput(PydanticModel):
    suburb: str = Field(..., description="Suburb name to calculate trends for (case-insensitive)")


@tool(name="get_suburb_trends", args_schema=SuburbTrendsInput)
def get_suburb_trends(suburb: str):
    """Calculates and returns the median price, property count, and average land size for a given suburb."""
    result = _provider.calculate_suburb_trends(suburb)
    if result is None:
        return None
    dumper = getattr(result, "model_dump", None)
    if callable(dumper):
        return dumper()
    return result.dict()


