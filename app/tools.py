from typing import List, Optional
import os

import pandas as pd
from langsmith import Client

# Support running as module (python -m app.tools) and as script (python app/tools.py)
try:
    from app.models import Property, SuburbTrends
except ModuleNotFoundError:  # When executed as a script, use local import
    from models import Property, SuburbTrends  # type: ignore


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

        # Normalize types where sensible
        with pd.option_context("mode.chained_assignment", None):
            # Clean price strings like "$1,200,000" -> "1200000"
            if "Price" in df.columns:
                price_str = df["Price"].astype(str).str.replace(r"[^0-9\.]", "", regex=True)
                price_str = price_str.replace({"": pd.NA})
                df["Price"] = pd.to_numeric(price_str, errors="coerce")
            df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")
            df["Bathroom"] = pd.to_numeric(df["Bathroom"], errors="coerce")
            df["Landsize"] = pd.to_numeric(df["Landsize"], errors="coerce")
            df["YearBuilt"] = pd.to_numeric(df["YearBuilt"], errors="coerce")

        # Drop rows with missing price only
        df = df.dropna(subset=["Price"])  # must have a price

        # Fill non-critical fields after coercion
        df["Bathroom"] = df["Bathroom"].fillna(0).astype(int)
        df["Landsize"] = df["Landsize"].fillna(0.0)

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

        address_norm = address.strip().casefold()
        df = self.df.copy()
        mask = df["Address"].astype(str).str.strip().str.casefold() == address_norm
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
from pathlib import Path

# Use the confirmed absolute dataset path
_DATASET_PATH = "/Users/muhammad/Documents/mySelf/projects/Australian Real Estate AI Agent/data/Melbourne_housing_FULL.csv"
_provider = RealEstateDataProvider(_DATASET_PATH)

# Initialize LangSmith client for tool tracing
_langsmith_client = None
if os.getenv("LANGCHAIN_API_KEY"):
    try:
        _langsmith_client = Client()
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize LangSmith client in tools: {e}")


class PropertySearchInput(PydanticModel):
    address: str = Field(..., description="Full street address to look up (case-insensitive)")


@tool(args_schema=PropertySearchInput)
def get_property_details(address: str):
    """Searches for and retrieves the details of a specific property by its address."""
    # Log tool execution to LangSmith if available
    if _langsmith_client:
        try:
            _langsmith_client.create_run(
                name="get_property_details",
                run_type="tool",
                inputs={"address": address},
                project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
            )
        except Exception as e:
            print(f"⚠️  LangSmith logging error in get_property_details: {e}")
    
    result = _provider.find_property_by_address(address)
    if result is None:
        # Log no result found
        if _langsmith_client:
            try:
                _langsmith_client.create_run(
                    name="get_property_details_no_result",
                    run_type="tool",
                    outputs={"result": None, "message": f"No property found for address: {address}"},
                    project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
                )
            except Exception as e:
                print(f"⚠️  LangSmith logging error in get_property_details: {e}")
        return None
    
    # Return a JSON-serializable dict using original dataset column names via aliases
    dumper = getattr(result, "model_dump", None)
    if callable(dumper):
        output = dumper(by_alias=True)
    else:
        output = result.dict(by_alias=True)
    
    # Log successful result to LangSmith if available
    if _langsmith_client:
        try:
            _langsmith_client.create_run(
                name="get_property_details_success",
                run_type="tool",
                outputs={"result": output, "message": f"Property found for address: {address}"},
                project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
            )
        except Exception as e:
            print(f"⚠️  LangSmith logging error in get_property_details: {e}")
    
    return output


class SuburbTrendsInput(PydanticModel):
    suburb: str = Field(..., description="Suburb name to calculate trends for (case-insensitive)")


@tool(args_schema=SuburbTrendsInput)
def get_suburb_trends(suburb: str):
    """Calculates and returns the median price, property count, and average land size for a given suburb."""
    # Log tool execution to LangSmith if available
    if _langsmith_client:
        try:
            _langsmith_client.create_run(
                name="get_suburb_trends",
                run_type="tool",
                inputs={"suburb": suburb},
                project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
            )
        except Exception as e:
            print(f"⚠️  LangSmith logging error in get_suburb_trends: {e}")
    
    result = _provider.calculate_suburb_trends(suburb)
    if result is None:
        # Log no result found
        if _langsmith_client:
            try:
                _langsmith_client.create_run(
                    name="get_suburb_trends_no_result",
                    run_type="tool",
                    outputs={"result": None, "message": f"No data found for suburb: {suburb}"},
                    project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
                )
            except Exception as e:
                print(f"⚠️  LangSmith logging error in get_suburb_trends: {e}")
        return None
    
    dumper = getattr(result, "model_dump", None)
    if callable(dumper):
        output = dumper()
    else:
        output = result.dict()
    
    # Log successful result to LangSmith if available
    if _langsmith_client:
        try:
            _langsmith_client.create_run(
                name="get_suburb_trends_success",
                run_type="tool",
                outputs={"result": output, "message": f"Trends calculated for suburb: {suburb}"},
                project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
            )
        except Exception as e:
            print(f"⚠️  LangSmith logging error in get_suburb_trends: {e}")
    
    return output


### testing

# Add this at the very bottom of app/tools.py

if __name__ == '__main__':
    # Use the confirmed absolute dataset path
    csv_file_path = "/Users/muhammad/Documents/mySelf/projects/Australian Real Estate AI Agent/data/Melbourne_housing_FULL.csv"

    print(f"Attempting to load data from: {csv_file_path}")

    try:
        # Test 1: Initialize the data provider
        provider = RealEstateDataProvider(csv_path=csv_file_path)
        print("✅ Data provider initialized successfully.")
        print("DataFrame columns:", provider.df.columns.tolist())
        print("DataFrame head:\n", provider.df.head())

        # Test 2: Test the property search function
        # Use an address you KNOW is in the CSV file
        test_address = "85 Turner St" # <-- IMPORTANT: Change to a real address from your CSV
        print(f"\nSearching for address: '{test_address}'...")
        property_result = provider.find_property_by_address(test_address)
        if property_result:
            print("✅ Found Property:")
            print(property_result.model_dump_json(indent=2))
        else:
            print("❌ FAILED to find property. Check your address and search logic.")

        # Test 3: Test the suburb trends function
        # Use a suburb you KNOW is in the CSV file
        test_suburb = "Abbotsford" # <-- IMPORTANT: Change to a real suburb from your CSV
        print(f"\nCalculating trends for suburb: '{test_suburb}'...")
        suburb_result = provider.calculate_suburb_trends(test_suburb)
        if suburb_result:
            print("✅ Calculated Trends:")
            print(suburb_result.model_dump_json(indent=2))
        else:
            print("❌ FAILED to calculate trends. Check your suburb name and calculation logic.")

    except FileNotFoundError:
        print("\n❌ CRITICAL ERROR: FileNotFoundError!")
        print("The CSV file was not found. Please check the 'csv_file_path' variable.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")