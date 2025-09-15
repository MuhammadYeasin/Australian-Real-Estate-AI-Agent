from typing import List, Optional
import os
from urllib.parse import urlencode

import pandas as pd
import requests
from langsmith import Client

# Support running as module (python -m app.tools) and as script (python app/tools.py)
try:
    from app.models import Property, SuburbTrends
except ModuleNotFoundError:  # When executed as a script, use local import
    from models import Property, SuburbTrends  # type: ignore


class RealEstateDataProvider:
    """Provides access and simple analytics over real estate data via Domain.com.au API.

    Uses HTTP API to query real estate listings on demand.
    """


    def __init__(self, api_base_url: str) -> None:
        self.api_base_url = api_base_url.strip("/")
        self._http: requests.Session = requests.Session()
        self._api_headers: dict = {}
        self._agency_id: str = os.getenv("REAL_ESTATE_AGENCY_ID") or "22473"
        
        # Configure auth header (Domain API commonly uses Bearer token)
        token = os.getenv("REAL_ESTATE_API_TOKEN") or "04fd474f22acfbad451b634bea9ad0d9"
        if token:
            self._api_headers["Authorization"] = f"Bearer {token}"
        # Add Domain.com.au specific headers
        self._api_headers["accept"] = "application/json"
        self._api_headers["X-Api-Call-Source"] = "live-api-browser"


    # -------- API helpers --------
    def _parse_price_from_text(self, text: Optional[str]) -> float:
        if not text:
            return 0.0
        try:
            import re
            numbers = re.findall(r"\d+[\,\.]?\d*", text)
            if not numbers:
                return 0.0
            numeric = numbers[0].replace(",", "")
            return float(numeric)
        except Exception:
            return 0.0

    def _map_external_property(self, item: dict) -> pd.Series:
        """Map an external API listing (Domain-style) to our canonical columns."""
        address_parts = item.get("addressParts", {}) or {}
        display_address = address_parts.get("displayAddress")
        if not display_address:
            unit = address_parts.get("unitNumber")
            street_no = address_parts.get("streetNumber")
            street = address_parts.get("street")
            suburb = address_parts.get("suburb")
            state = address_parts.get("stateAbbreviation")
            postcode = address_parts.get("postcode")
            parts: List[str] = []
            if unit and street_no:
                parts.append(f"{unit}/{street_no}")
            elif street_no:
                parts.append(str(street_no))
            if street:
                parts.append(str(street))
            local_parts: List[str] = []
            if suburb:
                local_parts.append(str(suburb))
            if state:
                local_parts.append(str(state).upper())
            if postcode:
                local_parts.append(str(postcode))
            display_address = ", ".join([" ".join(parts), " ".join(local_parts)]).strip(", ") if parts or local_parts else None

        property_types = item.get("propertyTypes") or []
        property_type = property_types[0] if property_types else None

        price_details = item.get("priceDetails") or {}
        display_price = price_details.get("displayPrice")
        price_value = self._parse_price_from_text(display_price)

        mapped = {
            "Address": display_address,
            "Suburb": address_parts.get("suburb"),
            "Rooms": int(item.get("bedrooms") or 0) if item.get("bedrooms") is not None else None,
            "Type": property_type or "",
            "Price": price_value,
            "Bathroom": int(item.get("bathrooms") or 0) if item.get("bathrooms") is not None else 0,
            "Landsize": float(item.get("landSize") or 0.0) if item.get("landSize") is not None else 0.0,
            "YearBuilt": int(item.get("yearBuilt")) if item.get("yearBuilt") is not None else None,
        }
        return pd.Series(mapped)


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
        if not address:
            return None

        try:
            # Fetch listings from agency endpoint and filter client-side by full address
            # Pagination loop with sane cap
            page = 1
            page_size = 50
            address_norm = address.strip().casefold()
            while page <= 10:
                query = urlencode({
                    "listingStatusFilter": "live",
                    "pageNumber": page,
                    "pageSize": page_size,
                })
                url = f"{self.api_base_url}/v1/agencies/{self._agency_id}/listings?{query}"
                resp = self._http.get(url, headers=self._api_headers, timeout=20)
                resp.raise_for_status()
                data = resp.json() or []
                if isinstance(data, dict):
                    items = data.get("results") or data.get("data") or []
                else:
                    items = data
                if not items:
                    break
                for item in items:
                    row = self._map_external_property(item)
                    row_addr = str(row.get("Address") or "").strip().casefold()
                    if row_addr and row_addr == address_norm:
                        return self._row_to_property(row)
                if len(items) < page_size:
                    break
                page += 1
            return None
        except Exception:
            return None

    def calculate_suburb_trends(self, suburb: str) -> Optional[SuburbTrends]:
        """Calculate median price, count, and average land size for a suburb (case-insensitive)."""
        if not suburb:
            return None

        try:
            # Aggregate listings from the agency endpoint and compute trends client-side
            page = 1
            page_size = 100
            rows: List[pd.Series] = []
            suburb_norm = suburb.strip().casefold()
            while page <= 10:
                query = urlencode({
                    "listingStatusFilter": "live",
                    "pageNumber": page,
                    "pageSize": page_size,
                })
                url = f"{self.api_base_url}/v1/agencies/{self._agency_id}/listings?{query}"
                resp = self._http.get(url, headers=self._api_headers, timeout=25)
                resp.raise_for_status()
                data = resp.json() or []
                if isinstance(data, dict):
                    items = data.get("results") or data.get("data") or []
                else:
                    items = data
                if not items:
                    break
                for item in items:
                    s = self._map_external_property(item)
                    s_suburb = str(s.get("Suburb") or "").casefold()
                    if s_suburb == suburb_norm:
                        rows.append(s)
                if len(items) < page_size:
                    break
                page += 1
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df = df.dropna(subset=["Price"]) if "Price" in df.columns else df
            if df.empty:
                return None
            median_price = float(df["Price"].median()) if "Price" in df.columns else 0.0
            property_count = int(len(df))
            average_land_size = float(df["Landsize"].mean()) if "Landsize" in df.columns else 0.0
            canonical_suburb = str(df.iloc[0]["Suburb"]) if "Suburb" in df.columns else suburb
            return SuburbTrends(
                suburb=canonical_suburb or suburb,
                median_price=median_price,
                property_count=property_count,
                average_land_size=average_land_size,
            )
        except Exception:
            return None

# ----------------------------
# LangChain tool integrations
# ----------------------------
from langchain.tools import tool
from pydantic import BaseModel as PydanticModel, Field
from pathlib import Path

# Configure data source: always use Domain.com.au API
_API_BASE = os.getenv("REAL_ESTATE_API_BASE") or "https://api.domain.com.au/sandbox"
_provider = RealEstateDataProvider(api_base_url=_API_BASE)

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
                "get_property_details",
                {"address": address},
                "tool",
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
                    "get_property_details_no_result",
                    {"result": None, "message": f"No property found for address: {address}"},
                    "tool",
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
                "get_property_details_success",
                {"result": output, "message": f"Property found for address: {address}"},
                "tool",
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
                "get_suburb_trends",
                {"suburb": suburb},
                "tool",
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
                    "get_suburb_trends_no_result",
                    {"result": None, "message": f"No data found for suburb: {suburb}"},
                    "tool",
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
                "get_suburb_trends_success",
                {"result": output, "message": f"Trends calculated for suburb: {suburb}"},
                "tool",
                project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
            )
        except Exception as e:
            print(f"⚠️  LangSmith logging error in get_suburb_trends: {e}")
    
    return output


### testing

# Add this at the very bottom of app/tools.py

if __name__ == '__main__':
    # Test API mode
    api_base = os.getenv("REAL_ESTATE_API_BASE") or "https://api.domain.com.au/sandbox"
    
    print(f"Using Domain.com.au API. Base URL: {api_base}")
    print(f"Agency ID: {os.getenv('REAL_ESTATE_AGENCY_ID', '22473')}")
    print(f"API Token: {os.getenv('REAL_ESTATE_API_TOKEN', '04fd474f22acfbad451b634bea9ad0d9')[:10]}...")

    try:
        # Test 1: Initialize the data provider
        provider = RealEstateDataProvider(api_base_url=api_base)
        print("✅ Data provider initialized successfully.")

        # Test 2: Test the property search function
        # Use an address from the API data
        test_address = "209/7 Sterling Circuit, Camperdown NSW 2050" # From API data
        print(f"\nSearching for address: '{test_address}'...")
        property_result = provider.find_property_by_address(test_address)
        if property_result:
            print("✅ Found Property:")
            print(property_result.model_dump_json(indent=2))
        else:
            print("❌ FAILED to find property. Check your address and search logic.")

        # Test 3: Test the suburb trends function
        # Use a suburb from the API data
        test_suburb = "Camperdown" # From API data
        print(f"\nCalculating trends for suburb: '{test_suburb}'...")
        suburb_result = provider.calculate_suburb_trends(test_suburb)
        if suburb_result:
            print("✅ Calculated Trends:")
            print(suburb_result.model_dump_json(indent=2))
        else:
            print("❌ FAILED to calculate trends. Check your suburb name and calculation logic.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")