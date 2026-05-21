import os
import pandas as pd
from langchain.tools import tool

@tool("csv_country_lookup")
def csv_country_lookup(country_name: str) -> str:
    """
    Looks up country information (Capital, GDP in USD) from the local database.
    Input should be the name of a country (e.g. 'Japan', 'Germany', or 'Egypt').
    """
    try:
        # Determine path to countries.csv relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.abspath(os.path.join(current_dir, "..", "data", "countries.csv"))
        
        if not os.path.exists(csv_path):
            return f"Error: CSV database not found at {csv_path}"
            
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Clean country input
        search_term = country_name.strip().lower()
        
        # Attempt exact match first
        match = df[df["Country"].str.strip().str.lower() == search_term]
        
        # If no exact match, try substring lookup
        if match.empty:
            match = df[df["Country"].str.strip().str.lower().str.contains(search_term, na=False)]
            
        if match.empty:
            countries_list = ", ".join(df["Country"].tolist())
            return (
                f"Country '{country_name}' was not found in the CSV database. "
                f"Available countries are: {countries_list}."
            )
            
        # Format the result nicely
        row = match.iloc[0]
        country = row["Country"]
        capital = row["Capital"]
        gdp = row["GDP"]
        
        return f"Country: {country} | Capital: {capital} | GDP: ${gdp:,.2f} USD"
        
    except Exception as e:
        return f"Error occurred during CSV lookup: {str(e)}"
