import re
import requests
from langchain.tools import tool

@tool("currency_converter")
def currency_converter(query: str) -> str:
    """
    Converts currency amounts between different currencies using the latest USD rates from exchange rate API.
    Input query MUST match the format 'amount FROM_CURRENCY to TO_CURRENCY' (e.g. '100 USD to EUR' or '4500000000000 USD to EUR').
    """
    try:
        # Pre-process query: strip white spaces, remove currency symbols like $, €, £
        query = re.sub(r'[\$\€\£\¥]', '', query).strip()
        
        # Regex to parse amount (allowing decimals, commas, scientific notation), source currency, and target currency
        pattern = r"([\d\.,eE\+\-]+)\s*([A-Za-z]{3})\s+to\s+([A-Za-z]{3})"
        match = re.search(pattern, query, re.IGNORECASE)
        
        if not match:
            return (
                "Error: Input query does not match the expected pattern. "
                "Please use the format: '<amount> <from_currency> to <to_currency>' (e.g. '100 USD to EUR')."
            )
            
        amount_str, from_curr, to_curr = match.groups()
        
        # Parse amount
        amount_str = amount_str.replace(",", "")
        try:
            amount = float(amount_str)
        except ValueError:
            return f"Error: Could not parse '{amount_str}' as a number."
            
        from_curr = from_curr.upper()
        to_curr = to_curr.upper()
        
        # Call API
        api_url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code != 200:
            return f"Error: Failed to fetch exchange rates from API. HTTP Status: {response.status_code}"
            
        data = response.json()
        rates = data.get("rates", {})
        
        # Verify both currencies are supported
        if from_curr not in rates:
            return f"Error: Unsupported currency code '{from_curr}'."
        if to_curr not in rates:
            return f"Error: Unsupported currency code '{to_curr}'."
            
        # Convert currencies (using USD rates as reference)
        usd_rate = rates[from_curr]
        target_rate = rates[to_curr]
        
        # Convert to USD first, then to target
        amount_in_usd = amount / usd_rate
        converted_amount = amount_in_usd * target_rate
        
        return f"{amount:,.2f} {from_curr} is equal to {converted_amount:,.2f} {to_curr} (1 {from_curr} = {target_rate/usd_rate:,.4f} {to_curr})"
        
    except Exception as e:
        return f"Error occurred during currency conversion: {str(e)}"
