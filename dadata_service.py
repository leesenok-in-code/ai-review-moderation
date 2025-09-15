import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

def get_company_data_by_inn(inn):
    try:
        url = "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party"
        headers = {
            "Authorization": f"Token {os.getenv('DADATA_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {"query": inn}
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            logging.error(f"Non-JSON response: {response.text[:200]}")
            return None
        
        result = response.json()
        
        if not result.get('suggestions'):
            return None
            
        first_suggestion = result['suggestions'][0]
        
        name = first_suggestion.get('value', '')
        data = first_suggestion.get('data', {})
        address = data.get('address', {})
        address_value = address.get('value', '') if isinstance(address, dict) else ''
        address_city = ''
        
        if isinstance(address, dict):
            address_data = address.get('data', {})
            if isinstance(address_data, dict):
                address_city = address_data.get('city', '')
        
        return {
            'name': name,
            'ogrn': data.get('ogrn', ''),
            'address': address_value,
            'city': address_city
        }
        
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error: {str(req_err)}")
    except Exception as e:
        logging.error(f"Data processing error: {str(e)}")
    
    return None