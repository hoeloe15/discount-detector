import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional
import logging
from urllib.parse import urlparse
import random
from time import sleep
import json
import re
from config import Config
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Force reload environment variables
load_dotenv(override=True)

class SmartScraper:
    def __init__(self, url: str):
        """
        Initialize the scraper with a URL and necessary configurations
        Args:
            url (str): The URL of the product page to scrape
        """
        self.url = url
        self.domain = urlparse(url).netloc
        
        # Debug print the configuration
        print("\nAzure OpenAI Configuration:")
        print(f"Endpoint: {Config.AZURE_OPENAI_ENDPOINT}")
        print(f"Deployment: {Config.AZURE_OPENAI_DEPLOYMENT_NAME}")
        print(f"API Version: {Config.AZURE_OPENAI_API_VERSION}")
        
        # Initialize Azure OpenAI client with correct endpoint formatting
        self.llm_client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_KEY or os.environ.get('AZURE_OPENAI_API_KEY'),
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT.split('/openai/')[0]  # Get base URL only
        )
        
        self.deployment_name = Config.AZURE_OPENAI_DEPLOYMENT_NAME
        
        # List of user agents to rotate through for avoiding detection
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        ]

    async def _safe_request(self) -> tuple[Optional[str], int]:
        """Make a safe request with error handling and return HTML content and status code"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._get_headers()
                async with session.get(self.url, headers=headers) as response:
                    content = await response.text()
                    return content, response.status
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
            return None, 500

    def _get_headers(self) -> dict:
        """
        Generate random headers for each request to mimic browser behavior
        Returns:
            dict: Headers dictionary with randomized user agent
        """
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }

    async def _check_for_blocking(self, response_text: str, status_code: int) -> bool:
        """
        Check if the website is blocking our scraping attempts using both
        traditional indicators and LLM analysis
        
        Args:
            response_text (str): HTML response from the website
            status_code (int): HTTP status code
        Returns:
            bool: True if blocking is detected, False otherwise
        """
        # Check common blocking indicators
        blocking_indicators = [
            status_code in [403, 429, 503],
            "captcha" in response_text.lower(),
            "access denied" in response_text.lower(),
            "rate limit" in response_text.lower(),
        ]

        if any(blocking_indicators):
            logging.error(f"⛔ BLOCKING DETECTED! Status code: {status_code}")
            return True

        # Use LLM to detect more sophisticated blocking methods
        try:
            prompt = """Analyze this HTML response and determine if it shows signs of blocking or anti-bot measures.
            Consider things like:
            - Presence of CAPTCHAs
            - Access denied messages
            - Robot detection messages
            - Unusual redirects
            
            Response format: JSON with keys 'is_blocked' (boolean) and 'reason' (string)"""

            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.deployment_name,
                messages=[{"role": "user", "content": f"{prompt}\n\nFirst 1000 chars of HTML: {response_text[:1000]}"}],
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            if result.get('is_blocked'):
                logging.error(f"🤖 LLM detected blocking: {result.get('reason')}")
                return True
                
        except Exception as e:
            logging.error(f"Error in LLM blocking check: {str(e)}")
        
        return False

    async def _extract_data_with_llm(self, html_content: str) -> Optional[dict]:
        """
        Use Azure OpenAI to extract product information from HTML content
        
        Args:
            html_content (str): Raw HTML content from the webpage
        Returns:
            Optional[dict]: Extracted product data or None if extraction failed
        """
        try:
            # Clean HTML by removing scripts and styles to reduce token usage
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Clean HTML by removing scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Focus on elements likely to contain prices
            price_elements = soup.find_all(class_=lambda x: x and any(term in x.lower() for term in ['price', 'prijs', 'amount', 'bedrag']))
            product_section = str(soup.find(class_=lambda x: x and any(term in x.lower() for term in ['product', 'artikel', 'item'])))
            
            clean_html = f"""
            Price-related elements:
            {' '.join(str(elem) for elem in price_elements)}
            
            Product section:
            {product_section}
            """

            prompt = """Extract product information from this HTML. 
            Find the following information:
            - Product name
            - Current price (as a decimal number, e.g., 9.99)
            - Currency symbol or code (e.g., €, EUR)
            - Any promotional text or deals
            
            Important: 
            - Look for price elements with classes containing 'price', 'prijs', 'amount'
            - Price might be formatted as "9.99", "€9.99", "9,99 EUR", or similar
            - Convert comma-separated prices to decimal (e.g., "9,99" → 9.99)
            
            Format the response as JSON with keys: 'name', 'price' (number), 'currency', 'promotion' (null if none)"""

            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model=self.deployment_name,
                messages=[{"role": "user", "content": f"{prompt}\n\nHTML content: {clean_html[:4000]}"}],
                response_format={ "type": "json_object" }
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            
            # Add metadata to the extracted data
            extracted_data.update({
                'timestamp': datetime.now().isoformat(),
                'url': self.url
            })
            
            return extracted_data

        except Exception as e:
            logging.error(f"LLM extraction error: {str(e)}")
            return None

    async def scrape(self) -> Optional[dict]:
        """
        Main scraping method that coordinates the entire scraping process
        
        Returns:
            Optional[dict]: Scraped product data or None if scraping failed
        """
        html_content, status_code = await self._safe_request()
        
        if html_content is None:
            return None

        # Check for blocking
        if await self._check_for_blocking(html_content, status_code):
            logging.error("Scraping blocked by website")
            return None

        # Save HTML response for debugging
        with open('last_response.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info("💾 Saved HTML response to 'last_response.html'")

        return await self._extract_data_with_llm(html_content)

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    url = "https://www.etos.nl/producten/therme-hammam-showergel-200-ml-110902232.html"
    scraper = SmartScraper(url)
    result = await scraper.scrape()
    
    if result:
        print("\n📦 Scraped Data:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("❌ Failed to scrape data")

if __name__ == "__main__":
    asyncio.run(main())
