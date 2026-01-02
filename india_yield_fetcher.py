"""
India 10-Year G-Sec Yield Data Fetcher
Multiple source integration with fallback mechanisms

Developed by: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import logging
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'fred_series': 'INDIRLTLT01STM',  # India 10Y Monthly
    'min_yield': 0.5,                  # Minimum reasonable yield (%)
    'max_yield': 15.0,                 # Maximum reasonable yield (%)
    'cache_dir': './data',
    'cache_file': 'india_yield_cache.json',
    'timeout': 30,
    'retry_attempts': 3
}

# ============================================================================
# SOURCE 1: FRED (Federal Reserve Economic Data)
# ============================================================================

class FREDDataFetcher:
    """Fetch India 10Y yield from FRED database"""
    
    @staticmethod
    def fetch_data(series_code: str = CONFIG['fred_series']) -> Optional[pd.DataFrame]:
        """
        Fetch data from FRED using pandas_datareader
        
        Args:
            series_code: FRED series code (INDIRLTLT01STM for monthly)
        
        Returns:
            pd.DataFrame with yield data, or None if failed
        """
        try:
            import pandas_datareader as pdr
            
            logger.info(f"🔄 Fetching from FRED (series: {series_code})...")
            
            data = pdr.get_data_fred(series_code)
            
            # Rename column
            data.columns = ['Yield']
            data = data[data['Yield'].notna()]
            
            logger.info(f"✅ FRED fetch successful: {len(data)} records")
            return data
        
        except ImportError:
            logger.error("❌ pandas_datareader not installed. Install with: pip install pandas_datareader")
            return None
        except Exception as e:
            logger.error(f"❌ FRED fetch failed: {str(e)}")
            return None
    
    @staticmethod
    def get_latest_yield(data: pd.DataFrame) -> Optional[Tuple[float, str]]:
        """
        Extract latest yield from FRED data
        
        Returns:
            Tuple of (yield_value, date_string) or None
        """
        try:
            latest_value = data.iloc[-1, 0]
            latest_date = data.index[-1].strftime('%Y-%m-%d')
            
            logger.info(f"✅ Latest FRED yield: {latest_value:.3f}% ({latest_date})")
            return (latest_value, latest_date)
        except Exception as e:
            logger.error(f"❌ Error extracting latest yield: {e}")
            return None

# ============================================================================
# SOURCE 2: TRADING ECONOMICS (Web Scraping)
# ============================================================================

class TradingEconomicsDataFetcher:
    """Fetch India 10Y yield from Trading Economics website"""
    
    BASE_URL = "https://api.tradingeconomics.com"
    WEBSITE_URL = "https://tradingeconomics.com/india/government-bond-yield"
    
    @staticmethod
    def fetch_via_api(api_key: Optional[str] = None) -> Optional[float]:
        """
        Fetch using Trading Economics API (requires registration)
        
        Args:
            api_key: Your Trading Economics API key
        
        Returns:
            Latest yield value or None
        """
        if not api_key:
            logger.warning("⚠️ No API key provided for Trading Economics")
            return None
        
        try:
            logger.info("🔄 Fetching from Trading Economics API...")
            
            url = f"{TradingEconomicsDataFetcher.BASE_URL}/country/india"
            params = {'c': api_key}
            
            response = requests.get(url, params=params, timeout=CONFIG['timeout'])
            response.raise_for_status()
            
            data = response.json()
            
            # Find 10-year bond yield
            for item in data:
                if '10-year' in item.get('name', '').lower() and 'bond' in item.get('name', '').lower():
                    value = float(item['last'])
                    logger.info(f"✅ Trading Econ API: {value:.3f}%")
                    return value
            
            logger.warning("⚠️ 10-year bond yield not found in API response")
            return None
        
        except Exception as e:
            logger.error(f"❌ Trading Econ API fetch failed: {e}")
            return None
    
    @staticmethod
    def fetch_via_web_scraping() -> Optional[float]:
        """
        Fetch by scraping Trading Economics website
        
        Returns:
            Latest yield value or None
        """
        try:
            logger.info("🔄 Scraping Trading Economics website...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                TradingEconomicsDataFetcher.WEBSITE_URL,
                headers=headers,
                timeout=CONFIG['timeout']
            )
            response.raise_for_status()
            
            # Parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for yield value (pattern may change - monitor carefully)
            # Try multiple selectors for robustness
            selectors = [
                ('span', {'class': 'lighterText'}),
                ('span', {'data-test': 'instrument-header-last-price'}),
                ('div', {'class': 'stat-value'})
            ]
            
            for tag, attrs in selectors:
                element = soup.find(tag, attrs)
                if element:
                    text = element.text.strip().replace('%', '')
                    try:
                        value = float(text)
                        logger.info(f"✅ Trading Econ scraping: {value:.3f}%")
                        return value
                    except ValueError:
                        continue
            
            logger.warning("⚠️ Could not extract yield from Trading Econ page")
            return None
        
        except ImportError:
            logger.error("❌ BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return None
        except Exception as e:
            logger.error(f"❌ Trading Econ scraping failed: {e}")
            return None

# ============================================================================
# SOURCE 3: INVESTING.COM (Web Scraping)
# ============================================================================

class InvestingComDataFetcher:
    """Fetch India 10Y yield from Investing.com"""
    
    URL = "https://www.investing.com/rates-bonds/india-10-year-bond-yield-historical-data"
    
    @staticmethod
    def fetch_via_scraping() -> Optional[float]:
        """
        Fetch by scraping Investing.com
        
        Returns:
            Latest yield value or None
        """
        try:
            logger.info("🔄 Scraping Investing.com...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                InvestingComDataFetcher.URL,
                headers=headers,
                timeout=CONFIG['timeout']
            )
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price data
            price_elem = soup.find('span', {'data-test': 'instrument-header-last-price'})
            
            if price_elem:
                value = float(price_elem.text.strip())
                logger.info(f"✅ Investing.com: {value:.3f}%")
                return value
            
            logger.warning("⚠️ Could not extract yield from Investing.com")
            return None
        
        except ImportError:
            logger.error("❌ BeautifulSoup not installed")
            return None
        except Exception as e:
            logger.error(f"❌ Investing.com scraping failed: {e}")
            return None

# ============================================================================
# SOURCE 4: WORLD GOVERNMENT BONDS (Web Scraping)
# ============================================================================

class WorldGovtBondsDataFetcher:
    """Fetch India 10Y yield from World Government Bonds"""
    
    URL = "http://www.worldgovernmentbonds.com/bond-historical-data/india/10-years/"
    
    @staticmethod
    def fetch_via_scraping() -> Optional[float]:
        """
        Fetch by scraping World Government Bonds
        
        Returns:
            Latest yield value or None
        """
        try:
            logger.info("🔄 Scraping World Government Bonds...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                WorldGovtBondsDataFetcher.URL,
                headers=headers,
                timeout=CONFIG['timeout']
            )
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text and look for yield pattern
            text_content = soup.get_text()
            
            import re
            pattern = r'(\d+\.\d+)\s*%'
            matches = re.findall(pattern, text_content)
            
            if matches:
                # Usually first match is current yield
                value = float(matches[0])
                
                # Validate
                if CONFIG['min_yield'] <= value <= CONFIG['max_yield']:
                    logger.info(f"✅ World Govt Bonds: {value:.3f}%")
                    return value
            
            logger.warning("⚠️ Could not extract yield from World Govt Bonds")
            return None
        
        except ImportError:
            logger.error("❌ BeautifulSoup not installed")
            return None
        except Exception as e:
            logger.error(f"❌ World Govt Bonds scraping failed: {e}")
            return None

# ============================================================================
# CACHING LAYER
# ============================================================================

class CacheManager:
    """Manage local cache of yield data"""
    
    @staticmethod
    def save_yield(yield_value: float, source: str):
        """Save yield to cache"""
        try:
            import os
            os.makedirs(CONFIG['cache_dir'], exist_ok=True)
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'yield': yield_value,
                'source': source,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
            cache_file = f"{CONFIG['cache_dir']}/{CONFIG['cache_file']}"
            
            # Read existing data
            try:
                with open(cache_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
            
            # Append new data
            history.append(cache_data)
            
            # Keep last 500 records
            history = history[-500:]
            
            # Write back
            with open(cache_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"✅ Cached yield: {yield_value:.3f}% from {source}")
        
        except Exception as e:
            logger.error(f"❌ Cache save failed: {e}")
    
    @staticmethod
    def load_latest_yield() -> Optional[Tuple[float, str]]:
        """Load latest cached yield"""
        try:
            cache_file = f"{CONFIG['cache_dir']}/{CONFIG['cache_file']}"
            
            with open(cache_file, 'r') as f:
                history = json.load(f)
            
            if history:
                latest = history[-1]
                return (latest['yield'], latest['source'])
        
        except Exception as e:
            logger.warning(f"⚠️ Cache load failed: {e}")
        
        return None
    
    @staticmethod
    def get_cache_age() -> Optional[int]:
        """Get age of cache in seconds"""
        try:
            cache_file = f"{CONFIG['cache_dir']}/{CONFIG['cache_file']}"
            
            with open(cache_file, 'r') as f:
                history = json.load(f)
            
            if history:
                cached_time = datetime.fromisoformat(history[-1]['timestamp'])
                age = (datetime.now() - cached_time).total_seconds()
                return int(age)
        
        except:
            return None

# ============================================================================
# VALIDATION & DATA QUALITY
# ============================================================================

class DataValidator:
    """Validate yield data quality"""
    
    @staticmethod
    def validate_yield(yield_value: float) -> bool:
        """Check if yield is within reasonable range"""
        
        if yield_value < CONFIG['min_yield'] or yield_value > CONFIG['max_yield']:
            logger.warning(
                f"⚠️ Yield {yield_value:.3f}% outside expected range "
                f"({CONFIG['min_yield']}-{CONFIG['max_yield']}%)"
            )
            return False
        
        return True
    
    @staticmethod
    def check_data_freshness(last_update_time: datetime, max_age_days: int = 7) -> bool:
        """Check if data is fresh enough"""
        
        age_days = (datetime.now() - last_update_time).days
        
        if age_days > max_age_days:
            logger.warning(f"⚠️ Data is {age_days} days old (max: {max_age_days})")
            return False
        
        return True

# ============================================================================
# MAIN FETCHER CLASS (INTELLIGENT FALLBACK)
# ============================================================================

class IntelligentYieldFetcher:
    """
    Main class for fetching India 10Y yield with intelligent fallback
    
    Priority order:
    1. FRED (most reliable)
    2. Trading Economics API (if key provided)
    3. Trading Economics Scraping
    4. Investing.com Scraping
    5. World Govt Bonds Scraping
    6. Local Cache
    """
    
    def __init__(self, trading_econ_api_key: Optional[str] = None):
        self.trading_econ_api_key = trading_econ_api_key
        self.last_source = None
    
    def fetch_yield(self, use_cache: bool = True, cache_ttl_hours: int = 6) -> Optional[Tuple[float, str]]:
        """
        Fetch yield with intelligent fallback
        
        Args:
            use_cache: Whether to use cached data if available
            cache_ttl_hours: Cache time-to-live in hours
        
        Returns:
            Tuple of (yield_value, source_name) or None if all sources fail
        """
        
        logger.info("=" * 60)
        logger.info("🚀 Starting intelligent yield fetch...")
        logger.info("=" * 60)
        
        # 1. Check cache first
        if use_cache:
            cache_age = CacheManager.get_cache_age()
            if cache_age and cache_age < (cache_ttl_hours * 3600):
                cached_data = CacheManager.load_latest_yield()
                if cached_data:
                    value, source = cached_data
                    logger.info(f"✅ Using cached yield: {value:.3f}% from {source} ({cache_age//3600}h old)")
                    self.last_source = source
                    return (value, source)
        
        # 2. Try FRED (most reliable)
        fred_data = FREDDataFetcher.fetch_data()
        if fred_data is not None:
            result = FREDDataFetcher.get_latest_yield(fred_data)
            if result:
                value, date = result
                if DataValidator.validate_yield(float(value)):
                    CacheManager.save_yield(float(value), 'FRED')
                    self.last_source = 'FRED'
                    return (float(value), 'FRED')
        
        # 3. Try Trading Economics API
        if self.trading_econ_api_key:
            te_value = TradingEconomicsDataFetcher.fetch_via_api(self.trading_econ_api_key)
            if te_value and DataValidator.validate_yield(te_value):
                CacheManager.save_yield(te_value, 'Trading Economics API')
                self.last_source = 'Trading Economics API'
                return (te_value, 'Trading Economics API')
        
        # 4. Try Trading Economics Scraping
        te_scrape_value = TradingEconomicsDataFetcher.fetch_via_web_scraping()
        if te_scrape_value and DataValidator.validate_yield(te_scrape_value):
            CacheManager.save_yield(te_scrape_value, 'Trading Economics (Scrape)')
            self.last_source = 'Trading Economics (Scrape)'
            return (te_scrape_value, 'Trading Economics (Scrape)')
        
        # 5. Try Investing.com
        inv_value = InvestingComDataFetcher.fetch_via_scraping()
        if inv_value and DataValidator.validate_yield(inv_value):
            CacheManager.save_yield(inv_value, 'Investing.com')
            self.last_source = 'Investing.com'
            return (inv_value, 'Investing.com')
        
        # 6. Try World Government Bonds
        wgb_value = WorldGovtBondsDataFetcher.fetch_via_scraping()
        if wgb_value and DataValidator.validate_yield(wgb_value):
            CacheManager.save_yield(wgb_value, 'World Govt Bonds')
            self.last_source = 'World Govt Bonds'
            return (wgb_value, 'World Govt Bonds')
        
        # 7. Fall back to cache (even if old)
        cached_data = CacheManager.load_latest_yield()
        if cached_data:
            value, source = cached_data
            logger.warning(f"⚠️ Using stale cache: {value:.3f}% from {source}")
            self.last_source = f'{source} (STALE)'
            return (value, source)
        
        # All sources failed
        logger.error("❌ ALL SOURCES FAILED - Cannot fetch India 10Y yield")
        return None

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of the fetcher"""
    
    # Create fetcher instance
    fetcher = IntelligentYieldFetcher(trading_econ_api_key=None)
    
    # Fetch yield with cache
    result = fetcher.fetch_yield(use_cache=True, cache_ttl_hours=6)
    
    if result:
        value, source = result
        print(f"\n✅ SUCCESS:")
        print(f"   Yield: {value:.3f}%")
        print(f"   Source: {source}")
    else:
        print("\n❌ FAILED: Could not fetch yield from any source")

if __name__ == "__main__":
    example_usage()
