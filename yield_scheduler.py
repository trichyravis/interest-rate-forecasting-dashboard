"""
Automated Yield Data Scheduler
Collects India 10Y G-Sec yield data at regular intervals

Developed by: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import logging
import json
import os
from india_yield_fetcher import IntelligentYieldFetcher, CacheManager

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/yield_scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

SCHEDULES = {
    'fred_weekly': {
        'schedule_type': 'cron',
        'day_of_week': 'mon',
        'hour': 8,
        'minute': 0,
        'description': 'FRED data update (Weekly on Monday 8 AM)'
    },
    'web_scrape_daily': {
        'schedule_type': 'cron',
        'hour': 18,
        'minute': 0,
        'description': 'Web scraping update (Daily at 6 PM)'
    },
    'cache_check': {
        'schedule_type': 'interval',
        'hours': 6,
        'description': 'Cache freshness check (Every 6 hours)'
    }
}

# ============================================================================
# SCHEDULER CLASS
# ============================================================================

class YieldDataScheduler:
    """Manages automated yield data collection"""
    
    def __init__(self, trading_econ_api_key=None):
        self.scheduler = BackgroundScheduler()
        self.fetcher = IntelligentYieldFetcher(trading_econ_api_key=trading_econ_api_key)
        self.stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'last_fetch_time': None,
            'last_yield_value': None,
            'sources_used': {}
        }
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def fetch_yield_task(self):
        """Background task to fetch yield data"""
        
        logger.info("=" * 70)
        logger.info("📊 YIELD FETCH TASK STARTED")
        logger.info("=" * 70)
        
        self.stats['total_fetches'] += 1
        
        result = self.fetcher.fetch_yield(use_cache=True, cache_ttl_hours=6)
        
        if result:
            value, source = result
            self.stats['successful_fetches'] += 1
            self.stats['last_yield_value'] = value
            self.stats['last_fetch_time'] = datetime.now().isoformat()
            
            # Track source usage
            if source not in self.stats['sources_used']:
                self.stats['sources_used'][source] = 0
            self.stats['sources_used'][source] += 1
            
            logger.info(f"✅ FETCH SUCCESSFUL")
            logger.info(f"   Yield: {value:.3f}%")
            logger.info(f"   Source: {source}")
            logger.info(f"   Timestamp: {datetime.now().isoformat()}")
            
            # Save statistics
            self._save_statistics()
            
        else:
            self.stats['failed_fetches'] += 1
            logger.error(f"❌ FETCH FAILED")
            logger.error(f"   Total failures: {self.stats['failed_fetches']}")
            
            # Alert if too many failures
            if self.stats['failed_fetches'] > 3:
                self._send_alert(
                    f"⚠️ ALERT: {self.stats['failed_fetches']} consecutive fetch failures",
                    severity='HIGH'
                )
        
        logger.info("=" * 70)
    
    def cache_check_task(self):
        """Check cache freshness"""
        
        cache_age = CacheManager.get_cache_age()
        
        if cache_age:
            hours_old = cache_age / 3600
            logger.info(f"📦 Cache status: {hours_old:.1f} hours old")
            
            if hours_old > 24:
                logger.warning(f"⚠️ Cache is {hours_old:.1f} hours old - consider fetching fresh data")
        else:
            logger.warning("⚠️ No cache data available")
    
    def add_jobs(self):
        """Add all scheduled jobs"""
        
        logger.info("🔧 Adding scheduled jobs...")
        
        # Weekly FRED fetch
        self.scheduler.add_job(
            self.fetch_yield_task,
            trigger='cron',
            day_of_week='mon',
            hour=8,
            minute=0,
            id='fred_fetch',
            name='FRED Weekly Fetch',
            replace_existing=True
        )
        logger.info("✅ Added: FRED Weekly Fetch (Monday 8 AM)")
        
        # Daily web scrape
        self.scheduler.add_job(
            self.fetch_yield_task,
            trigger='cron',
            hour=18,
            minute=0,
            id='web_scrape',
            name='Web Scrape Fetch',
            replace_existing=True
        )
        logger.info("✅ Added: Web Scrape Fetch (Daily 6 PM)")
        
        # 6-hourly cache check
        self.scheduler.add_job(
            self.cache_check_task,
            trigger='interval',
            hours=6,
            id='cache_check',
            name='Cache Freshness Check',
            replace_existing=True
        )
        logger.info("✅ Added: Cache Check (Every 6 hours)")
        
        # Immediate initial fetch
        self.scheduler.add_job(
            self.fetch_yield_task,
            trigger='date',
            run_date=datetime.now() + timedelta(seconds=5),
            id='initial_fetch',
            name='Initial Fetch',
            replace_existing=True,
            misfire_grace_time=10
        )
        logger.info("✅ Added: Initial Fetch (in 5 seconds)")
    
    def start(self):
        """Start the scheduler"""
        
        logger.info("=" * 70)
        logger.info("🚀 STARTING YIELD DATA SCHEDULER")
        logger.info("=" * 70)
        
        self.add_jobs()
        self.scheduler.start()
        
        logger.info("✅ Scheduler started successfully")
        logger.info(f"📋 Active jobs: {len(self.scheduler.get_jobs())}")
        
        for job in self.scheduler.get_jobs():
            logger.info(f"   - {job.name} (id: {job.id})")
    
    def stop(self):
        """Stop the scheduler gracefully"""
        
        logger.info("=" * 70)
        logger.info("🛑 STOPPING SCHEDULER")
        logger.info("=" * 70)
        
        self.scheduler.shutdown(wait=True)
        
        logger.info("✅ Scheduler stopped")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Total Fetches: {self.stats['total_fetches']}")
        logger.info(f"   Successful: {self.stats['successful_fetches']}")
        logger.info(f"   Failed: {self.stats['failed_fetches']}")
        logger.info(f"   Last Yield: {self.stats['last_yield_value']}")
    
    def get_statistics(self):
        """Get scheduler statistics"""
        return self.stats
    
    def _save_statistics(self):
        """Save statistics to file"""
        
        stats_file = 'data/scheduler_stats.json'
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Error saving statistics: {e}")
    
    def _send_alert(self, message, severity='INFO'):
        """Send alert notifications"""
        
        logger.warning(f"\n{'='*70}")
        logger.warning(f"🚨 ALERT ({severity}): {message}")
        logger.warning(f"{'='*70}\n")
        
        # TODO: Implement email/SMS notifications
        # Example: Send email, Slack message, etc.

# ============================================================================
# FLASK INTEGRATION FOR STREAMLIT
# ============================================================================

class StreamlitIntegration:
    """Integration utilities for Streamlit dashboard"""
    
    def __init__(self, scheduler: YieldDataScheduler):
        self.scheduler = scheduler
    
    def display_scheduler_status(self):
        """Display scheduler status in Streamlit"""
        
        try:
            import streamlit as st
            
            stats = self.scheduler.get_statistics()
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔄 Data Scheduler Status")
            
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                st.metric("Total Fetches", stats['total_fetches'])
                st.metric("Successful", stats['successful_fetches'])
            
            with col2:
                st.metric("Failed", stats['failed_fetches'])
                if stats['last_fetch_time']:
                    st.metric("Last Fetch", stats['last_fetch_time'][-8:])
            
            if stats['sources_used']:
                st.sidebar.markdown("**Sources Used:**")
                for source, count in stats['sources_used'].items():
                    st.sidebar.write(f"- {source}: {count}x")
        
        except ImportError:
            pass

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for scheduler"""
    
    logger.info("Starting India 10Y Yield Data Scheduler")
    
    # Initialize scheduler
    scheduler = YieldDataScheduler(
        trading_econ_api_key=os.getenv('TRADING_ECON_API_KEY')
    )
    
    # Start scheduler
    scheduler.start()
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")
        scheduler.stop()

if __name__ == "__main__":
    main()

# ============================================================================
# QUICK REFERENCE: RUN INSTRUCTIONS
# ============================================================================

"""
SETUP INSTRUCTIONS:

1. Install dependencies:
   pip install -r requirements.txt

2. Create logs and data directories:
   mkdir logs
   mkdir data

3. Run scheduler standalone:
   python yield_scheduler.py

4. Run with environment variable (Trading Economics API):
   export TRADING_ECON_API_KEY="your_api_key"
   python yield_scheduler.py

5. Integrate with Streamlit (in app.py):
   from yield_scheduler import YieldDataScheduler, StreamlitIntegration
   
   scheduler = YieldDataScheduler()
   scheduler.start()
   
   integration = StreamlitIntegration(scheduler)
   integration.display_scheduler_status()

SCHEDULED TASKS:
- Monday 8 AM (IST): FRED weekly fetch
- Daily 6 PM (IST): Web scraping attempt
- Every 6 hours: Cache freshness check
- Initial: First fetch on startup (5 seconds delay)
"""
