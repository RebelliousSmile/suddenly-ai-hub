#!/usr/bin/env python3
"""
Scraper JDROLL.org avec Playwright

Extraction des campagnes de jdroll.org
Utilise Playwright pour executer le JavaScript AngularJS

Usage:
    python scripts/scraper_jdroll.py [--output OUTPUT] [--pages PAGES]
"""

import argparse
import json
import time
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class JDROLLScraper:
    """Scraper pour JDROLL.org using Playwright"""
    
    BASE_URL = "http://www.jdroll.org"
    FORUM_URL = f"{BASE_URL}/forum/825"
    
    def __init__(self, output_path="/tmp/jdroll_campaigns.json", max_pages=10):
        self.output_path = Path(output_path)
        self.max_pages = max_pages
        self.campaigns = []
        
    def scrape_page(self, page, url):
        """Scrape a single page"""
        logger.info(f"Scraping: {url}")
        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait for AngularJS to load
            try:
                page.wait_for_selector('.campaign', timeout=10000)
            except PlaywrightTimeout:
                page.wait_for_selector('.campagne', timeout=10000)
            
            time.sleep(2)  # Extra wait for dynamic content
            
            campaigns = self._extract_campaigns(page)
            logger.info(f"Found {len(campaigns)} campaigns")
            self.campaigns.extend(campaigns)
            
            return True
            
        except PlaywrightTimeout as e:
            logger.warning(f"Timeout on page: {url}")
            return False
        except Exception as e:
            logger.error(f"Error scraping page: {e}")
            return False
    
    def _extract_campaigns(self, page):
        """Extract campaign data from the page"""
        campaigns = page.evaluate('''() => {
            const elements = document.querySelectorAll(".campaign, .campagne, [class*='campaign']");
            const results = [];
            
            elements.forEach((el, index) => {
                const title = el.querySelector(".title, h2, h3, h4")?.textContent?.trim();
                const description = el.querySelector(".description, .summary")?.textContent?.trim();
                const universe = el.querySelector(".universe, .genre, .world")?.textContent?.trim();
                const system = el.querySelector(".system, .rule-set")?.textContent?.trim();
                const author = el.querySelector(".author, .creator")?.textContent?.trim();
                const link = el.querySelector("a[href*='/campagne/']")?.href || null;
                
                if (title || link) {
                    results.push({
                        id: index,
                        title: title,
                        description: description,
                        universe: universe,
                        system: system,
                        author: author,
                        link: link,
                        scraped_at: new Date().toISOString()
                    });
                }
            });
            
            return results;
        }''')
        
        return campaigns
    
    def scrape(self):
        """Main scraping method"""
        logger.info(f"Starting JDROLL scraper...")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Max pages: {self.max_pages}")
        
        with sync_playwright() as p:
            logger.info("Launching browser...")
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu'
                ]
            )
            
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='fr-FR',
                timezone_id='Europe/Paris'
            )
            
            page = context.new_page()
            
            page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            for page_num in range(1, self.max_pages + 1):
                logger.info(f"Page {page_num}/{self.max_pages}")
                
                url = self.FORUM_URL if page_num == 1 else f"{self.FORUM_URL}?page={page_num}"
                
                if not self.scrape_page(page, url):
                    logger.warning(f"Failed to scrape page {page_num}")
                    break
                
                # Respectful delay
                time.sleep(2)
            
            browser.close()
            logger.info("Browser closed")
        
        return self.campaigns
    
    def save(self, campaigns=None):
        """Save campaigns to file"""
        campaigns = campaigns or self.campaigns
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        output_data = {
            'total_campaigns': len(campaigns),
            'scraped_at': __import__('datetime').datetime.now().isoformat(),
            'source': 'jdroll.org',
            'scrape_method': 'Playwright (AngularJS)',
            'campaigns': campaigns
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(campaigns)} campaigns to {self.output_path}")
        
        # Also save as JSONL
        jsonl_path = self.output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for campaign in campaigns:
                f.write(json.dumps(campaign, ensure_ascii=False) + '\n')
        
        logger.info(f"Also saved to {jsonl_path}")
        
        return campaigns


def main():
    parser = argparse.ArgumentParser(description='Scrape JDROLL.org campaigns')
    parser.add_argument('--output', '-o', default='/tmp/jdroll_campaigns.json',
                       help='Output file path')
    parser.add_argument('--pages', '-p', type=int, default=10,
                       help='Number of pages to scrape')
    
    args = parser.parse_args()
    
    scraper = JDROLLScraper(output_path=args.output, max_pages=args.pages)
    campaigns = scraper.scrape()
    scraper.save(campaigns)
    
    print(f"\n{'='*80}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"Total campaigns found: {len(campaigns)}")
    print(f"Output file: {scraper.output_path}")
    
    if campaigns:
        print(f"\nSample campaign:")
        sample = campaigns[0]
        for key, value in sample.items():
            if key not in ['source_url', 'scraped_at']:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
