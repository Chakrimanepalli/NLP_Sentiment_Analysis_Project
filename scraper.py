"""
Amazon Review Scraper
Extracts customer reviews from Amazon product pages
NO EMOJIS VERSION - STREAMLIT COMPATIBLE
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import random
import warnings
warnings.filterwarnings('ignore')

class AmazonReviewScraper:
    """Scraper class for extracting Amazon product reviews"""

    def __init__(self, product_url):
        self.product_url = product_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/'
        }

    def extract_asin(self):
        """Extract ASIN from Amazon product URL"""
        try:
            if '/dp/' in self.product_url:
                asin = self.product_url.split('/dp/')[1].split('/')[0].split('?')[0]
            elif '/product/' in self.product_url:
                asin = self.product_url.split('/product/')[1].split('/')[0].split('?')[0]
            else:
                asin = None
            return asin
        except Exception as e:
            print(f"Error extracting ASIN: {e}")
            return None

    def get_review_url(self, asin):
        """Generate review page URL from ASIN"""
        return f"https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"

    def scrape_reviews_selenium(self, max_reviews=1000):
        """Scrape reviews using Selenium for dynamic content"""
        asin = self.extract_asin()
        if not asin:
            print("Could not extract ASIN from URL")
            return pd.DataFrame()

        review_url = self.get_review_url(asin)

        # Setup Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')

        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Error initializing WebDriver: {e}")
            print("Please ensure ChromeDriver is installed and in PATH")
            return pd.DataFrame()

        reviews_data = []
        page_num = 1

        try:
            while len(reviews_data) < max_reviews:
                print(f"Scraping page {page_num}...")

                driver.get(review_url)
                time.sleep(random.uniform(2, 4))

                soup = BeautifulSoup(driver.page_source, 'lxml')
                review_elements = soup.find_all('div', {'data-hook': 'review'})

                if not review_elements:
                    print("No more reviews found")
                    break

                for review in review_elements:
                    try:
                        rating_elem = review.find('i', {'data-hook': 'review-star-rating'})
                        rating = rating_elem.get_text().strip().split()[0] if rating_elem else 'N/A'

                        title_elem = review.find('a', {'data-hook': 'review-title'})
                        title = title_elem.get_text().strip() if title_elem else 'N/A'

                        body_elem = review.find('span', {'data-hook': 'review-body'})
                        body = body_elem.get_text().strip() if body_elem else 'N/A'

                        date_elem = review.find('span', {'data-hook': 'review-date'})
                        date = date_elem.get_text().strip() if date_elem else 'N/A'

                        verified_elem = review.find('span', {'data-hook': 'avp-badge'})
                        verified = 'Yes' if verified_elem else 'No'

                        author_elem = review.find('span', {'class': 'a-profile-name'})
                        author = author_elem.get_text().strip() if author_elem else 'Anonymous'

                        reviews_data.append({
                            'rating': rating,
                            'title': title,
                            'review_text': body,
                            'date': date,
                            'verified_purchase': verified,
                            'author': author
                        })

                    except Exception as e:
                        continue

                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
                    review_url = next_button.get_attribute('href')
                    page_num += 1

                    if len(reviews_data) >= max_reviews:
                        break

                except:
                    print("No next page found")
                    break

        except Exception as e:
            print(f"Error during scraping: {e}")

        finally:
            driver.quit()

        df = pd.DataFrame(reviews_data)
        print(f"\nTotal reviews scraped: {len(df)}")

        return df

    def scrape_reviews_requests(self, max_reviews=1000):
        """Alternative scraper using requests library"""
        asin = self.extract_asin()
        if not asin:
            print("Could not extract ASIN from URL")
            return pd.DataFrame()

        review_url = self.get_review_url(asin)
        reviews_data = []
        page_num = 1

        while len(reviews_data) < max_reviews:
            try:
                print(f"Scraping page {page_num}...")

                response = requests.get(review_url, headers=self.headers)
                time.sleep(random.uniform(2, 5))

                if response.status_code != 200:
                    print(f"Failed to fetch page {page_num}: Status {response.status_code}")
                    break

                soup = BeautifulSoup(response.content, 'lxml')
                review_elements = soup.find_all('div', {'data-hook': 'review'})

                if not review_elements:
                    print("No more reviews found")
                    break

                for review in review_elements:
                    try:
                        rating_elem = review.find('i', {'data-hook': 'review-star-rating'})
                        rating = rating_elem.get_text().strip().split()[0] if rating_elem else 'N/A'

                        title_elem = review.find('a', {'data-hook': 'review-title'})
                        title = title_elem.get_text().strip() if title_elem else 'N/A'

                        body_elem = review.find('span', {'data-hook': 'review-body'})
                        body = body_elem.get_text().strip() if body_elem else 'N/A'

                        date_elem = review.find('span', {'data-hook': 'review-date'})
                        date = date_elem.get_text().strip() if date_elem else 'N/A'

                        verified_elem = review.find('span', {'data-hook': 'avp-badge'})
                        verified = 'Yes' if verified_elem else 'No'

                        author_elem = review.find('span', {'class': 'a-profile-name'})
                        author = author_elem.get_text().strip() if author_elem else 'Anonymous'

                        reviews_data.append({
                            'rating': rating,
                            'title': title,
                            'review_text': body,
                            'date': date,
                            'verified_purchase': verified,
                            'author': author
                        })

                    except Exception as e:
                        continue

                next_page = soup.find('li', {'class': 'a-last'})
                if next_page and next_page.find('a'):
                    review_url = 'https://www.amazon.in' + next_page.find('a')['href']
                    page_num += 1
                else:
                    break

            except Exception as e:
                print(f"Error: {e}")
                break

        df = pd.DataFrame(reviews_data)
        print(f"\nTotal reviews scraped: {len(df)}")

        return df


if __name__ == "__main__":
    product_url = "https://www.amazon.in/dp/B08L5VCMHR"

    scraper = AmazonReviewScraper(product_url)
    print("Starting review scraping with Selenium...")
    reviews_df = scraper.scrape_reviews_selenium(max_reviews=1000)

    if not reviews_df.empty:
        reviews_df.to_csv('amazon_reviews.csv', index=False, encoding='utf-8')
        print(f"\nReviews saved to 'amazon_reviews.csv'")
        print(f"\nSample reviews:")
        print(reviews_df.head())
    else:
        print("No reviews were scraped")
