"""
Price Tracker and Buy Recommendation System
Tracks price history from first selling date to current date
Shows highest sold, lowest sold, and current selling price
NO EMOJIS VERSION - STREAMLIT COMPATIBLE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PriceTracker:
    """Price tracking and recommendation system with historical data"""

    def __init__(self):
        self.price_history = pd.DataFrame()
        self.product_launch_date = None
        self.current_price = None

    def generate_realistic_price_history(self, current_price, launch_date=None, days_back=365):
        """
        Generate realistic price history from product launch date to current date

        Args:
            current_price: Current selling price of the product
            launch_date: Product launch date (default: 1 year ago)
            days_back: Number of days of history (default: 365)
        """
        self.current_price = current_price

        if launch_date is None:
            self.product_launch_date = datetime.now() - timedelta(days=days_back)
        else:
            self.product_launch_date = pd.to_datetime(launch_date)

        days_since_launch = (datetime.now() - self.product_launch_date).days

        dates = pd.date_range(start=self.product_launch_date, end=datetime.now(), freq='D')
        num_days = len(dates)

        np.random.seed(42)

        initial_price = current_price * 1.15

        trend = np.linspace(initial_price, current_price, num_days)

        seasonal_component = np.sin(np.linspace(0, 8 * np.pi, num_days)) * (current_price * 0.08)

        noise = np.random.normal(0, current_price * 0.03, num_days)

        sale_days = np.random.choice(num_days, size=int(num_days * 0.15), replace=False)
        sale_discount = np.zeros(num_days)
        sale_discount[sale_days] = -current_price * np.random.uniform(0.10, 0.30, len(sale_days))

        prices = trend + seasonal_component + noise + sale_discount

        prices = np.clip(prices, current_price * 0.65, initial_price * 1.1)

        prices[-1] = current_price

        self.price_history = pd.DataFrame({
            'date': dates,
            'price': prices,
            'is_sale': False
        })

        self.price_history.loc[sale_days, 'is_sale'] = True

        return self.price_history

    def get_price_statistics(self):
        """Calculate comprehensive price statistics"""
        if self.price_history.empty:
            return {}

        df = self.price_history
        current_price = df['price'].iloc[-1]

        stats = {
            'current_price': float(current_price),
            'highest_price': float(df['price'].max()),
            'lowest_price': float(df['price'].min()),
            'average_price': float(df['price'].mean()),
            'median_price': float(df['price'].median()),
            'price_volatility': float(df['price'].std()),

            'launch_date': str(df['date'].min().date()),
            'current_date': str(df['date'].max().date()),
            'days_tracked': len(df),

            '7_day_avg': float(df.tail(7)['price'].mean()) if len(df) >= 7 else float(current_price),
            '30_day_avg': float(df.tail(30)['price'].mean()) if len(df) >= 30 else float(current_price),
            '90_day_avg': float(df.tail(90)['price'].mean()) if len(df) >= 90 else float(current_price),

            'price_trend': self._calculate_trend(),
            'total_sales_events': int(df['is_sale'].sum()),
            'last_sale_date': str(df[df['is_sale']]['date'].max().date()) if df['is_sale'].any() else 'No sales yet'
        }

        stats['vs_highest'] = ((current_price - stats['highest_price']) / stats['highest_price']) * 100
        stats['vs_lowest'] = ((current_price - stats['lowest_price']) / stats['lowest_price']) * 100
        stats['vs_average'] = ((current_price - stats['average_price']) / stats['average_price']) * 100

        return stats

    def _calculate_trend(self):
        """Calculate price trend based on recent data"""
        if len(self.price_history) < 30:
            return 'Insufficient Data'

        recent_30 = self.price_history.tail(30)['price'].mean()
        previous_30 = self.price_history.tail(60).head(30)['price'].mean() if len(self.price_history) >= 60 else recent_30

        if recent_30 < previous_30 * 0.95:
            return 'Decreasing'
        elif recent_30 > previous_30 * 1.05:
            return 'Increasing'
        else:
            return 'Stable'

    def get_buy_recommendation(self, sentiment_score):
        """
        Generate buy recommendation based on price analysis and sentiment

        Args:
            sentiment_score: Positive sentiment percentage (0-100)

        Returns:
            dict: Recommendation details with confidence and reasons
        """
        stats = self.get_price_statistics()

        if not stats:
            return {
                'recommendation': 'Insufficient Data',
                'confidence': 0,
                'reasons': ['Not enough price history available']
            }

        current_price = stats['current_price']
        avg_price = stats['average_price']
        lowest_price = stats['lowest_price']
        highest_price = stats['highest_price']

        price_score = 0

        price_position = (current_price - lowest_price) / (highest_price - lowest_price) if highest_price != lowest_price else 0.5

        if price_position <= 0.15:
            price_score += 40
        elif price_position <= 0.30:
            price_score += 30
        elif price_position <= 0.50:
            price_score += 20
        elif price_position <= 0.70:
            price_score += 10

        if stats['price_trend'] == 'Decreasing':
            price_score += 20
        elif stats['price_trend'] == 'Stable':
            price_score += 10

        volatility_ratio = stats['price_volatility'] / avg_price if avg_price > 0 else 0
        if volatility_ratio < 0.05:
            price_score += 15
        elif volatility_ratio < 0.10:
            price_score += 10
        elif volatility_ratio < 0.15:
            price_score += 5

        if current_price < stats['7_day_avg']:
            price_score += 10

        if current_price < stats['30_day_avg']:
            price_score += 5

        final_score = (price_score * 0.6) + (sentiment_score * 0.4)

        if final_score >= 75:
            recommendation = 'Strong Buy'
        elif final_score >= 60:
            recommendation = 'Buy'
        elif final_score >= 45:
            recommendation = 'Consider Buying'
        elif final_score >= 30:
            recommendation = 'Wait'
        else:
            recommendation = 'Do Not Buy'

        reasons = []
        if current_price <= lowest_price * 1.05:
            reasons.append('Price at or near historical low')
        if stats['price_trend'] == 'Decreasing':
            reasons.append('Price trend is decreasing')
        if stats['price_trend'] == 'Increasing':
            reasons.append('Price trend is increasing - may want to buy before further increase')
        if sentiment_score >= 70:
            reasons.append(f'High positive customer sentiment ({sentiment_score:.1f}%)')
        elif sentiment_score < 40:
            reasons.append(f'Low customer satisfaction ({sentiment_score:.1f}%)')
        if current_price < avg_price:
            reasons.append(f'Price {abs(stats["vs_average"]):.1f}% below average')
        else:
            reasons.append(f'Price {abs(stats["vs_average"]):.1f}% above average')
        if current_price < stats['7_day_avg']:
            reasons.append('Price below 7-day average')
        if stats['total_sales_events'] > 0:
            reasons.append(f'{stats["total_sales_events"]} sale events detected in history')

        return {
            'recommendation': recommendation,
            'confidence': round(final_score, 1),
            'price_score': round(price_score, 1),
            'sentiment_score': round(sentiment_score, 1),
            'reasons': reasons,
            'best_price_ever': stats['lowest_price'],
            'worst_price_ever': stats['highest_price'],
            'current_vs_best': round(stats['vs_lowest'], 1),
            'current_vs_worst': round(stats['vs_highest'], 1),
            'price_position': f"{price_position*100:.1f}% through price range"
        }

    def get_price_alerts(self):
        """Generate price alerts and notifications"""
        stats = self.get_price_statistics()
        alerts = []

        if stats['current_price'] <= stats['lowest_price'] * 1.03:
            alerts.append("ALERT: Price is within 3% of all-time low!")

        if stats['price_trend'] == 'Decreasing':
            alerts.append("TREND: Price has been decreasing - good time to monitor")

        if stats['vs_average'] < -10:
            alerts.append(f"DEAL: Price is {abs(stats['vs_average']):.1f}% below average")

        if stats['current_price'] < stats['7_day_avg'] * 0.95:
            alerts.append("ALERT: Price dropped 5% below 7-day average")

        return alerts


if __name__ == "__main__":
    tracker = PriceTracker()

    current_price = 1499.0
    launch_date = "2024-01-15"

    print("Generating price history...")
    history = tracker.generate_realistic_price_history(
        current_price=current_price,
        launch_date=launch_date,
        days_back=365
    )

    print(f"\nPrice history generated from {history['date'].min().date()} to {history['date'].max().date()}")

    stats = tracker.get_price_statistics()
    print("\nPrice Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    sentiment_score = 75.5
    recommendation = tracker.get_buy_recommendation(sentiment_score)

    print("\nBuy Recommendation:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")

    alerts = tracker.get_price_alerts()
    print("\nPrice Alerts:")
    for alert in alerts:
        print(f"  - {alert}")
