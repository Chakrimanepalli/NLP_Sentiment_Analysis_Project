'''
Streamlit Dashboard for Amazon Review Sentiment Analysis
Complete end-to-end sentiment analysis and price recommendation system
NO EMOJIS VERSION - PRODUCTION READY FOR STREAMLIT
'''

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from scraper import AmazonReviewScraper
    from sentiment_model import SentimentAnalyzer, TraditionalSentimentClassifier
    from price_tracker import PriceTracker
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

st.set_page_config(
    page_title="Amazon Sentiment Analysis & Price Tracker",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-box {
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        background-color: #f0f8f0;
        margin: 1rem 0;
    }
</style>
''', unsafe_allow_html=True)

if 'reviews_df' not in st.session_state:
    st.session_state.reviews_df = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'price_tracker' not in st.session_state:
    st.session_state.price_tracker = PriceTracker()


def generate_sample_reviews(num_reviews=1000):
    np.random.seed(42)

    positive_reviews = [
        "Excellent product! Highly recommended.",
        "Best purchase ever. Great quality and fast delivery.",
        "Amazing product, exceeded my expectations.",
        "Very satisfied with this purchase. Worth every penny.",
        "Outstanding quality and performance."
    ]

    negative_reviews = [
        "Poor quality. Not worth the money.",
        "Disappointed with this product. Would not recommend.",
        "Waste of money. Product stopped working after a week.",
        "Very bad experience. Customer service was unhelpful.",
        "Not as described. Quality is terrible."
    ]

    neutral_reviews = [
        "Product is okay. Nothing special.",
        "Average quality for the price.",
        "It works but could be better.",
        "Decent product, meets basic expectations.",
        "Neither good nor bad. Just average."
    ]

    reviews_list = []
    for i in range(num_reviews):
        sentiment_choice = np.random.choice(['positive', 'neutral', 'negative'], p=[0.65, 0.20, 0.15])

        if sentiment_choice == 'positive':
            review_text = np.random.choice(positive_reviews)
            rating = np.random.choice(['4.0', '5.0'])
        elif sentiment_choice == 'negative':
            review_text = np.random.choice(negative_reviews)
            rating = np.random.choice(['1.0', '2.0'])
        else:
            review_text = np.random.choice(neutral_reviews)
            rating = '3.0'

        date = datetime.now() - timedelta(days=np.random.randint(1, 365))

        reviews_list.append({
            'rating': rating,
            'title': f"Review {i+1}",
            'review_text': review_text,
            'date': date.strftime('%B %d, %Y'),
            'verified_purchase': np.random.choice(['Yes', 'No'], p=[0.8, 0.2]),
            'author': f"Customer_{i+1}"
        })

    return pd.DataFrame(reviews_list)


def display_wordcloud(text, title):
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")


def main():
    st.markdown('<h1 class="main-header">Amazon Review Sentiment Analysis & Price Tracker</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.title("Controls")

        data_source = st.radio(
            "Select Data Source:",
            ["Scrape Amazon Reviews", "Upload CSV File", "Use Sample Data"]
        )

        st.markdown("---")
        st.markdown("### Project Info")
        st.info("""
        Features:
        - Web scraping from Amazon
        - VADER & TextBlob sentiment analysis
        - LSTM neural network classification
        - Price history from launch date
        - Buy recommendations
        - Interactive visualizations
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Collection", 
        "Sentiment Analysis", 
        "ML Classification",
        "Price & Recommendations",
        "Dashboard"
    ])

    with tab1:
        st.header("Data Collection")

        if data_source == "Scrape Amazon Reviews":
            st.subheader("Scrape Reviews from Amazon")

            product_url = st.text_input(
                "Enter Amazon Product URL:",
                placeholder="https://www.amazon.in/dp/B08L5VCMHR"
            )

            max_reviews = st.slider("Number of Reviews to Scrape:", 100, 2000, 1000, 100)

            col1, col2 = st.columns(2)
            with col1:
                scrape_button = st.button("Start Scraping", type="primary")
            with col2:
                use_selenium = st.checkbox("Use Selenium (more reliable)", value=True)

            if scrape_button and product_url:
                with st.spinner("Scraping reviews... This may take a few minutes."):
                    try:
                        scraper = AmazonReviewScraper(product_url)

                        if use_selenium:
                            reviews_df = scraper.scrape_reviews_selenium(max_reviews)
                        else:
                            reviews_df = scraper.scrape_reviews_requests(max_reviews)

                        if not reviews_df.empty:
                            st.session_state.reviews_df = reviews_df
                            st.success(f"Successfully scraped {len(reviews_df)} reviews!")
                            st.dataframe(reviews_df.head(10))

                            csv = reviews_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="amazon_reviews.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("No reviews were scraped. Please check the URL and try again.")

                    except Exception as e:
                        st.error(f"Error during scraping: {str(e)}")

        elif data_source == "Upload CSV File":
            st.subheader("Upload Your Reviews CSV")
            st.info("CSV should contain columns: review_text, rating, date")

            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    reviews_df = pd.read_csv(uploaded_file)
                    st.session_state.reviews_df = reviews_df
                    st.success(f"Loaded {len(reviews_df)} reviews!")
                    st.dataframe(reviews_df.head(10))
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

        else:
            st.subheader("Using Sample Amazon Reviews")

            if st.button("Generate Sample Data"):
                sample_reviews = generate_sample_reviews(1000)
                st.session_state.reviews_df = sample_reviews
                st.success(f"Generated {len(sample_reviews)} sample reviews!")
                st.dataframe(sample_reviews.head(10))

    with tab2:
        st.header("Sentiment Analysis Results")

        if st.session_state.reviews_df is not None:
            reviews_df = st.session_state.reviews_df

            if st.button("Analyze Sentiments", type="primary"):
                with st.spinner("Analyzing sentiments..."):
                    analyzer = st.session_state.analyzer

                    sentiment_results = []
                    progress_bar = st.progress(0)

                    for idx, text in enumerate(reviews_df['review_text']):
                        result = analyzer.analyze_sentiment(text)
                        sentiment_results.append(result)
                        progress_bar.progress((idx + 1) / len(reviews_df))

                    sentiment_df = pd.DataFrame(sentiment_results)
                    result_df = pd.concat([reviews_df, sentiment_df], axis=1)

                    st.session_state.sentiment_results = result_df

                    st.success("Sentiment analysis complete!")

            if st.session_state.sentiment_results is not None:
                result_df = st.session_state.sentiment_results

                st.subheader("Sentiment Summary")
                col1, col2, col3, col4 = st.columns(4)

                sentiment_counts = result_df['ensemble_sentiment'].value_counts()
                total = len(result_df)

                with col1:
                    positive_pct = (sentiment_counts.get('Positive', 0) / total) * 100
                    st.metric("Positive Reviews", f"{positive_pct:.1f}%", 
                             delta=f"{sentiment_counts.get('Positive', 0)} reviews")

                with col2:
                    neutral_pct = (sentiment_counts.get('Neutral', 0) / total) * 100
                    st.metric("Neutral Reviews", f"{neutral_pct:.1f}%",
                             delta=f"{sentiment_counts.get('Neutral', 0)} reviews")

                with col3:
                    negative_pct = (sentiment_counts.get('Negative', 0) / total) * 100
                    st.metric("Negative Reviews", f"{negative_pct:.1f}%",
                             delta=f"{sentiment_counts.get('Negative', 0)} reviews")

                with col4:
                    avg_compound = result_df['vader_compound'].mean()
                    st.metric("Avg VADER Score", f"{avg_compound:.3f}",
                             delta="Compound Score")

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positive': '#4CAF50',
                            'Neutral': '#FFC107',
                            'Negative': '#F44336'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    fig_hist = px.histogram(
                        result_df,
                        x='vader_compound',
                        nbins=50,
                        title="VADER Compound Score Distribution",
                        labels={'vader_compound': 'Compound Score'},
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("---")
                st.subheader("Word Cloud")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Positive Words"):
                        positive_text = ' '.join(result_df[result_df['ensemble_sentiment']=='Positive']['review_text'])
                        if positive_text:
                            display_wordcloud(positive_text, "Positive Reviews")

                with col2:
                    if st.button("Neutral Words"):
                        neutral_text = ' '.join(result_df[result_df['ensemble_sentiment']=='Neutral']['review_text'])
                        if neutral_text:
                            display_wordcloud(neutral_text, "Neutral Reviews")

                with col3:
                    if st.button("Negative Words"):
                        negative_text = ' '.join(result_df[result_df['ensemble_sentiment']=='Negative']['review_text'])
                        if negative_text:
                            display_wordcloud(negative_text, "Negative Reviews")

                st.markdown("---")
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Sentiment Results",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

        else:
            st.warning("Please collect data first (Tab 1)")

    with tab3:
        st.header("Machine Learning Classification")

        if st.session_state.sentiment_results is not None:
            result_df = st.session_state.sentiment_results

            st.subheader("Train LSTM Model")

            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Training Epochs:", 5, 20, 10)
            with col2:
                batch_size = st.selectbox("Batch Size:", [64, 128, 256], index=1)

            if st.button("Train LSTM Model", type="primary"):
                with st.spinner("Training LSTM model... This may take several minutes."):
                    try:
                        analyzer = st.session_state.analyzer
                        result_df['sentiment'] = result_df['ensemble_sentiment']

                        history = analyzer.train_lstm_model(
                            result_df, 
                            epochs=epochs,
                            batch_size=batch_size
                        )

                        st.success("Model training complete!")

                        analyzer.save_model()
                        st.info("Model saved successfully!")

                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")

            st.markdown("---")
            st.subheader("Test Prediction")

            test_text = st.text_area("Enter text to analyze:", 
                                     "This product is absolutely amazing! Best purchase ever.")

            if st.button("Predict Sentiment"):
                analyzer = st.session_state.analyzer
                result = analyzer.analyze_sentiment(test_text)

                st.write("### Prediction Results")
                st.write(f"**VADER Sentiment:** {result['vader_sentiment']}")
                st.write(f"**VADER Compound Score:** {result['vader_compound']:.3f}")
                st.write(f"**TextBlob Sentiment:** {result['textblob_sentiment']}")
                st.write(f"**TextBlob Polarity:** {result['textblob_polarity']:.3f}")
                st.write(f"**Ensemble Prediction:** {result['ensemble_sentiment']}")

        else:
            st.warning("Please analyze sentiments first (Tab 2)")

    with tab4:
        st.header("Price History & Buy Recommendations")

        st.subheader("Product Price Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            current_price = st.number_input("Current Price (Rs):", min_value=100.0, value=1499.0, step=100.0)
        with col2:
            launch_date = st.date_input("Product Launch Date:", 
                                       value=datetime.now() - timedelta(days=365))
        with col3:
            days_history = st.slider("Days of History:", 30, 730, 365)

        if st.button("Generate Price History", type="primary"):
            tracker = st.session_state.price_tracker

            price_history = tracker.generate_realistic_price_history(
                current_price=current_price,
                launch_date=str(launch_date),
                days_back=days_history
            )

            st.success("Price history generated!")

            st.subheader("Price History Chart")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=price_history['date'],
                y=price_history['price'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ))

            stats = tracker.get_price_statistics()

            fig.add_hline(y=stats['highest_price'], line_dash="dash", 
                         line_color="red", 
                         annotation_text=f"Highest: Rs {stats['highest_price']:.2f}")

            fig.add_hline(y=stats['lowest_price'], line_dash="dash", 
                         line_color="green",
                         annotation_text=f"Lowest: Rs {stats['lowest_price']:.2f}")

            fig.add_hline(y=stats['current_price'], line_dash="dot",
                         line_color="orange",
                         annotation_text=f"Current: Rs {stats['current_price']:.2f}")

            fig.update_layout(
                title="Product Price History (Launch Date to Current)",
                xaxis_title="Date",
                yaxis_title="Price (Rs)",
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Price Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"Rs {stats['current_price']:.2f}")
                st.metric("Highest Price", f"Rs {stats['highest_price']:.2f}")

            with col2:
                st.metric("Lowest Price", f"Rs {stats['lowest_price']:.2f}")
                st.metric("Average Price", f"Rs {stats['average_price']:.2f}")

            with col3:
                st.metric("7-Day Average", f"Rs {stats['7_day_avg']:.2f}")
                st.metric("30-Day Average", f"Rs {stats['30_day_avg']:.2f}")

            with col4:
                st.metric("Price Trend", stats['price_trend'])
                st.metric("Days Tracked", stats['days_tracked'])

            st.markdown("---")
            st.subheader("Buy Recommendation")

            if st.session_state.sentiment_results is not None:
                result_df = st.session_state.sentiment_results
                sentiment_counts = result_df['ensemble_sentiment'].value_counts()
                total = len(result_df)
                positive_pct = (sentiment_counts.get('Positive', 0) / total) * 100
            else:
                positive_pct = 70.0
                st.info("Using default sentiment score of 70%. Analyze reviews in Tab 2 for accurate recommendations.")

            recommendation = tracker.get_buy_recommendation(positive_pct)

            st.markdown(f'''
            <div class="recommendation-box">
                <h2 style="text-align: center;">Recommendation: {recommendation['recommendation']}</h2>
                <h3 style="text-align: center; color: #666;">Confidence: {recommendation['confidence']}%</h3>
                <hr>
                <p><strong>Price Score:</strong> {recommendation['price_score']:.1f}%</p>
                <p><strong>Sentiment Score:</strong> {recommendation['sentiment_score']:.1f}%</p>
                <p><strong>Best Price Ever:</strong> Rs {recommendation['best_price_ever']:.2f}</p>
                <p><strong>Current vs Best:</strong> {recommendation['current_vs_best']:.1f}%</p>
                <hr>
                <p><strong>Reasons:</strong></p>
                <ul>
                    {''.join([f"<li>{reason}</li>" for reason in recommendation['reasons']])}
                </ul>
            </div>
            ''', unsafe_allow_html=True)

            alerts = tracker.get_price_alerts()
            if alerts:
                st.subheader("Price Alerts")
                for alert in alerts:
                    st.warning(alert)

    with tab5:
        st.header("Executive Dashboard")

        if st.session_state.sentiment_results is not None:
            result_df = st.session_state.sentiment_results

            st.subheader("Overall Summary")

            sentiment_counts = result_df['ensemble_sentiment'].value_counts()
            total = len(result_df)
            positive_pct = (sentiment_counts.get('Positive', 0) / total) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Reviews", total)
            with col2:
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            with col3:
                avg_rating = result_df['rating'].astype(str).str.replace(',', '.').astype(float).mean()
                st.metric("Average Rating", f"{avg_rating:.2f}/5.0")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment Breakdown")
                fig = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title="Review Count by Sentiment",
                    labels={'x': 'Sentiment', 'y': 'Count'},
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Neutral': '#FFC107',
                        'Negative': '#F44336'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("VADER Score Distribution")
                fig = px.box(
                    result_df,
                    y='vader_compound',
                    title="VADER Compound Score Range",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Please analyze data in previous tabs to see the dashboard.")


if __name__ == "__main__":
    main()
