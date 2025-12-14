# STMA - Stock Trading Media Analysis

A comprehensive financial data analysis and anomaly detection system for S&P 500 stocks that combines media sentiment analysis, financial data processing, and multi-method anomaly detection.

## Overview

This project provides an end-to-end pipeline for:
- **Data Collection**: Fetching financial data, news articles, and Reddit posts for S&P 500 companies
- **Sentiment Analysis**: Using BERT-based models to analyze article sentiment
- **Anomaly Detection**: Detecting market anomalies using statistical methods, media follow analysis, and event studies
- **Infrastructure**: AWS-based infrastructure with PostgreSQL database

## Project Structure

```
stma/
├── anomaly_det/          # Anomaly detection system
│   ├── indicators/       # Custom technical indicators
│   │   ├── velocity_indicator.py
│   │   ├── supertrend.py
│   │   ├── impulsemacd.py
│   │   ├── squeeze.py
│   │   └── obv.py
│   ├── event_study/     # Event study analysis outputs
│   ├── final_study_output/  # Final aggregated results and reports
│   ├── narrative_test/  # Media follow analysis outputs
│   ├── main.py          # Main orchestration pipeline (entry point)
│   ├── helper.py        # Data fetching from database
│   ├── helper2.py       # Market data processing utilities
│   ├── feature_engineering.py  # Technical indicator calculation
│   ├── statistical.py   # Statistical anomaly detection
│   ├── inference.py     # Aggregation and report generation
│   ├── media_follow.py  # Media follow analysis (trend follower vs contrarian)
│   └── study.py         # Event study analysis
├── data/                # Reference data (e.g., sp500.csv)
├── fin_data/            # Financial data files (CSV per ticker)
├── infra/               # Infrastructure as Code
│   ├── cdk/             # AWS CDK setup (Python)
│   └── db/              # Database schema (SQL)
├── pipelines/           # Data collection pipelines
│   ├── fin/             # Financial data fetchers
│   │   ├── fetch_fin.py      # Single ticker fetcher
│   │   └── fetch_fin_full.py # Batch ticker fetcher
│   ├── helpers/         # Pipeline helper functions
│   ├── scraper/         # Content scraping pipeline
│   ├── BERT.py          # BERT sentiment analysis pipeline
│   ├── reddit_scraper.py    # Reddit submission fetcher
│   └── sp500_media_pipeline.py  # Media Cloud article fetcher
├── scripts/             # Utility scripts
│   ├── check_duplicates.py
│   ├── delete_non_top250.py
│   ├── generate_visualizations.py
│   ├── init_db.sh
│   ├── run_bert_nohup.sh
│   ├── run_content_scraper_nohup.sh
│   ├── run_pipeline_nohup.sh
│   ├── run_pipeline.sh
│   └── seed_sp500.py
└── visualizations/      # Generated visualization outputs
```

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL client tools

### Environment Configuration

Create a `.env` file in the `pipelines/` directory with your credentials:

```env
# Database Configuration
DB_HOST=your-rds-endpoint
DB_PORT=5432
DB_NAME=projectdb
DB_USER=your-username
DB_PASSWORD=your-password

# Media Cloud API Keys (supports multiple keys for parallel processing)
MEDIA_CLOUD_API_KEY_1=your-api-key-1
MEDIA_CLOUD_API_KEY_2=your-api-key-2
MEDIA_CLOUD_API_KEY_3=your-api-key-3
```

### 3. Install Dependencies

Install project dependencies:
```bash
pip install -r requirements.txt
```

### Data Collection Pipelines

The pipelines are located in `pipelines/` and can be run independently:

#### Financial Data Collection
```bash
cd pipelines
python fin/fetch_fin.py          # Fetch single ticker
python fin/fetch_fin_full.py     # Fetch all tickers from CSV
```

#### Media Article Collection
```bash
cd pipelines
python sp500_media_pipeline.py   # Fetch articles from Media Cloud API
```

#### Content Scraping
```bash
cd pipelines
python scraper/scraper_pipeline.py  # Scrape full article content
```

#### Reddit Data Collection
```bash
cd pipelines
python reddit_scraper.py  # Fetch Reddit submissions for S&P 500 companies
```

#### BERT Sentiment Analysis
```bash
cd pipelines
python BERT.py  # Analyze article sentiment using FinBERT
```

#### Running the Complete Pipeline

The main entry point runs all processing steps in order:

```bash
python main.py
```

#### Output Directories

The pipeline generates outputs in three directories:
- `final_study_output/`: Silent shock reports and time series plots
- `narrative_test/`: Media behavior classification (trend follower vs contrarian)
- `event_study/`: Event study statistics and trajectory plots
- `visualizations/`: Visualizations of market data and technical indicators along with bias index


## Output

The main pipeline (`main.py`) generates:
### Impact Analysis
- CSV and PNG tables with impact analysis results
Output of `main.py` is stored in `visualizations/` and `final_study_output/`

### Final Study Output (`final_study_output/`) only anomaly detection
- **Silent Shocks Report**: Text report analyzing silent anomalies
- **Time Series Plot**: Visualization of silent shock patterns over time

### Narrative Test (`narrative_test/`)
- **Media Behavior Classification**: Pie chart showing trend follower vs contrarian stocks
- **Master Report**: CSV with classification results for all stocks

### Event Study (`event_study/`)
- **Event Study Statistics**: CSV and PNG table with statistical significance results
- **Trajectory Plot**: Visualization of media bias path before and after price shocks