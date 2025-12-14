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

## Infrastructure

The infrastructure is managed using AWS CDK and consists of:
- **VPC**: One Public VPC (No NAT Gateway)
- **RDS**: PostgreSQL Database (Max 100GB, PostgreSQL 16)
- **Deployment**: Automated via GitHub Actions workflow `.github/workflows/deploy.yml` on changes to `infra/cdk/`

## Getting Started

### Prerequisites

- Python 3.8+
- AWS CLI configured
- CDK CLI installed (`npm install -g aws-cdk`)
- PostgreSQL client tools

### 1. Infrastructure Setup

1. **Install CDK Dependencies**:
   ```bash
   cd infra/cdk
   pip install -r requirements.txt
   ```

2. **Deploy Infrastructure**:
   ```bash
   cd infra/cdk
   cdk deploy
   ```

3. **Database Setup**:
   Apply the schema in `infra/db/schema.sql` to your RDS instance:
   ```bash
   psql -h <DB_HOST> -U <DB_USER> -d projectdb -f infra/db/schema.sql
   ```

### 2. Environment Configuration

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

### 4. Data Collection Pipelines

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

### 5. Anomaly Detection Pipeline

The anomaly detection system processes financial data through an integrated pipeline:

#### Running the Complete Pipeline

The main entry point runs all processing steps in order:

```bash
cd anomaly_det
python main.py
```

This single command executes the complete pipeline:
1. **Data Fetching**: Retrieves data from the database
2. **Market Data Processing**: Processes and normalizes market data (renames `norm_bias_score` to `bias_index`)
3. **Feature Engineering**: Adds technical indicators (velocity, OBV, MACD, impulse, supertrend, squeeze)
4. **Statistical Anomaly Detection**: Detects silent shocks and volume anomalies using Z-scores
5. **Result Aggregation**: Aggregates statistical results and generates reports
6. **Media Follow Analysis**: Analyzes whether media follows or contradicts price trends
7. **Event Study**: Studies media reaction to extreme price movements (>2%)

#### Output Directories

The pipeline generates outputs in three directories:
- `final_study_output/`: Silent shock reports and time series plots
- `narrative_test/`: Media behavior classification (trend follower vs contrarian)
- `event_study/`: Event study statistics and trajectory plots

## Anomaly Detection Methods

The system uses multiple complementary analysis methods:

1. **Statistical Anomaly Detection** (`statistical.py`)
   - Z-score analysis for price movements
   - Volume shock detection (relative volume > 3x)
   - "Silent anomaly" detection (price/volume shocks without corresponding sentiment)
   - Identifies days where market moves occur without matching media sentiment

2. **Media Follow Analysis** (`media_follow.py`)
   - Tests whether media sentiment follows or contradicts price movements
   - Classifies stocks as "Trend Follower" or "Contrarian" based on regression analysis
   - Uses 14-day lag to test if price returns today affect future media bias

3. **Event Study** (`study.py`)
   - Analyzes media reaction to extreme price movements (>2% moves)
   - Tracks bias trajectory 5 days before and 10 days after price shocks
   - Statistical significance testing for media response patterns
   - Generates trajectory plots and statistical tables

## Technical Indicators

The system calculates various technical indicators:
- **Velocity Indicator**: Price momentum measure
- **OBV (On-Balance Volume)**: Volume-weighted price indicator
- **MACD**: Moving Average Convergence Divergence
- **Impulse Histogram**: Momentum indicator
- **SuperTrend**: Trend-following indicator
- **Squeeze Indicator**: Volatility compression indicator

## Scripts

Utility scripts in `scripts/`:
- `seed_sp500.py`: Seed database with S&P 500 company data
- `check_duplicates.py`: Check for duplicate entries
- `delete_non_top250.py`: Clean up non-top 250 companies
- `generate_visualizations.py`: Generate data visualizations
- `run_pipeline.sh`: Run pipelines in tmux session
- `init_db.sh`: Initialize database schema

## Output

The anomaly detection pipeline generates:

### Final Study Output (`final_study_output/`)
- **Silent Shocks Report**: Text report analyzing silent anomalies
- **Time Series Plot**: Visualization of silent shock patterns over time

### Narrative Test (`narrative_test/`)
- **Media Behavior Classification**: Pie chart showing trend follower vs contrarian stocks
- **Master Report**: CSV with classification results for all stocks

### Event Study (`event_study/`)
- **Event Study Statistics**: CSV and PNG table with statistical significance results
- **Trajectory Plot**: Visualization of media bias path before and after price shocks

## Development

### Running Pipelines in Background

Use the provided shell scripts to run pipelines in tmux sessions:
```bash
./scripts/run_pipeline.sh
./scripts/run_bert_nohup.sh
./scripts/run_content_scraper_nohup.sh
```

### Monitoring

- Pipeline logs are saved to `pipelines/logs/`
- Use `tmux attach -t <session_name>` to monitor running pipelines

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `pandas`, `numpy`: Data processing
- `scikit-learn`: Machine learning models
- `yfinance`: Financial data fetching
- `mediacloud`: Media Cloud API client
- `transformers`, `torch`: BERT sentiment analysis
- `psycopg2`: PostgreSQL database connection
- `beautifulsoup4`, `newspaper3k`: Web scraping
- `matplotlib`, `seaborn`: Visualization