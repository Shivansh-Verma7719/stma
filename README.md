# STMA - Stock Trading Media Analysis

A comprehensive financial data analysis and anomaly detection system for S&P 500 stocks that combines media sentiment analysis, financial data processing, and multi-method anomaly detection.

## Overview

This project provides an end-to-end pipeline for:
- **Data Collection**: Fetching financial data, news articles, and Reddit posts for S&P 500 companies
- **Sentiment Analysis**: Using BERT-based models to analyze article sentiment
- **Anomaly Detection**: Detecting market anomalies using multiple ML methods (regression, statistical, isolation forest, clustering)
- **Infrastructure**: AWS-based infrastructure with PostgreSQL database

## Project Structure

```
stma/
├── anomaly_det/          # Anomaly detection system
│   ├── clustering/       # DBSCAN clustering results
│   ├── fetch/            # Raw financial data
│   ├── fin_data/         # Processed financial data with bias index
│   ├── fin_process/      # Feature-engineered data
│   ├── forest/           # Isolation Forest results
│   ├── regression/       # Regression anomaly results
│   ├── statistical/     # Statistical anomaly results
│   ├── indicators/      # Custom technical indicators
│   ├── clustering.py    # DBSCAN clustering anomaly detection
│   ├── feature_engineering.py  # Technical indicator calculation
│   ├── forest.py        # Isolation Forest anomaly detection
│   ├── helper.py        # Data preparation utilities
│   ├── inference.py     # Main inference pipeline (merges all methods)
│   ├── regression.py    # Regression-based anomaly detection
│   └── statistical.py   # Statistical anomaly detection
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

The anomaly detection system processes financial data through multiple methods:

#### Step 1: Feature Engineering
Add technical indicators to financial data:
```bash
cd anomaly_det
python feature_engineering.py
```

#### Step 2: Run Anomaly Detection Methods

Run each detection method (they can be run in parallel):

```bash
# Regression-based anomaly detection
python regression.py

# Statistical anomaly detection (Z-scores, volume shocks)
python statistical.py

# Isolation Forest anomaly detection
python forest.py

# DBSCAN clustering anomaly detection
python clustering.py
```

#### Step 3: Inference and Aggregation
Merge all method results and generate reports:
```bash
python inference.py
```

This will:
- Merge results from all 4 detection methods
- Generate visualizations
- Create a comprehensive text report
- Save aggregated results to `final_study_output/`

## Anomaly Detection Methods

The system uses four complementary methods:

1. **Regression Anomaly Detection** (`regression.py`)
   - Models the relationship between bias index (sentiment) and stock returns
   - Identifies days where actual returns deviate significantly from expected returns

2. **Statistical Anomaly Detection** (`statistical.py`)
   - Z-score analysis for price movements
   - Volume shock detection (relative volume > 3x)
   - "Silent anomaly" detection (price/volume shocks without corresponding sentiment)

3. **Isolation Forest** (`forest.py`)
   - Unsupervised ML method that identifies outliers
   - Uses multiple features: bias_index, velocity, volatility (psi), MACD, momentum, trend

4. **DBSCAN Clustering** (`clustering.py`)
   - Clusters days based on sentiment vs. returns relationship
   - Identifies noise points (anomalies) that don't fit normal patterns

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
- **CSV Files**: Per-method results in respective directories
- **Aggregated Dataset**: `final_study_output/study_master_dataset.csv`
- **Visualizations**: Plots saved to `final_study_output/`
- **Text Report**: Comprehensive analysis report

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