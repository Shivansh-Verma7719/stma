# Project Setup

This project contains data pipelines and infrastructure code.

## Structure

- `pipelines/`: Contains data pipelines (e.g., S&P 500, Reddit).
- `helpers/`: General helper functions.
- `infra/`: Infrastructure code.
  - `cdk/`: AWS CDK setup (Python).
  - `db/`: Database schema (SQL).

## Infrastructure

The infrastructure is managed using AWS CDK and consists of:
- One Public VPC (No NAT Gateway).
- RDS PostgreSQL Database (Max 100GB).

## Deployment

The infrastructure is deployed via GitHub Actions workflow `.github/workflows/deploy.yml` on changes to `infra/cdk/`.

## Getting Started

### Infrastructure
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
   Apply the schema in `infra/db/schema.sql` to your RDS instance.

### Pipelines (Data Collection)
The pipelines are a standalone Python application located in `pipelines/`.

1. **Setup**:
   ```bash
   cd pipelines
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Ensure `pipelines/.env` exists with your credentials:
   ```
   DB_HOST=...
   DB_PORT=5432
   DB_NAME=projectdb
   DB_USER=...
   DB_PASSWORD=...
   MEDIA_CLOUD_API_KEY=...
   ```

3. **Run Pipelines**:
   ```bash
   # 1. Fetch Articles
   python sp500_media_pipeline.py

   # 2. Scrape Content
   python content_scraper_pipeline.py
   ```
