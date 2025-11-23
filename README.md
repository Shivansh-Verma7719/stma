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

1. **Install Dependencies**:
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
