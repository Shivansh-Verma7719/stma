-- Migration to add sentiment score columns to articles table
BEGIN;

-- Add columns for positive and negative sentiment scores
ALTER TABLE articles 
ADD COLUMN IF NOT EXISTS pos_score FLOAT,
ADD COLUMN IF NOT EXISTS neg_score FLOAT;

-- Add indexes for querying by sentiment
CREATE INDEX IF NOT EXISTS idx_articles_pos_score ON articles(pos_score);
CREATE INDEX IF NOT EXISTS idx_articles_neg_score ON articles(neg_score);

COMMIT;
