-- Migration script to add state tracking columns to companies table
-- Run this on existing databases to add resumable processing support

-- Add current_page column (tracks pagination progress for each company)
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS current_page INTEGER DEFAULT 0;

-- Add is_processed column (marks whether company processing is complete)
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS is_processed BOOLEAN DEFAULT FALSE;

-- Add last_error column (stores last error message for debugging)
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS last_error TEXT;

-- Add last_updated column (tracks when state was last modified)
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Create index for efficient querying of unprocessed companies
CREATE INDEX IF NOT EXISTS idx_companies_is_processed ON companies(is_processed);

-- Update trigger to automatically set last_updated on any row modification
CREATE OR REPLACE FUNCTION update_companies_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_companies_last_updated ON companies;
CREATE TRIGGER trigger_companies_last_updated
    BEFORE UPDATE ON companies
    FOR EACH ROW
    EXECUTE FUNCTION update_companies_last_updated();
