-- Add norm_bias_score column to bias_index table
ALTER TABLE "public"."bias_index" 
ADD COLUMN IF NOT EXISTS "norm_bias_score" numeric;
