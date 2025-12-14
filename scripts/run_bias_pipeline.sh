# Runs the bias pipeline
# first it will bias_calculator.py then normalize_bias.py
# then it will run impact_analysis.py

python pipelines/bias_calculator.py --prod

python pipelines/normalize_bias_scores.py

python pipelines/impact_analysis.py
