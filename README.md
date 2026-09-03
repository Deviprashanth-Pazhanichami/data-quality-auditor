# 🔍 Data Quality Score Auditor

I built this because I kept hearing that "data cleaning" is a huge part of any analyst's job, but it's usually done ad-hoc — a bit of code in a notebook, never reused. So I made a tool instead: point it at any CSV and it checks for missing values, duplicates, outliers, and formatting issues, then gives you one health score (0–100) and a ranked list of what to fix first.

**🔗 Try it here:** https://data-quality-score-auditor.streamlit.app/

## What it does

- Checks for missing values, duplicate rows, statistical outliers, and formatting problems (inconsistent casing, stray whitespace, mixed date formats)
- Combines all of that into a single weighted Health Score
- Ranks every issue as HIGH/MEDIUM/LOW so you know what to fix first, not just what's wrong
- Has an "Auto-Clean" button that fixes what's safe to fix automatically, and tells you exactly what it changed

## A decision I made on purpose: outliers don't get deleted

My first instinct was to just delete anything statistically weird. But then I tested it on a real movies dataset and realized a $200M movie budget isn't a data entry error — it's just a blockbuster. Deleting it would throw away real information. So instead, Auto-Clean caps outliers at a reasonable range instead of removing them, and logs it so you can review the decision yourself.

## Tested on real data, not just made-up examples

- The Titanic passenger dataset — correctly caught that 77% of `Cabin` values are missing, which matches what's actually documented about this dataset
- A real IMDB movie dataset (~7,600 movies) — scored it 96.6/100, correctly flagged that 28% of movies are missing a reported budget
- A messy dataset I built myself with mixed date formats and inconsistent casing on purpose — Auto-Clean got it to a perfect 100/100

## Built with

Python (pandas, numpy) for the core logic, Streamlit for the web app, Plotly for the charts and the 3D outlier view.

## Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

The core checker also works on its own from the command line, no web app needed:

```bash
python data_quality_auditor.py your_data.csv
```

## Files

- `app.py` — the Streamlit web app
- `data_quality_auditor.py` — the actual logic (works standalone too)
- `requirements.txt` — dependencies
