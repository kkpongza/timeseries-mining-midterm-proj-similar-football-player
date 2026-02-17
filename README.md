# Player Similarity Dashboard

An interactive dashboard to visualize and compare football players using spatial heatmaps transformed into serpentine time-series representations, with player similarity computed via cosine similarity on behavioral wave patterns.

## Direct Installation (Bash)

Copy and paste these commands into your terminal to get started immediately:

```bash
# 1. Clone & Enter Directory
git clone https://github.com/kkpongza/timeseries-mining-midterm-proj-similar-football-player.git
cd timeseries-mining-midterm-proj-similar-football-player

# 2. Install Dependencies (including gdown for Google Drive)
pip install streamlit pandas matplotlib gdown

# 3. Download the Dataset from Google Drive
gdown --folder 1CcgvtBeu7rCiFmrsOwY6DAr2LUzBYeAd

# 4. Move CSV files to project root
mv timeseries_mining_proj_midterm_similar_football_player/*.csv . # (MAC)
Move-Item timeseries_mining_proj_midterm_similar_football_player\*.csv . # (Window)

# 5. Run the Application
streamlit run app.py
```

## Dashboard Features
- **Feature Selection** Choose spatial behavior type (passing, receiving, carries, position, goals).
- **Player Similarity:** Find top-K similar players using cosine similarity on serpentine wave representations.
- **Wave Comparison:** Visualize behavioral patterns as 1D time-series signals.
- **Heatmap Visualization:** Compare spatial activity heatmaps across players on a full football pitch.

---
*Data Source: Based on StatsBomb Open Data (2015/16 Premier League Season).*
