from flask import Flask, jsonify, request, send_from_directory, render_template, redirect, url_for
from flask_cors import CORS

import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

RAW_DATA_DIR = "../data/raw"
PREPARED_DATA_DIR = "../data/prepared"

print("--- LOADING DATA ---")

print("Loading games...")
games = pd.read_csv(os.path.join(RAW_DATA_DIR, "games.csv"))
print("Loaded games!")

print("Loading plays...")
plays = pd.read_csv(os.path.join(RAW_DATA_DIR, "plays.csv"))
print("Loaded plays!")

# Load the pre-processed and filtered tracking data
TRACKING_FILE = os.path.join(PREPARED_DATA_DIR, "output.parquet")
print(f"Loading processed tracking data ...")
tracking = pd.read_parquet(TRACKING_FILE)
print(f"Loaded processed tracking data! ({len(tracking)} rows)")

# Get unique plays available in the loaded tracking data
valid_plays = tracking[["gameId", "playId"]].drop_duplicates()
print(f"Identified {len(valid_plays)} unique plays available in the tracking data.")

# Simplified team colors dictionary (Abbr -> Primary Color)
team_colors = {
    "ARI": "0x97233F",  # Arizona Cardinals: Cardinal Red
    "ATL": "0xA71930",  # Atlanta Falcons: Red
    "BAL": "0x241773",  # Baltimore Ravens: Purple
    "BUF": "0x003087",  # Buffalo Bills: Blue
    "CAR": "0x0085CA",  # Carolina Panthers: Panther Blue
    "CHI": "0x0B162A",  # Chicago Bears: Navy Blue
    "CIN": "0xFB4F14",  # Cincinnati Bengals: Orange
    "CLE": "0xFF3C00",  # Cleveland Browns: Brown
    "DAL": "0x002244",  # Dallas Cowboys: Navy Blue
    "DEN": "0xFB4F14",  # Denver Broncos: Orange
    "DET": "0x005A8B",  # Detroit Lions: Honolulu Blue
    "GB":  "0x203731",  # Green Bay Packers: Dark Green
    "HOU": "0x03202F",  # Houston Texans: Deep Steel Blue
    "IND": "0x002C5F",  # Indianapolis Colts: Blue
    "JAX": "0x006778",  # Jacksonville Jaguars: Teal
    "KC":  "0xE31837",  # Kansas City Chiefs: Red
    "LA":  "0x003087",  # Los Angeles Rams: Royal Blue
    "LAC": "0x002244",  # Los Angeles Chargers: Navy Blue
    "LV":  "0xA5ACAF",  # Las Vegas Raiders: Silver
    "MIA": "0x008E97",  # Miami Dolphins: Aqua
    "MIN": "0x4F2683",  # Minnesota Vikings: Purple
    "NE":  "0x002244",  # New England Patriots: Navy Blue
    "NO":  "0xD3A625",  # New Orleans Saints: Old Gold
    "NYG": "0x0A2342",  # New York Giants: Dark Blue
    "NYJ": "0x203731",  # New York Jets: Dark Green
    "PHI": "0x004953",  # Philadelphia Eagles: Midnight Green
    "PIT": "0xFFB612",  # Pittsburgh Steelers: Yellow
    "SF":  "0xAA0000",  # San Francisco 49ers: Red
    "SEA": "0x002244",  # Seattle Seahawks: College Navy
    "TB":  "0xD50A0A",  # Tampa Bay Buccaneers: Red
    "TEN": "0x0C2340",  # Tennessee Titans: Navy Blue
    "WAS": "0x773141"   # Washington Commanders: Burgundy
}

def calculate_ball_trajectory(play_data):
    outcome_events = ["pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete", "pass_outcome_interception", "pass_outcome_touchdown"]

    ball_data = play_data[play_data["displayName"] == "football"].copy()

    if "pass_forward" in ball_data["event"].values and any(x in ball_data["event"].values for x in outcome_events):
        pass_start = ball_data[ball_data["event"] == "pass_forward"]["frameId"].iloc[0]
        pass_end = ball_data[ball_data["event"].isin(outcome_events)]["frameId"].iloc[0]
        T = (pass_end - pass_start) * 0.10  # 4 FPS
        vz = (0.5 * 10.71 * T**2 - 2.187) / T
        ball_data["z"] = 0
        mask = (ball_data["frameId"] >= pass_start) & (ball_data["frameId"] <= pass_end)
        ball_data.loc[mask, "z"] = ball_data[mask].apply(
            lambda row: 2.187 + vz * ((row["frameId"] - pass_start) * 0.10) - 
                        0.5 * 10.71 * ((row["frameId"] - pass_start) * 0.10)**2, axis=1
        )
    else:
        ball_data["z"] = 0

    return ball_data

# Serve static files
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/visualize/<int:game_id>/<int:play_id>")
def visualize(game_id, play_id):
    """Serve the visualization page with embedded parameters"""
    print(f"Visualizing game {game_id}, play {play_id}")
    
    # Serve the visualization page with parameters
    return send_from_directory(".", "visualize.html")

@app.route("/<path:filename>")
def static_files(filename):
    print(f"Serving static file: {filename}")
    try:
        # Set proper MIME type for JavaScript modules
        if filename.endswith(".js"):
            return send_from_directory(".", filename, mimetype="application/javascript")
        return send_from_directory(".", filename)
    except Exception as e:
        print(f"Error serving {filename}: {e}")
        return "File not found", 404

@app.route("/games", methods=["GET"])
def get_games():
    # Convert games data to JSON including all available fields
    games_data = games.to_dict(orient="records")
    return jsonify(games_data)

@app.route("/game_plays/<game_id>", methods=["GET"])
def get_game_plays(game_id):
    # Filter plays for a specific game
    game_plays_all = plays[plays["gameId"] == int(game_id)].copy()
    
    # Filter further to only include plays present in our tracking data
    game_plays_filtered = pd.merge(game_plays_all, valid_plays, on=["gameId", "playId"], how="inner")

    # Prepare the final output
    game_plays_filtered = pd.merge(game_plays_filtered, games[["gameId", "homeTeamAbbr", "visitorTeamAbbr"]], on="gameId")
    game_plays_json = game_plays_filtered.replace({np.nan: None}).sort_values(by="playId").to_dict(orient="records")
    
    print(f"Returning {len(game_plays_json)} plays for game {game_id}")
    return jsonify(game_plays_json)

@app.route("/play", methods=["GET"])
def get_play_data():
    game_id = int(request.args.get("gameId"))
    play_id = int(request.args.get("playId"))
    
    print(f"Fetching data for game {game_id}, play {play_id}")
    
    # Filter data for the specific play
    play_data = tracking[(tracking["gameId"] == game_id) & (tracking["playId"] == play_id)].copy()

    if play_data.empty:
        print(f"No tracking data found for game {game_id}, play {play_id}")
        return jsonify({"error": "No tracking data found", "frames": []}), 404
    
    # Merge necessary play information
    play_info = plays[plays["playId"] == play_id][["gameId", "playId", "possessionTeam", "absoluteYardlineNumber", "yardsToGo"]]
    play_data = pd.merge(play_data, play_info, on=["gameId", "playId"])

    # Assign colors based on possession team
    possession_team = play_data["possessionTeam"].iloc[0] # Get possession team for the play
    
    def assign_color(row):
        club = row["club"]
        if club == "football":
            return "0x8b4513"
        elif club == possession_team:
            return "0xFFFFFF"
        else:
            # Get primary color for the non-possession (defense) team
            return team_colors.get(club, "0x808080") # Default grey if team not found

    play_data["color"] = play_data.apply(assign_color, axis=1)
    
    # Assign z = 0 to players
    play_data.loc[play_data["displayName"] != "football", "z"] = 0
    
    # Calculate ball trajectory
    ball_data = calculate_ball_trajectory(play_data)
    play_data = pd.concat([play_data[play_data["displayName"] != "football"], ball_data])
    
    # Structure data by frame
    frames = []
    for frame_id in sorted(play_data["frameId"].unique()): # Ensure frames are sorted
        frame_data = play_data[play_data["frameId"] == frame_id]
        # Include EPA and rank data for each object
        objects = frame_data[[
            "nflId", "displayName", "x", "y", "z", "color", "club",
            "expected_epa", "decision_rank"
            ]].replace({np.nan: None}).to_dict(orient='records')
        frames.append({"frameId": int(frame_id), "objects": objects})

    # Get line of scrimmage and first down marker
    line_of_scrimmage = play_data["absoluteYardlineNumber"].iloc[0]
    play_direction = play_data["playDirection"].iloc[0]
    yards_to_go = play_data["yardsToGo"].iloc[0]
    if play_direction == "right":
        first_down_marker = line_of_scrimmage + yards_to_go
    else:
        first_down_marker = line_of_scrimmage - yards_to_go

    line_of_scrimmage = -(line_of_scrimmage - 60)
    first_down_marker = -(first_down_marker - 60)

    return jsonify(
        {
            "frames": frames,
            "line_of_scrimmage": int(line_of_scrimmage),
            "first_down_marker": int(first_down_marker)
        }
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)