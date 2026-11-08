"""Step 50: build DataToProject/games.json (one entry per cropped rally: videos, score, players) and DataToProject/Info/flags/*.png.
Scores come from the ShuttleSet set CSV (rally N, last row = score after the rally; before = after minus the point just won).
Player A is the first-named player (checked: A wins set 2 of every match here). Height/weight/rank are from web search
(Wikipedia; see PLAYERS), rank = career-high BWF men's singles ranking. Flags are drawn here, no downloads.
Run from Integration/ar: ../.venv/bin/python scripts/build_game_info.py"""
import csv, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]            # Post-Game-AR
DATA = ROOT/"Integration"/"DataToProject"
SET1 = ROOT/"set-1"

PLAYERS = {   # name: (country, flag file, height m, weight kg, rank text); sources in the Step 50 log
    "Kento MOMOTA": ("JPN", 1.75, 68, "No. 1 (peak)"),
    "CHOU Tien Chen": ("TPE", 1.80, 78, "No. 2 (peak)"),
    "CHEN Long": ("CHN", 1.87, 75, "No. 1 (peak)"),
    "Anders ANTONSEN": ("DEN", 1.85, 80, "No. 2 (peak)"),
}
# (rally file, match folder, A, B, label); A/B = first/second named player in the folder name
CLIPS = [("rally_002", "Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final", "Kento MOMOTA", "CHOU Tien Chen", "Korea Open 2019 Final"),
         ("rally_010", "Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final", "Kento MOMOTA", "CHOU Tien Chen", "Korea Open 2019 Final"),
         ("rally_028", "CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal", "CHEN Long", "CHOU Tien Chen", "Denmark Open 2019 QF"),
         ("rally_031", "CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal", "CHEN Long", "CHOU Tien Chen", "Denmark Open 2019 QF"),
         ("rally_015", "CHOU_Tien_Chen_Anders_ANTONSEN_Fuzhou_Open_2019_Semi-finals", "CHOU Tien Chen", "Anders ANTONSEN", "Fuzhou Open 2019 SF"),
         ("rally_017", "CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage", "CHEN Long", "CHOU Tien Chen", "World Tour Finals 2019"),
         ("rally_029", "CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage", "CHEN Long", "CHOU Tien Chen", "World Tour Finals 2019"),  # 329 frames = this one, not Fuzhou 029 (376)
         ("rally_016", "Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals", "Kento MOMOTA", "CHOU Tien Chen", "Malaysia Open 2018 QF"),
         ("rally_026", "Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals", "Kento MOMOTA", "CHOU Tien Chen", "Malaysia Open 2018 QF")]

W, H = 300, 200
RED, WHITE, BLUE, YEL = (48, 12, 200), (255, 255, 255), (150, 40, 0), (0, 222, 255)   # BGR


def star(img, c, r, rot_deg, color):
    pts = [(c[0]+r*(1 if i % 2 == 0 else .382)*np.sin(np.deg2rad(rot_deg+36*i)), c[1]-r*(1 if i % 2 == 0 else .382)*np.cos(np.deg2rad(rot_deg+36*i))) for i in range(10)]
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA)


def flag(code):
    img = np.full((H, W, 3), 255, np.uint8)
    if code == "JPN":
        cv2.circle(img, (W//2, H//2), int(H*.3), (45, 0, 188), -1, cv2.LINE_AA)
    elif code == "DEN":
        img[:] = (60, 0, 198)
        cv2.rectangle(img, (int(W*12/37), 0), (int(W*16/37), H), WHITE, -1)
        cv2.rectangle(img, (0, int(H*12/28)), (W, int(H*16/28)), WHITE, -1)
    elif code == "CHN":
        img[:] = (40, 35, 222)
        u = W/30
        star(img, (5*u, 5*u), 3*u, 0, YEL)
        for cx, cy in ((10, 2), (12, 4), (12, 7), (10, 9)):
            star(img, (cx*u, cy*u), u, np.degrees(np.arctan2(cx-5, 5-cy)) + 0, YEL)
    elif code == "TPE":                                   # ROC national flag (the Chinese Taipei Olympic flag is a different design)
        img[:] = (40, 35, 222)
        cv2.rectangle(img, (0, 0), (W//2, H//2), (140, 40, 0), -1)
        c = (W//4, H//4)
        for i in range(12):
            a = np.deg2rad(30*i)
            tri = [(c[0]+H*.24*np.sin(a), c[1]-H*.24*np.cos(a)),
                   (c[0]+H*.07*np.sin(a-.26), c[1]-H*.07*np.cos(a-.26)), (c[0]+H*.07*np.sin(a+.26), c[1]-H*.07*np.cos(a+.26))]
            cv2.fillPoly(img, [np.round(tri).astype(np.int32)], WHITE, cv2.LINE_AA)
        cv2.circle(img, c, int(H*.085), (140, 40, 0), -1, cv2.LINE_AA)
        cv2.circle(img, c, int(H*.07), WHITE, -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W-1, H-1), (90, 90, 90), 1)
    return img


def scores(match, rally):
    rows = [r for r in csv.DictReader(open(SET1/match/"set1.csv", encoding="utf-8-sig")) if int(r["rally"]) == rally]
    last = rows[-1]
    a, b = int(last["roundscore_A"]), int(last["roundscore_B"])
    win = last["getpoint_player"]
    return (a-(win == "A"), b-(win == "B")), (a, b), len(rows)


def main():
    (DATA/"Info"/"flags").mkdir(parents=True, exist_ok=True)
    for code in {v[0] for v in PLAYERS.values()}:
        cv2.imwrite(str(DATA/"Info"/"flags"/f"{code}.png"), flag(code))
    games = []
    for rally, match, a, b, label in CLIPS:
        before, after, hits = scores(match, int(rally[-3:]))
        players = []
        for name, i in ((a, 0), (b, 1)):
            code, h, w, rank = PLAYERS[name]
            players.append({"name": name, "flag": f"Info/flags/{code}.png", "height": f"{h:.2f} m", "weight": f"{w} kg", "rank": rank,
                            "score_before": before[i], "score_after": after[i]})
        games.append({"name": f"{label} {rally.replace('_', ' ')}", "video": f"GameRally/{rally}.mp4", "topdown": f"TopDownDerived/{rally}.mp4",
                      "match": match, "rally": rally, "annotated_hits": hits, "players": players})
    (DATA/"games.json").write_text(json.dumps(games, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    for g in games:
        print(g["name"], [(p["name"], p["score_before"], p["score_after"]) for p in g["players"]], g["annotated_hits"])


if __name__ == "__main__":
    main()
