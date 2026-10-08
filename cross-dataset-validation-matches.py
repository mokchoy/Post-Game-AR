import csv
import math
import re
from pathlib import Path

ROOT = Path(r"C:\Users\ILLEGEAR\Documents\Post-Game-AR")

# Known overlaps from the published list:
# https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
# TrackNet training match number -> ShuttleSet folder, published rally names.
PUBLISHED = {
    4: (
        "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals",
        "1_03_02 1_15_10 2_02_05 2_05_07 2_14_17 "
        "3_02_00 3_07_05 3_18_17",
    ),
    5: (
        "Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final",
        "1_01_01 1_01_02 1_19_18 1_21_19 2_15_13 2_20_17",
    ),
    13: (
        "CHOU_Tien_Chen_NG_Ka_Long_Angus_Sudirman_Cup_2019_Group_Stage",
        "1_09_10 1_17_15 2_06_05 2_07_05 2_09_08 2_17_11",
    ),
    20: (
        "NG_Ka_Long_Angus_SHI_Yu_Qi_Thailand_Masters_2020_SemiFinals",
        "1_09_05 1_11_10 2_00_01 2_05_08 2_07_08 2_19_14",
    ),
    21: (
        "Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals",
        "1_02_01 1_16_17 1_19_19 2_02_03 2_04_04 2_09_08 2_12_08",
    ),
}

EXPECTED_REJECTED = {(5, "1_01_02"), (5, "1_19_18")}

COORDS = (
    "player_location_x", "player_location_y",
    "opponent_location_x", "opponent_location_y",
    "landing_x", "landing_y",
)
REQUIRED = (
    *COORDS, "rally", "ball_round", "frame_num",
    "roundscore_A", "roundscore_B", "player", "flaw",
)


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def integer(value):
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"invalid nonnegative integer: {value!r}")
    return int(number)


def score(row):
    return integer(row["roundscore_A"]), integer(row["roundscore_B"])


def check_rows(rows):
    """Check annotation completeness; return separate frame-order warnings."""
    if not rows or len({integer(row["rally"]) for row in rows}) != 1:
        raise ValueError("missing or ambiguous rally")
    if len({score(row) for row in rows}) != 1:
        raise ValueError("inconsistent scores within rally")

    for index, row in enumerate(rows, 1):
        if (row["flaw"] or "").strip().lower() not in ("", "0", "0.0", "false"):
            raise ValueError(f"stroke {index}: flaw={row['flaw']!r}")
        if row["player"] not in ("A", "B"):
            raise ValueError(f"stroke {index}: invalid player")

        # Finite coordinates establish completeness, not spatial accuracy.
        for field in COORDS:
            try:
                valid = math.isfinite(float(row[field]))
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise ValueError(
                    f"stroke {index}: invalid {field}={row[field]!r}"
                )

    shots = [integer(row["ball_round"]) for row in rows]
    if shots != list(range(1, len(rows) + 1)):
        raise ValueError("missing, duplicated, or out-of-order strokes")

    frames = [integer(row["frame_num"]) for row in rows]
    return [
        (i, a, b)
        for i, (a, b) in enumerate(zip(frames, frames[1:]), 1)
        if a >= b
    ]


def self_check():
    row = dict.fromkeys(COORDS, "0")  # Zero is a valid image coordinate.
    row.update(
        rally="1", ball_round="1", frame_num="10",
        roundscore_A="1", roundscore_B="0", player="A", flaw="",
    )
    assert check_rows([row]) == []
    assert check_rows([
        row, dict(row, ball_round="2", frame_num="9")
    ]) == [(1, 10, 9)]

    for bad in (
        [],
        [dict(row, landing_x="nan")],
        [dict(row, flaw="1")],
        [dict(row, ball_round="2")],
    ):
        try:
            check_rows(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted invalid data")


def main():
    self_check()

    sources = {
        root: read_csv(root / "match.csv", ("id", "video", "winner", "loser"))
        for root in (ROOT / "set-1", ROOT / "set-2")
    }
    passed, rejected, timing_warnings = set(), set(), set()
    stroke_total = 0

    for tracknet_id, (folder, rally_names) in PUBLISHED.items():
        candidates = [
            (root, match)
            for root, matches in sources.items()
            for match in matches
            if normalize(match["video"]) == normalize(folder)
        ]

        for rally_name in rally_names.split():
            key = (tracknet_id, rally_name)
            set_number, winner_score, loser_score = map(
                int, rally_name.split("_")
            )
            errors, counts = [], []

            if not candidates:
                errors.append("match absent from both match.csv files")

            for root, match in candidates:
                label = f"{root.name}/match ID {match['id']}"
                try:
                    if sum(r == root for r, _ in candidates) != 1:
                        raise ValueError("ambiguous match metadata")

                    winner = normalize(match["winner"])
                    loser = normalize(match["loser"])
                    if (
                        not winner or not loser
                        or not normalize(folder).startswith(winner + loser)
                    ):
                        raise ValueError("winner/loser metadata mismatch")

                    path = root / match["video"] / f"set{set_number}.csv"
                    rows = read_csv(path, REQUIRED)

                    # HackMD names use POST-rally winner/loser scores.
                    target_score = (winner_score, loser_score)
                    ids = {
                        row["rally"] for row in rows
                        if score(row) == target_score
                    }
                    if len(ids) != 1:
                        raise ValueError(
                            f"score {target_score}: found {len(ids)} rallies"
                        )

                    # Validate EVERY row of the identified rally.
                    selected = [row for row in rows if row["rally"] in ids]
                    warnings = check_rows(selected)
                    if warnings:
                        timing_warnings.add(key)
                        print(
                            f"WARN   match{tracknet_id}/{rally_name}: {label}, "
                            f"backward/repeated frames {warnings}"
                        )

                    counts.append(len(selected))
                    print(
                        f"PASS   match{tracknet_id}/{rally_name}: {label}, "
                        f"rally {selected[0]['rally']}, "
                        f"{len(selected)} strokes"
                    )

                except (
                    OSError, ValueError, TypeError, KeyError, csv.Error
                ) as error:
                    errors.append(f"{label}: {error}")

            if errors:
                rejected.add(key)
                print(
                    f"REJECT match{tracknet_id}/{rally_name}: "
                    f"{'; '.join(errors)}"
                )
            else:
                passed.add(key)
                stroke_total += counts[0]  # Count duplicate copies once.

    print(
        f"\n{len(passed)}/33 unique rallies passed; "
        f"{len(rejected)} rejected; {stroke_total} stroke records."
    )
    print(
        f"Frame-order warnings: "
        f"{len(passed & timing_warnings)} accepted rallies."
    )
    print("Trajectory CSV contents and video/frame alignment: NOT CHECKED.")

    expected_passed = {
        (match_id, name)
        for match_id, (_, names) in PUBLISHED.items()
        for name in names.split()
    } - EXPECTED_REJECTED

    confirmed = (
        passed == expected_passed and rejected == EXPECTED_REJECTED
    )
    print(
        "31-rally annotation baseline:",
        "CONFIRMED" if confirmed else "CHANGED / FAILED",
    )
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())