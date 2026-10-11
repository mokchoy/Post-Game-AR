"""Download set-1 matches under their annotation-folder names, reusing legacy files."""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


PROJECT = Path(__file__).resolve().parent
MATCH_CSV = PROJECT / "set-1" / "match.csv"
DEFAULT_VIDEOS_DIR = Path(r"G:\My Drive\Match_Downloads_3")

# Verified set-1 aliases; the shared 2019 title belongs to ID 1, never ID 3.
LEGACY_FILENAMES = {
    1: "F ｜ MS ｜ Kento MOMOTA (JPN) [1] vs. CHOU Tien Chen (TPE) [2] ｜ BWF 2019.mp4",
    2: "Group B ｜ MS ｜ CHOU Tien Chen (TPE) vs. CHEN Long (CHN) ｜ BWF 2019.mp4",
    4: "QF ｜ MS ｜ CHEN Long (CHN) [5] vs. CHOU Tien Chen (TPE) [2] ｜ BWF 2019.mp4",
    5: "F ｜ MS ｜ Kento MOMOTA (JPN) [1] vs CHOU Tien Chen (TPE) [4] ｜ BWF 2018.mp4",
    6: "F ｜ MS ｜ CHOU Tien Chen (TPE) [4] vs Kento MOMOTA (JPN) [2] ｜ BWF 2018.mp4",
    7: "MS ｜ CHOU Tien Chen (TPE) [6] vs Kento MOMOTA (JPN) ｜ BWF 2018.mp4",
    8: "SF ｜ MS ｜ Anders ANTONSEN (DEN) [4] vs CHOU Tien Chen (TPE) [2] ｜ BWF 2019.mp4",
    9: "QF ｜ MS ｜ CHOU Tien Chien (TPE) vs Jonatan CHRISTIE (INA) ｜ BWF 2019.mp4",
    10: "G1 ｜ MS ｜ CHOU Tien Chen (TPE) vs NG Ka Long Angus (HKG) ｜ BWF 2019.mp4",
    11: "QF ｜ MS ｜ CHOU Tien Chen (TPE) [4] vs. Jonatan CHRISTIE (INA) [6] ｜ BWF 2019.mp4",
    13: "F ｜ MS ｜ Kento MOMOTA (JPN) [1] vs. Viktor AXELSEN (DEN) [5] ｜ BWF 2020.mp4",
    14: "QF ｜ MS ｜ Jonatan CHRISTIE (INA) [6] vs. Anders ANTONSEN (DEN) [4] ｜ BWF 2020.mp4",
    15: "F ｜ MS ｜ Anthony Sinisuka GINTING (INA) [7] vs. Anders ANTONSEN (DEN) [4] ｜ BWF 2020.mp4",
    16: "SF ｜ MS ｜ Viktor AXELSEN (DEN) [5] vs. Anthony Sinisuka GINTING (INA) [7] ｜ BWF 2020.mp4",
    17: "QF ｜ MS ｜ Jonatan CHRISTIE (INA) [6] vs. NG Ka Long Angus (HKG) ｜ BWF 2020.mp4",
    18: "QF ｜ MS ｜ SHI Yu Qi (CHN) [7] vs. Viktor AXELSEN (DEN) [2] ｜ BWF 2020.mp4",
    19: "QF ｜ MS ｜ Viktor AXELSEN (DEN) [5] vs. CHEN Long (CHN) [4] ｜ BWF 2020.mp4",
    20: "SF ｜ MS ｜ Viktor AXELSEN (DEN) [5] vs. NG Ka Long Angus (HKG) ｜ BWF 2020.mp4",
    21: "YONEX Thailand Open ｜ Day 4： An Se Young (KOR) [7] vs. Ratchanok Intanon (THA) [4].mp4",
    22: "YONEX Thailand Open ｜ Day 4： Busanan Ongbamrungphan (THA) vs. Mia Blichfeldt (DEN).mp4",
    23: "YONEX Thailand Open ｜ Day 4： Ng Ka Long Angus (HKG) [7] vs. Lee Cheuk Yiu (HKG).mp4",
    24: "YONEX Thailand Open ｜ Day 4： Rasmus Gemke (DEN) vs. Anthony Sinisuka Ginting (INA) [5].mp4",
    25: "YONEX Thailand Open ｜ Day 4： Carolina Marin (ESP) [5] vs. Supanida Katethong (THA).mp4",
    26: "YONEX Thailand Open ｜ Day 4： Viktor Axelsen (DEN) [4] vs. Jonatan Christie (INA) [6].mp4",
    27: "YONEX Thailand Open ｜ Day 5： Anthony Sinisuka Ginting (INA) [5] vs. Viktor Axelsen (DEN) [4].mp4",
    28: "YONEX Thailand Open ｜ Day 6： Ng Ka Long Angus (HKG) [7] vs. Viktor Axelsen (DEN) [4].mp4",
    29: "TOYOTA Thailand Open ｜ Day 4： An Se Young (KOR) [7] vs. Pornpawee Chochuwong (THA).mp4",
    30: "TOYOTA Thailand Open ｜ Day 4： Anders Antonsen (DEN) [3] vs. Sameer Verma (IND).mp4",
    31: "TOYOTA Thailand Open ｜ Day 4： Carolina Marin (ESP) [5] vs. Neslihan Yigit (TUR).mp4",
    32: "TOYOTA Thailand Open ｜ Day 4： Hans-Kristian Solberg Vittinghus (DEN) vs. Lee Cheuk Yu (HKG).mp4",
    33: "TOYOTA Thailand Open ｜ Day 4： Liew Daren (MAS) vs. Viktor Axelsen (DEN) [4].mp4",
    34: "TOYOTA Thailand Open ｜ Day 4： Ratchanok Intanon (THA) [4] vs. Pusarla V. Sindhu (IND) [6].mp4",
    35: "TOYOTA Thailand Open ｜ Day 5： Carolina Marin (ESP) [5] vs. An Se Young (KOR) [7].mp4",
    36: "TOYOTA Thailand Open ｜ Day 5： Hans-Kristian Solberg Vittinghus (DEN) vs. Anders Antonsen (DEN) [3].mp4",
    37: "TOYOTA Thailand Open ｜ Day 6： Hans-Kristian Solberg VIittinghus (DEN) vs. Viktor Axelsen (DEN) [4].mp4",
    38: "HSBC BWF World Tour Finals ｜ Day 3： Carolina Marin (ESP) [1] vs. An Se Young (KOR) [4].mp4",
    39: "HSBC BWF World Tour Finals ｜ Day 3： Lee Zii Jia (MAS) vs. Anthony Sinisuka Ginting (INA).mp4",
    40: "HSBC BWF World Tour Finals ｜ Day 3： Michelle Li (CAN) vs. Evgeniya Kosetskaya (RUS).mp4",
    41: "HSBC BWF World Tour Finals ｜ Day 3： Ng Ka Long Angus (HKG) vs. Kidambi Srikanth (IND).mp4",
    42: "HSBC BWF World Tour Finals ｜ Day 3： Pornpawee Chochuwong (THA) vs. Pusarla V. Sindhu (IND).mp4",
    43: "HSBC BWF World Tour Finals ｜ Day 4： Carolina Marin (ESP) [1] vs. Pornpawee Chochuwong (THA).mp4",
    44: "HSBC BWF World Tour Finals ｜ Day 5： Viktor Axelsen (DEN) vs. Anders Antonsen (DEN).mp4",
}


def read_matches(path=MATCH_CSV):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "video", "url"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing id, video, or url column")
        rows = list(reader)
    ids, names = set(), set()
    for row in rows:
        row["id"] = int(row["id"])
        name = row["video"]
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", name or "")
                or name.endswith((" ", "."))
                or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name, re.IGNORECASE)):
            raise ValueError(f"{path}: unsafe annotation-folder name {name!r}")
        if row["id"] <= 0 or row["id"] in ids or name.casefold() in names:
            raise ValueError(f"{path}: duplicate/invalid match ID or filename: {row['id']} {name!r}")
        row["url"] = (row.get("url") or "").strip()
        address = urlparse(row["url"])
        if row["url"] and (address.scheme not in ("http", "https") or not address.netloc):
            raise ValueError(f"{path}: invalid URL for ID {row['id']}")
        ids.add(row["id"])
        names.add(name.casefold())
    if not rows:
        raise ValueError(f"{path}: no matches found")
    return rows


def video_names(row):
    canonical = f"{row['video']}.mp4"
    legacy = LEGACY_FILENAMES.get(row["id"])
    return (canonical, legacy) if legacy else (canonical,)


def resolve_videos(matches, videos):
    lookup = {name.casefold(): path for name, path in videos.items()}
    matched, known = {}, set()
    for row in matches:
        candidates = video_names(row)
        known.update(name.casefold() for name in candidates)
        for name in candidates:
            if name.casefold() in lookup:
                matched[row["id"]] = lookup[name.casefold()]
                break
    unknown = sorted((name for name in videos if name.casefold() not in known), key=str.casefold)
    return matched, unknown


def download_matches(matches, videos_dir, check=False):
    root = Path(videos_dir).expanduser()
    videos = {p.name: p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"} if root.exists() else {}
    matched, unknown = resolve_videos(matches, videos)
    for name in unknown:
        print(f"UNRESOLVED EXISTING MP4 (preserved): {name}")
    reused = pending = missing_url = failed = 0
    for row in matches:
        match_id = row["id"]
        if match_id in matched:
            print(f"REUSE: ID {match_id}: {matched[match_id].name}")
            reused += 1
            continue
        if not row["url"]:
            print(f"MISSING URL: ID {match_id}: {row['video']}")
            missing_url += 1
            continue
        pending += 1
        output = root / f"{row['video']}.mp4"
        print(f"{'WOULD DOWNLOAD' if check else 'DOWNLOAD'}: ID {match_id}: {output}")
        if check:
            continue
        import yt_dlp
        options = {
            "retries": 3,
            "retry_sleep_functions": {"http": lambda n: 5},
            "format": "bestvideo+bestaudio/best",
            "outtmpl": os.path.join(str(root), f"{row['video']}.%(ext)s"),
            "merge_output_format": "mp4",
            "overwrites": False,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                status = downloader.download([row["url"]])
            if status:
                failed += 1
        except yt_dlp.utils.DownloadError as exc:
            print(f"DOWNLOAD FAILED: ID {match_id}: {exc}", file=sys.stderr)
            failed += 1
    print(f"Totals: roster={len(matches)}, reused={reused}, {'pending' if check else 'attempted'}={pending}, missing URL={missing_url}, failed={failed}, unresolved existing={len(unknown)}")
    return 2 if missing_url or failed or unknown else 0


def self_test():
    import contextlib
    import io
    import tempfile
    from unittest.mock import MagicMock, patch

    rows = read_matches()
    assert len(LEGACY_FILENAMES) == 42 and 3 not in LEGACY_FILENAMES and 12 not in LEGACY_FILENAMES
    assert len({name.casefold() for name in LEGACY_FILENAMES.values()}) == len(LEGACY_FILENAMES)
    selected = [row for row in rows if row["id"] in (1, 3, 12, 14, 39, 44)]
    with tempfile.TemporaryDirectory(prefix=".download-self-test-", dir=PROJECT) as temp:
        root = Path(temp)
        for match_id in (1, 14, 39, 44):
            (root / LEGACY_FILENAMES[match_id]).touch()
        with patch("yt_dlp.YoutubeDL") as mocked, contextlib.redirect_stdout(io.StringIO()) as output:
            status = download_matches(selected, root, check=True)
        assert status == 2 and not mocked.called
        assert "reused=4, pending=1, missing URL=1" in output.getvalue()
        assert "WOULD DOWNLOAD: ID 3:" in output.getvalue()
        assert "WOULD DOWNLOAD: ID 1:" not in output.getvalue()
        fake = MagicMock()
        fake.__enter__.return_value.download.return_value = 0
        with patch("yt_dlp.YoutubeDL", return_value=fake) as mocked, contextlib.redirect_stdout(io.StringIO()):
            status = download_matches(selected, root)
        assert status == 2 and mocked.call_count == 1
        korea = next(row for row in selected if row["id"] == 3)
        assert mocked.call_args.args[0]["outtmpl"] == str(root / f"{korea['video']}.%(ext)s")
        assert mocked.call_args.args[0]["overwrites"] is False
        fake.__enter__.return_value.download.assert_called_once_with([korea["url"]])
        canonical = root / f"{rows[0]['video']}.mp4"
        canonical.touch()
        inventory = {p.name: p for p in root.iterdir()}
        matched, unknown = resolve_videos(rows, inventory)
        assert matched[1] == canonical and 3 not in matched and not unknown
        missing = [row for row in rows if row["id"] in (2, 3)]
        fake.__enter__.return_value.download.side_effect = [2, 0]
        with patch("yt_dlp.YoutubeDL", return_value=fake) as mocked, contextlib.redirect_stdout(io.StringIO()) as output:
            status = download_matches(missing, root)
        assert status == 2 and mocked.call_count == 2 and "failed=1" in output.getvalue()
        bad_csv = root / "bad.csv"
        for name in ("../escape", r"..\escape", "CON", "trailing "):
            with bad_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("id", "video", "url"))
                writer.writerow(("1", name, ""))
            try:
                read_matches(bad_csv)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unsafe filename accepted: {name!r}")
        for duplicate in (("1", "Another_match", ""), ("2", "Valid_match", "")):
            with bad_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("id", "video", "url"))
                writer.writerow(("1", "Valid_match", ""))
                writer.writerow(duplicate)
            try:
                read_matches(bad_csv)
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate match ID or filename accepted")
    print("Self-test passed: canonical names, verified legacy reuse, ID 1/3 collision, missing URL, read-only checks, exact download arguments, and unsafe-path rejection.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only inventory; no network requests or downloads")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    return download_matches(read_matches(), args.videos_dir, check=args.check)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
