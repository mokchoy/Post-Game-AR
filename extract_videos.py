import os
import csv
import yt_dlp

urls = []
seen = set()

for csv_path in ("set-1/match.csv", "set-2/match.csv"):
    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if "url" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path}: missing 'url' column")

        for row in reader:
            url = (row.get("url") or "").strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)

print(f"Found {len(urls)} unique video links")

# Define your local Google Drive path
gdrive_path = r"G:\My Drive\Match_Downloads_2" 

# Configure download settings
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    # Join the GDrive path with the filename template
    'outtmpl': os.path.join(gdrive_path, '%(title)s.%(ext)s'), 
    'merge_output_format': 'mp4'
}

# Execute the download
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    print(f"Downloading {len(urls)} videos directly to Google Drive...")
    ydl.download(urls)