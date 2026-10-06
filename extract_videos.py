import os
import csv
import yt_dlp

urls = []

# Extract the first 5 URLs from match.csv
with open('set-1/match.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    next(reader, None)  # Skip the header row
    
    for row in reader:
        if row and row[12].strip():
            urls.append(row[12].strip())
            if len(urls) == 5:
                break

print(urls)

# Define your local Google Drive path
# Windows usually defaults to G:\My Drive\...
# Mac usually defaults to /Users/Username/Google Drive/My Drive/...
gdrive_path = r"G:\My Drive\Match_Downloads" 

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