import pandas as pd
import yt_dlp

def fetch_youtube_url(video_title):
    # Convert underscores to spaces for a better search query
    clean_title = video_title.replace('_', ' ')
    search_query = f"ytsearch1:{clean_title}"
    
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)
            if 'entries' in info and len(info['entries']) > 0:
                video_url = info['entries'][0].get('url')
                # Format to a standard watch URL if it only returns the ID
                if video_url and not video_url.startswith('http'):
                    return f"https://www.youtube.com/watch?v={video_url}"
                return video_url
        except Exception as e:
            print(f"Failed to fetch {video_title}: {e}")
    return None

def update_match_urls(file_path="match.csv"):
    df = pd.read_csv(file_path)
    
    # Ensure the 'url' column exists and is cast to an object type to prevent float64 TypeError
    if 'url' not in df.columns:
        df['url'] = None
    df['url'] = df['url'].astype(object)
    
    print(f"Searching YouTube for {len(df)} matches...")
    for idx, row in df.iterrows():
        # Only search if the url column is empty/NaN
        if pd.isna(row['url']) or str(row['url']).strip() == "":
            print(f"Searching: {row['video']}")
            url = fetch_youtube_url(row['video'])
            if url:
                df.at[idx, 'url'] = url
                print(f"Found: {url}")
            else:
                print("No result found.")
                
    df.to_csv(file_path, index=False)
    print(f"Successfully updated {file_path}.")

if __name__ == "__main__":
    update_match_urls(r"set-2\match.csv")