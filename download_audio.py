import csv
import os
import yt_dlp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_audio(video_id, url, output_folder):
    output_file = os.path.join(output_folder, f"{video_id}.mp3")

    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"File already exists for ID: {video_id}. Skipping download.")
        return None

    print(f"Downloading audio for ID: {video_id}, URL: {url}")

    # Set up yt-dlp options for downloading the smallest audio quality
    ydl_opts = {
        'format': 'worstaudio',
        'outtmpl': f'{output_folder}/{video_id}.%(ext)s',  # Save with video ID as filename
        'quiet': False,  # Set to True if you want to suppress download logs
    }

    # Download the audio
    try:
        start_time = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        end_time = time.time()
        download_time = end_time - start_time
        print(f"Downloaded {video_id} in {download_time:.2f} seconds")
        return download_time
    except Exception as e:
        print(f"Error downloading {url} (ID: {video_id}): {e}")
        with open("download_errors.log", "a") as log_file:
            log_file.write(f"ID: {video_id}, URL: {url}, Error: {e}\n")
        return None

def download_audio_from_tsv(tsv_file_path, output_folder):
    download_times = []

    # Open the TSV file
    with open(tsv_file_path, mode='r') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')  # Specify tab as the delimiter

        # Use ThreadPoolExecutor to parallelize downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for row in reader:
                video_id = row.get('id')
                url = row.get('url')

                if not url or not video_id:
                    print(f"Skipping row with missing data: {row}")
                    continue

                futures.append(executor.submit(download_audio, video_id, url, output_folder))

            for future in as_completed(futures):
                download_time = future.result()
                if download_time is not None:
                    download_times.append(download_time)

    if download_times:
        average_time = sum(download_times) / len(download_times)
        print(f"Average download time: {average_time:.2f} seconds")
    else:
        print("No downloads were completed.")

if __name__ == "__main__":
    # Specify the TSV file path and output folder
    tsv_file_path = 'dataset/id_url_mmsr.tsv'  # Replace with your TSV file name
    output_folder = 'downloads'  # Replace with your desired output folder

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Run the downloader
    download_audio_from_tsv(tsv_file_path, output_folder)