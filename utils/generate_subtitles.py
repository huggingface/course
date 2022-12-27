import pandas as pd
from tqdm.auto import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from youtubesearchpython import Playlist
from pathlib import Path
import argparse
import sys


def generate_subtitles(language: str, youtube_language_code: str = None):
    metadata = []
    formatter = SRTFormatter()
    path = Path(f"subtitles/{language}")
    path.mkdir(parents=True, exist_ok=True)
    playlist_videos = Playlist.getVideos("https://youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o")

    for idx, video in enumerate(playlist_videos["videos"]):
        video_id = video["id"]
        title = video["title"]
        title_formatted = title.lower().replace(" ", "-").replace(":", "").replace("?", "")
        id_str = f"{idx}".zfill(2)
        srt_filename = f"subtitles/{language}/{id_str}_{title_formatted}.srt"

        # Skip course events
        if "Event Day" in title:
            continue

        # Get transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        english_transcript = transcript_list.find_transcript(language_codes=["en", "en-US"])
        languages = pd.DataFrame(english_transcript.translation_languages)["language_code"].tolist()
        # Map mismatched language codes
        if language not in languages:
            if youtube_language_code is None:
                raise ValueError(
                    f"Language code {language} not found in YouTube's list of supported language: {languages}. Please provide a value for `youtube_language_code` and try again."
                )
            language_code = youtube_language_code
        else:
            language_code = language
        try:
            translated_transcript = english_transcript.translate(language_code)
            translated_transcript = translated_transcript.fetch()
            srt_formatted = formatter.format_transcript(translated_transcript)
            with open(srt_filename, "w", encoding="utf-8") as f:
                f.write(srt_formatted)
        except:
            print(f"Problem generating transcript for {title} with ID {video_id} at {video['link']}.")
            with open(srt_filename, "w", encoding="utf-8") as f:
                f.write("No transcript found for this video!")

        metadata.append({"id": video_id, "title": title, "link": video["link"], "srt_filename": srt_filename})
        break

    df = pd.DataFrame(metadata)
    df.to_csv(f"subtitles/{language}/metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language to generate subtitles for")
    parser.add_argument("--youtube_language_code", type=str, help="YouTube language code")
    args = parser.parse_args()
    generate_subtitles(args.language, args.youtube_language_code)
    print(f"All done! Subtitles stored at subtitles/{args.language}")
