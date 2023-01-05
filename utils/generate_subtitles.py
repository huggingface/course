import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from youtubesearchpython import Playlist
from pathlib import Path
import argparse

COURSE_VIDEOS_PLAYLIST = "https://youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o"
TASK_VIDEOS_PLAYLIST = "https://youtube.com/playlist?list=PLo2EIpI_JMQtyEr-sLJSy5_SnLCb4vtQf"
# These videos are not part of the course, but are part of the task playlist
TASK_VIDEOS_TO_SKIP = ["tjAIM7BOYhw", "WdAeKSOpxhw", "KWwzcmG98Ds", "TksaY_FDgnk", "leNG9fN9FQU", "dKE8SIt9C-w"]


def generate_subtitles(language: str, youtube_language_code: str = None, is_task_playlist: bool = False):
    metadata = []
    formatter = SRTFormatter()
    path = Path(f"subtitles/{language}")
    path.mkdir(parents=True, exist_ok=True)
    if is_task_playlist:
        playlist_videos = Playlist.getVideos(TASK_VIDEOS_PLAYLIST)
    else:
        playlist_videos = Playlist.getVideos(COURSE_VIDEOS_PLAYLIST)

    for idx, video in enumerate(playlist_videos["videos"]):
        video_id = video["id"]
        title = video["title"]
        title_formatted = title.lower().replace(" ", "-").replace(":", "").replace("?", "")
        id_str = f"{idx}".zfill(2)

        if is_task_playlist:
            srt_filename = f"{path}/tasks_{id_str}_{title_formatted}.srt"
        else:
            srt_filename = f"{path}/{id_str}_{title_formatted}.srt"

        # Skip course events
        if "Event Day" in title:
            continue

        # Skip task videos that don't belong to the course
        if video_id in TASK_VIDEOS_TO_SKIP:
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

    df = pd.DataFrame(metadata)

    if is_task_playlist:
        df.to_csv(f"{path}/metadata_tasks.csv", index=False)
    else:
        df.to_csv(f"{path}/metadata.csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Language to generate subtitles for")
    parser.add_argument("--youtube_language_code", type=str, help="YouTube language code")
    args = parser.parse_args()
    generate_subtitles(args.language, args.youtube_language_code, is_task_playlist=False)
    generate_subtitles(args.language, args.youtube_language_code, is_task_playlist=True)
    print(f"All done! Subtitles stored at subtitles/{args.language}")
