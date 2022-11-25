# Subtitles for the course videos

This folder contains all the subtitles for the course videos on YouTube.

## How to translate the subtitles

To translate the subtitles, we'll use two nifty libraries that can (a) grab all the YouTube videos associated with the course playlist and (b) translate them on the fly.

To get started, install the following:

```bash
python -m pip install youtube_transcript_api youtube-search-python pandas tqdm
```

Next, run the following script:

```bash
python utils/generate_subtitles.py --language LANG_CODE
```

where `LANG_CODE` is the same language ID used to denote the chosen language the `chapters` folder. If everything goes well, you should end up with a set of translated `.srt` files with timestamps in `subtitles/LANG_CODE` along with some metadata in `metadata.csv`.

Some languages like Simplified Chinese have a different language code to the one used in the course. For these languages, you also need to specify the YouTube language code, e.g.:

```bash
python utils/generate_subtitles.py --language zh-CN --youtube_language_code zh-Hans
```
