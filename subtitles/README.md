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

where `LANG_CODE` is the same language ID used to denote the chosen language the `chapters` folder. If everything goes well, you should end up with a set of translated `.srt` files with timestamps in the `subtitles/LANG_CODE` folder along with some metadata in `metadata.csv`.

Some languages like Simplified Chinese have a different YouTube language code (`zh-Hans`) to the one used in the course (`zh-CN`). For these languages, you also need to specify the YouTube language code, e.g.:

```bash
python utils/generate_subtitles.py --language zh-CN --youtube_language_code zh-Hans
```

Once you have the `.srt` files you can manually fix any translation errors and then open a pull request with the new files.

# Convert bilingual subtitles to monolingual subtitles

In some SRT files, the English caption line is conventionally placed at the last line of each subtitle block to enable easier comparison when correcting the machine translation.

For example, in the `zh-CN` subtitles, each block has the following format:

```
1
00:00:05,850 --> 00:00:07,713
- 欢迎来到 Hugging Face 课程。
- Welcome to the Hugging Face Course.
```

To upload the SRT file to YouTube, we need the subtitle in monolingual format, i.e. the above block should read:

```
1
00:00:05,850 --> 00:00:07,713
- 欢迎来到 Hugging Face 课程。
```

To handle this, we provide a script that converts the bilingual SRT files to monolingual ones. To perform the conversion, run:

```bash
python utils/convert_bilingual_monolingual.py --input_language_folder subtitles/LANG_ID --output_language_folder tmp-subtitles
```