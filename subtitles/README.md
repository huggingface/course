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

# How to convert bilingual subtitle to monolingual subtitle

# Logic

The english caption line is conventionally placed at the last line of each subtitle block in srt files. So removing the last line of each subtitle block would make the bilingual subtitle a monolingual subtitle. 

# Usage
> python3 convert_bilingual_monolingual.py -i \<input_file\> -o \<output_file\>

**Example**
* For instance, the input file name is "test.cn.en.srt", and you name your output file as "output_test.cn.srt" *
> python3 convert_bilingual_monolingual.py -i test.cn.en.srt -o output_test.cn.srt