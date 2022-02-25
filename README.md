# The Hugging Face Course

This repo contains the content that's used to create the **[Hugging Face course](https://huggingface.co/course/chapter1/1)**. The course teaches you about applying Transformers to various tasks in natural language processing and beyond. Along the way, you'll learn how to use the [Hugging Face](https://huggingface.co/) ecosystem — [🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Tokenizers](https://github.com/huggingface/tokenizers), and [🤗 Accelerate](https://github.com/huggingface/accelerate) — as well as the [Hugging Face Hub](https://huggingface.co/models). It's completely free and open-source!

## 🌎 Languages and translations

| Language                                               | Source        | Authors                                                                                                                                                                                                                                                                                                        |
|:-------------------------------------------------------|:--------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [English](https://huggingface.co/course/en/chapter1/1) | [`chapters/en`](https://github.com/huggingface/course/tree/master/chapters/en) | [@sgugger](https://github.com/sgugger), [@lewtun](https://github.com/lewtun), [@LysandreJik](https://github.com/LysandreJik), [@Rocketknight1](https://github.com/Rocketknight1), [@sashavor](https://github.com/sashavor), [@osanseviero](https://github.com/osanseviero), [@SaulLu](https://github.com/SaulLu), [@lvwerra](https://github.com/lvwerra) |

### Translating the course into your language

As part of our mission to democratise machine learning, we'd love to have the course available in many more languages! Please follow the steps below if you'd like to help translate the course into your language 🙏.

**🍴 Fork the repository**

To get started, you'll first need to [fork this repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo). You can do this by clicking on the **Fork** button on the top-right corner of this repo's page.

Once you've forked the repo, you'll want to get the files on your local machine for editing. You can do that by cloning the fork with Git as follows:

```bash
git clone https://github.com/YOUR-USERNAME/course
```

**📋 Copy-paste the English version with a new language code**

The course files are organised into two main directories:

* [`chapters`](https://github.com/huggingface/course/tree/master/chapters): all the text and code snippets associated with the course.
* [`static`](https://github.com/huggingface/course/tree/master/static): all the images and GIFs for the course website.

You'll only need to copy the files in the [`chapters/en`](https://github.com/huggingface/course/tree/master/chapters/en) directory, so first navigate to your fork of the repo and run the following:

```bash
cd ~/path/to/course
cp -r chapters/en chapters/LANG-ID
```

Here, `LANG-ID` should be one of the ISO 639-1 or ISO 639-2 language codes -- see [here](https://www.loc.gov/standards/iso639-2/php/code_list.php) for a handy table.

**✍️ Start translating**

Now comes the fun part - translating the text! The first thing we recommend is translating the `_chapters.yml` file. This file is used to render the table of contents on the website and provide the links to the Colab notebooks. The only fields you should change are `title`, `subtitle`, and `sections`. For example, here are the parts of `_chapters.yml` that we'd translate for [Chapter 0](https://huggingface.co/course/chapter0/1?fw=pt):

```yaml
- local: chapter0
  title: Setup  # Translate this!
  subtitle: This course looks cool, how can I run its code?  # Translate this!
  sections:
  - Setting up a working environment  # Translate this!
```

Once you have translated the `_chapters.yml` file, you can start translating the [MDX](https://mdxjs.com/) files associated with each chapter. We recommend opening a pull request once you've translated Chapter 0 so that we can test that the website renders correctly.

> 🙋 If you'd like others to help you with the translation, you can either [open an issue](https://github.com/huggingface/course/issues), post in our [forums](https://discuss.huggingface.co/c/course/20), or tag @_lewtun on Twitter to gain some visibility.

## 📔 Jupyter notebooks

The Jupyter notebooks containing all the code from the course are hosted on the [`huggingface/notebooks`](https://github.com/huggingface/notebooks) repo. If you wish to generate them locally, first install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run the following script:

```bash
python utils/generate_notebooks.py --output_dir nbs
```

This script extracts all the code snippets from the English chapters and stores them as notebooks in the `nbs` folder (which is ignored by Git by default).

## ✍️ Contributing a new chapter

> Note: we are not currently accepting community contributions for new chapters. These instructions are for the Hugging Face authors.

Adding a new chapter to the course is quite simple:

1. Create a new directory under `chapters/en/chapterX`, where `chapterX` is the chapter you'd like to add.
2. Add numbered MDX files `sectionX.mdx` for each section. If you need to include images, place them in the `static` directory and use the [HTML Images Syntax](https://www.w3schools.com/html/html_images.asp) with the path `/course/static/chapterX/your-image.png`.
3. Update the `_chapters.yml` file to include your chapter sections -- this information will render the table of contents on the website. If your section involves both the PyTorch and TensorFlow APIs of `transformers`, make sure you include links to both Colabs in the `colab` field.

If you get stuck, check out one of the existing chapters -- this will often show you the expected syntax.

Once you are happy with the content, open a pull request and tag [@lewtun](https://github.com/lewtun) for a review. We recommend adding the first chapter draft as a single pull request -- the team will then provide feedback internally to iterate on the content 🤗!

## 🙌 Acknowledgements

The structure of this repo and README are inspired by the wonderful [Advanced NLP with spaCy](https://github.com/ines/spacy-course) course.