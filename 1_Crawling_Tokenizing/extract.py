# !/usr/bin/python
import os
import sys

from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    """A subclass of HTMLParser that overrides original methods to parse news website articles."""

    def __init__(self) -> None:
        super().__init__()
        self.start_tag = None
        self.attr = None
        self.body = []
        self.start_title = None
        self.titles = []

    def handle_starttag(self, tag, attrs):
        """Identifies start tags and attributes of relevant texts (body of the article, title, subtitle)"""

        if tag == "p":  # article
            for attr in attrs:
                if attr == ('class', 'm-ten  m-offset-one l-eight l-offset-two textabsatz columns twelve'):
                    self.start_tag = tag
                    self.attr = attr

        if tag in ["span", "h2"]:  # title or subtitle
            for attr in attrs:
                if attr in [('class', 'article-breadcrumb__title--inside'),
                            ('class', 'meldung__subhead columns twelve  m-ten  m-offset-one l-eight l-offset-two')]:
                    self.start_title = tag
                    self.attr = attr

    def handle_endtag(self, tag):
        """identifies end tags of relevant texts"""

        if self.start_tag and tag == "p":
            self.start_tag = None
            self.attr = None

        if self.start_title and tag in ["span", "h2"]:
            self.start_title = None

    def handle_data(self, data):
        """processes arbitrary data (e.g. text nodes and the content of <script>...</script> and <style>...</style>)."""

        if self.start_tag or self.start_title:
            text = data.strip()
            self.body.append(text)


def get_files(folder, num_files=None):
    """takes a folder and number of files (num_files). recursively reads all files in subfolders.
    returns a list of paths of filenames depending on num_files"""

    file_paths = []

    for root, _, files in os.walk(folder, topdown=False):
        if root == folder:  # ignore all files in the root folder
            continue
        else:
            for f in files:
                if f == "index.html":
                    continue
                else:
                    file = os.path.join(root, f)
                    file_paths.append(file)
                if num_files:
                    if len(file_paths) == num_files:
                        return file_paths

    return file_paths


def main(folder):
    parser = MyHTMLParser()
    files = get_files(folder, 5)
    for file in files:
        with open(file, "r", encoding='utf-8') as f:
            text = f.read()
            parser.feed(text)
            for text in parser.body:
                print(text)


# python extract.py www.tagesschau.de > text.txt
if __name__ == "__main__":
    main(sys.argv[1])
