"""Module to download genre classified books from manybooks.net"""

from gutenberg.cleanup import strip_headers
import os
import re
from chardet.universaldetector import UniversalDetector
from urllib.request import Request, urlopen, URLError
import sqlite3

"""
Global Variables
"""
debug = True

adventure_links = {'http://manybooks.net/categories/ADV': 90}
biography_links = {'http://manybooks.net/categories/BIO': 88}
# fantasy_links = {'http://manybooks.net/categories/FAN': 15, 'http://manybooks.net/categories/MYT': 10}
history_links = {'http://manybooks.net/categories/HIS': 160}
mystery_links = {'http://manybooks.net/categories/MYS': 43}
poetry_links = {'http://manybooks.net/categories/POE': 71}
romance_links = {'http://manybooks.net/categories/ROM': 80}
scifi_links = {'http://manybooks.net/categories/SFC': 69}

categories = [(adventure_links, 'Adventure'), (biography_links, 'Biography'),
              (history_links, 'History'), (mystery_links, 'Mystery-Horror'),
              (poetry_links, 'Poetry'), (romance_links, 'Romance'),
              (scifi_links, 'Science Fiction')]

dl_dir = "Z:\\dev\\projects\\cs175\\manybooks"
sqlite_file = os.path.join('db', 'manybooks.sqlite')
html_cache = "cat_html"
microsoft_invalid = '[<>:\"/\\|?*]'


"""
Sanitize Texts Routine
"""


def sanitize_texts(directory):
    """
    Strip all header and copyright information from downloaded text files in the
    specified directory using gutenberg.strip_headers module and ensure proper
    file encodings.

    :param directory: <String> A string containing the full path to directory containing files to strip
    :return:
    """

    for item in os.listdir(directory):
        file_path = os.path.join(directory, item)
        if os.path.isfile(file_path):

            # Detect file encoding, takes time to run
            with open(file_path, 'rb') as inf:
                text = inf.readlines()
            detector = UniversalDetector()
            for line in text:
                detector.feed(line)
                if detector.done:
                    break
            detector.close()
            encoding = detector.result['encoding']

            # Open file, strip headers, and save result
            with open(file_path, 'r', encoding=encoding) as inf:
                text = inf.read()
            text = strip_headers(text).strip()
            os.remove(file_path)
            with open(file_path, 'w+', encoding=encoding) as outf:
                outf.write(text)


"""
Download Books Routine
"""


def open_url(url, outfile, flag):
    """
    Opens the page at url if outfile does not already exist, otherwise open
    the outfile cached version.

    :param url: <String> web address to open
    :param outfile: <String> cached file location
    :param flag: <String> either 'html' or 'txt'
    :return: <List[String]> text of the page
    """

    htmltxt = []

    # Check if html/txt exists in cached directory already first
    if os.path.isfile(outfile):
        with open(outfile, 'r') as r:
            htmltxt = r.readlines()
        return htmltxt

    if debug: print("DEBUG: Open URL was called")  # Lets us determine if cache miss

    # Fetch the html file from url, decode if necessary, cache it
    if flag == "html":
        num_url_errors = 0
        response = ""
        if debug: print("DEBUG: " + url)

        # Check for a response 3 times before throwing URL response error
        while num_url_errors < 3:
            try:
                response = list(urlopen(url))
            except URLError as e:
                print(e)
                num_url_errors += 1
            else:
                break
        if num_url_errors >= 3:
            return response

        # Run detector to get file encoding
        detector = UniversalDetector()
        for line in response:
            detector.feed(line)
            if detector.done: break
        detector.close()
        encoding = detector.result['encoding']
        if debug: print("DEBUG: " + encoding)
        htmltxt = [i.decode(encoding) for i in response]

        # Write the item to the cache location
        with open(outfile, 'w+', encoding=encoding) as w:
            for item in htmltxt:
                w.write(item)

    # Fetch text files and write them to specified directory
    if flag == "text":
        # Request the book text download url
        req = Request(url)
        textfile = list(urlopen(req))

        # Determine the file encoding
        detector = UniversalDetector()
        for line in textfile:
            detector.feed(line)
            if detector.done: break
        detector.close()
        encoding = detector.result['encoding']
        if debug: print("DEBUG: " + encoding)

        # Write the book text file to genre directory if encoding is usable
        if (encoding == 'ascii' or encoding == 'ISO-8859-2' or encoding == 'ISO-8859-1'
            or encoding == 'UTF-8'):
            outputtxt = [i.decode(encoding) for i in textfile]

            with open(outfile, 'w+', encoding=encoding) as w:
                for line in outputtxt:
                    w.write(line)

    return htmltxt


def generate_book_download_urls(html, genre):
    """
    Takes an HTML file represented as a list of strings and searches it for
    book entries to extract information about title, author, and download link
    for the book, then stores this information as a series of book objects (dicts)
    in a list and returns that list.

    :param html: <List[String]> html file of category page
    :param genre: <String> name of book genre
    :return: <List[Dict]> List of book objects
    """
    book_list = []
    start_line = 0

    # Skip all of the lines in the file until we get to the book container
    for line_num in range(start_line, len(html)):
        # Section starts at '<div class="grid_13 omega">'
        if "<div class=\"grid_13 omega\">" not in html[line_num]:
            continue
        else:
            start_line = line_num + 1
            break
    # If there's no grid_13 and we hit end of file, must not contain books so return
    if start_line == 0:
        return book_list

    # Starting with the book container, grab all of the book info and urls
    for line_num in range(start_line, len(html)):
        # A new book entry starts on line with '/titles/'
        if "/titles/" in html[line_num]:
            # Extract the beginning and ending indices for desired info on each book
            id_begin = html[line_num].find("/titles/") + 8
            id_end = html[line_num].find(".html", id_begin)
            title_begin = html[line_num].find(".html\">", id_begin) + 7
            title_end = html[line_num].find("</a>", title_begin)
            author_begin = html[line_num].rfind("<br />") + 6

            # Use the begin/end indices to grab the desired information
            idn = html[line_num][id_begin:id_end]
            title = html[line_num][title_begin:title_end]
            author = html[line_num][author_begin:].strip()

            # Build the download and info URLs from component information
            dl_url = "http://manybooks.net/send/1:text:.txt:text/" + idn + "/" + idn + ".txt"
            info_url = "http://manybooks.net/titles/" + idn + ".html"

            # Append a book object (dict) to the end of the book list
            book_list.append({"id": id, "title": title, "author": author,
                             "dl_url": dl_url, "info_url": info_url})
        # Section ends at '<div class="grid_6">'
        if "<div class=\"grid_6\">" in html[line_num]:
            break

    # Check if files already downloaded, and if so then don't append them to list to prevent duplicates
    book_list2 = []
    for book in book_list:
        outfile = os.path.join(dl_dir, genre, re.sub(microsoft_invalid, '', book['title'][:min(100, len(book['title']))]) +
                               '%%%' + re.sub(microsoft_invalid, '', book['author'][:min(100, len(book['author']))]) + '.txt')
        if not os.path.isfile(outfile):
            book_list2.append(book)

    return book_list2


def download_books(book_list, genre):
    """
    Generates a file name from the author and title of a book concatenated with its genre, then
    opens the URL generated for the download of that book. File name is passed to check if it
    already exists in cache.

    :param book_list: <List[Dict]> List of book objects
    :param genre: <String> Name of genre, also name of sub-directory
    :return:
    """
    for book in book_list:
        outfile = os.path.join(dl_dir, genre, re.sub(microsoft_invalid, '', book['title'][:min(100, len(book['title']))]) +
                               '%%%' + re.sub(microsoft_invalid, '', book['author'][:min(100, len(book['author']))]) + '.txt')
        open_url(book['dl_url'], outfile, "text")


def insert_into_db(conn, book_list, genre):
    """
    Takes a connection to a database, a list of book objects, and a genre name, and inserts
    a new entry into the database for each book in the book list. Information of each entry
    includes: title, author, genre, local file location, download url, information url.

    :param conn: <Connection> connection object for sqlite3 database
    :param book_list: <List[Dict]> list of dictionaries representing book objects
    :param genre: <String> the genre of the book
    :return:
    """

    c = conn.cursor()

    # Insert the book data into the table
    for book in book_list:
        local_file = os.path.join(dl_dir, genre, re.sub(microsoft_invalid, '', book['title'][:min(100, len(book['title']))]) +
                                  '%%%' + re.sub(microsoft_invalid, '', book['author'][:min(100, len(book['author']))]) + '.txt')
        if os.path.isfile(local_file):
            book_data = (book['title'], book['author'], genre, local_file, book['dl_url'], book['info_url'])
            c.execute("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)", book_data)

    conn.commit()


def download_all_books():
    """
    Goes through the list of categories to download from manybooks.net and all of their
    numbered pages. For each numbered category page, fetches the html with open_url and
    caches it locally for future calls, generates a book download list by parsing the html
    for book entries using generate_book_download_urls, downloads the book files generated
    and stores them locally using download_books, then inserts information about each book
    into a local database using insert_into_db.

    :return:
    """

    # Initialize connection to database (sqlite_file defined as global)
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    # Create Table if doesn't already exist
    c.execute('''CREATE TABLE IF NOT EXISTS books
             (title varchar(255), author varchar(255), genre varchar(255), local_file varchar(255)
             , dl_url varchar(255), info_url varchar(255))''')

    # Loop through categories tuples, create dirs for each and DL all associated items
    for links, genre in categories:
        os.makedirs(os.path.join(dl_dir, genre, html_cache), exist_ok=True)
        # categ is url to base genre page, pages is # pages within that genre
        for categ, pages in links.items():
            start_at = 1  # TODO: CHANGE THIS TO HIGHER START PAGE IF RESUMING PROCESS
            end_at = pages + 1  # TODO: CHANGE THIS TO LOWER END TO RUN ON ONLY A FEW PAGES
            for i in range(start_at, end_at):
                url = categ + '/' + str(i) + '/en'
                outfile = os.path.join(dl_dir, genre, html_cache, str(i) + ".html")
                html = open_url(url, outfile, "html")
                book_list = generate_book_download_urls(html, genre)
                download_books(book_list, genre)
                insert_into_db(conn, book_list, genre)
                # return  # TODO: UNCOMMENT THIS LINE TO RUN ON ONLY FIRST GENRE IN LIST

    conn.commit()
    conn.close()


"""
Main Function
"""


if __name__ == "__main__":
    print("No function has been selected to run. Uncomment one in main.")
    # Note: Routines can be run independently or in sequence, but sanitize would
    # need to be called on every genre directory generated since it doesn't recurse.

    # 1. Download All Books Routine
    # download_all_books()
    # 2. Sanitize Texts Routine
    # sanitize_texts("Z:\\sanitize")
