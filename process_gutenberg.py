"""Module to convert all raw gutenberg files into a usable format with sufficient data"""

from gutenberg.cleanup import strip_headers
import os
import json
from urllib.request import urlopen, URLError
import codecs
import sqlite3
from socket import timeout

"""
Global Variables
"""
debug = True

nameddir = '/Users/nikolaivogler/gutenberg-collapsed'
goodreads_cache = '/Users/nikolaivogler/gutenberg-collapsed/goodreads_cache'
gbooks_cache = '/Users/nikolaivogler/gutenberg-collapsed/gbooks_cache'
root = '/Users/nikolaivogler/gutenberg'
sqlite_file = os.path.join('db', 'books.sqlite')

last_dir = '/1\n'
dir_list = os.path.join('data', 'gut_dirs_trimmed.txt')
delim = '%%%'
processed = '$$$'
skipped = '~~~'

genre_types = ["fiction", "non-fiction"]
history_genres = ["historical-fiction", "history", "historical", "war", "military", "roman"]
main_genres = ["fantasy", "romance", "mystery-horror", "poetry", "science-fiction", "historical",
               "general-fiction", "history", "biography", "non-fiction"]
genre_mapping = {"mystery": "mystery-horror", "horror": "mystery-horror", "thriller": "mystery-horror",
                 "contemporary-romance": "romance", "urban-fantasy": "fantasy",
                 "crime": "mystery-horror", "erotica": "romance", "suspense": "mystery-horror",
                 "paranormal-romance": "romance", "historical-romance": "romance", "magic": "fantasy", "vampires": "fantasy",
                 "supernatural": "fantasy", "memoir": "biography",
                 "mystery-thriller": "mystery-horror", "love": "romance",
                 "bdsm": "romance", "mythology": "fantasy", "romantic-suspense": "romance",
                 "science-fiction-fantasy": "science-fiction", "erotic-romance": "romance",
                 "detective": "mystery-horror", "paranormal": "mystery-horror"}
other_genres = ["adventure", "humor", "philosophy", "science", "drama", "psychology",
                "religion", "politics", "action", "travel", "comedy", "self-help"]


"""
Strip Prefixes Routines
"""


def strip_processed_prefix(direc):
    """
    Strips the prefix associated with previously processed files so
    that they will be processed on subsequent runs of process_raw_book.

    :param direc: <String> Path to directory to strip prefixes from filenames
    :return:
    """
    for filename in os.listdir(direc):
        if filename.startswith(processed):
            os.rename(direc + '/' + filename, direc + '/' + filename.replace(processed, '', 1))


def strip_skipped_prefix(direc):
    """
    Strips the prefix associated with previously skipped files so
    that they will be processed on subsequent runs of process_raw_book.

    :param direc: <String> Path to directory to strip prefixes from filenames
    :return:
    """
    for filename in os.listdir(direc):
        if filename.startswith(skipped):
            os.rename(direc + '/' + filename, direc + '/' + filename.replace(skipped, '', 1))


"""
Scrape Genre Listing Routines
"""


def scrape_genre_listing():
    """
    Parses the URLs in the urls list (genre pages on goodreads) and extract all of
    the genre names, storing in a dict which maps genre name to its tagging
    frequency.

    :param direc: <String> Path to directory to strip prefixes from filenames
    :return: <Dict> maps genre names : number of tags
    """
    # The URLs for the genre listings on goodreads
    urls = ["https://www.goodreads.com/genres/list?page=1"
        , "https://www.goodreads.com/genres/list?page=2"
        , "https://www.goodreads.com/genres/list?page=3"]
    genre_dict = {}

    # Grab each URL and process it
    for url in urls:
        # Try each URL three times before throwing URL request error
        response = ""
        num_url_errors = 0
        while num_url_errors < 3:
            try:
                response = urlopen(url, timeout=3)
            except URLError as e:
                print(e)
                num_url_errors += 1
            else:
                break
        if num_url_errors >= 3:
            return genre_dict
        htmltxt = [i.decode('utf-8') for i in response.readlines()]

        # Skip everything in the html until the genre section begins
        start_line = 0
        for line_num in range(start_line, len(htmltxt)):
            # <option value=\"top-level\" > denotes genre section beginning
            if "<option value=\"top-level\" >" not in htmltxt[line_num]:
                continue
            else:
                start_line = line_num + 1
                break

        # Traverse the genre section and contextually extract name and tagged amount,
        # associate with one another as key:value and store in dict
        last_inserted = ""
        for line_num in range(start_line, len(htmltxt)):
            stripped = htmltxt[line_num].strip()
            # <a class=\"mediumText actionLinkLite\" href=\"/genres/ denotes genre of entry
            if "<a class=\"mediumText actionLinkLite\" href=\"/genres/" in htmltxt[line_num]:
                beginIndex = htmltxt[line_num].find(">")
                endIndex = htmltxt[line_num].find("<", beginIndex)
                last_inserted = htmltxt[line_num][beginIndex+1:endIndex]
                genre_dict[last_inserted] = 0
            # Subsequent line beginning with a digit maps to the previous genre
            elif stripped and stripped[0].isdigit():
                beginIndex = 0
                endIndex = stripped.find(" b", beginIndex)
                num_books = int(stripped[beginIndex:endIndex].replace(",", ""))
                genre_dict[last_inserted] = num_books
            # <div class=\"moreLink\"> denotes end of genres section
            elif "<div class=\"moreLink\">" in htmltxt[line_num]:
                break

    return genre_dict


"""
Process Files Routines
"""


def process_file(direc, file):
    """
    Given a file and directory, extracts the title and author from
    the file if it's an English language text. Then strips the header
    information using Gutenberg module and stores the new file in a
    nameddir directory with filename "title%%%author.txt" for future
    processing.

    :param direc: <String> Path to directory containing the file, no trailing '/'
    :param file: <String> Name of the file
    """
    # Grab author and title from the top if language is English
    title = ""
    author = ""
    lang = False
    text = ""
    enc = 'ISO-8859-1'
    path = direc + '/' + file
    with open(path, 'r', encoding=enc) as inf:
        text = inf.read()
        inf.seek(0)  # reset buffer to read author and title
        for line in inf:
            if "Title:" in line:
                title = line.replace("Title: ", "").strip()
            if "Author:" in line:
                author = line.replace("Author: ", "").strip()
            if "Language:" in line and "English" in line:
                lang = True
                break

    # Generate new file name like 'Title%%%Author.txt' for easy lookup
    filename = title[:min(100, len(title))].replace("/", "") + delim\
               + author[:min(100, len(author))].replace("/", "") + ".txt"

    # Remove copyright and metadata from the file
    text = strip_headers(text).strip()

    # Save the file in 'nameddir' (global var) directory
    if lang:
        with open(nameddir + '/' + filename, 'w+') as outf:
            outf.write(text)


def process_all_files(direc_list):
    """
    Processes all files in the directory specified by direc_list by
    first checking if the file is a potential valid book (data stores
    and readmes excluded), then processing the first file in any given
    directory since subsequent files are duplicates with different
    encodings.

    :param direc_list: <String> directory to process files from
    :return:
    """
    found = False
    with open(direc_list, 'r') as dirs:
        for direc in dirs:
            # found and last_dir (global) used to continue process from middle if aborted
            if direc != last_dir and found is False:
                continue
            found = True

            # Open the first file in a given directory (subsequent files are differently
            # encoded duplicates in gutenberg mirror) and process it
            direc = (root + direc).strip()
            if not os.path.isdir(direc):
                continue
            for (dirpath, dirnames, filenames) in os.walk(direc):
                if len(filenames) != 0:
                    if ".DS" not in filenames[0] and "readme" not in filenames[0]:
                        process_file(direc, filenames[0])
                break


"""
Process Books Routines
"""


def get_isbn(title, author):
    """
    Takes a title and an author and inserts them into a query string to the google books
    API, then extracts ISBN information from the response if present. If not ISBN is
    found or the response returns no results, "-1" is returned.

    :param title: <String> Book title
    :param author: <String> Book author
    :return: <List<String>> list of ISBNs or empty list if not found
    """
    # Replace spaces with HTML code for space for use in URL
    title = title.replace(" ", "%20")
    author = author.replace(" ", "%20")

    # Request each URL three times and report errors if request fails
    response = ""
    num_url_errors = 0
    while num_url_errors < 3:
        try:
            response = urlopen('https://www.googleapis.com/books/v1/volumes?q=intitle:{title}+inauthor:{author}'.format(title=title, author=author), timeout=3)
        except URLError as e:
            if debug: print("Error requesting GB URL for book {}, by {}".format(title, author))
            print(e)
            num_url_errors += 1
        except timeout as e:
            if debug: print("Timeout Error requesting GB URL for book {}, by {}".format(title, author))
            print(e)
            num_url_errors += 1
        except UnicodeEncodeError as e:
            if debug: print("UnicodeEncodeError requesting GB URL for book {}, by {}".format(title, author))
            print(e)
            num_url_errors += 1
        else:
            break
    # Request did not succeed so could not retrieve ISBN
    if num_url_errors >= 3:
        return []

    # Parse the returned JSON object for ISBNs
    reader = codecs.getreader("utf-8")
    json_obj = json.load(reader(response))
    isbn_list = []  # list containing both ISBN_10s and ISBN_13s in order of Google Books results
    item = 0
    while len(isbn_list) < 6:
        try:
            isbn_dict = {id_dict['type']: id_dict['identifier'] for id_dict in json_obj['items'][item]['volumeInfo']['industryIdentifiers']}
        except KeyError as e:    # empty page so can't do dict lookup
            break
        except IndexError as e:  # no more items left in Google Books results
            break
        if 'ISBN_13' in isbn_dict:
            isbn_list.append(isbn_dict['ISBN_13'])
        if 'ISBN_10' in isbn_dict:
            isbn_list.append(isbn_dict['ISBN_10'])
        item += 1

    return isbn_list  # return empty isbn_list if nothing found


def get_genre_info(url):
    """
    Attempts to open the url passed and then parses the resulting text to
    find the shelves section and extract the genres and frequencies for the
    book and store them in a dict mapping. Returns the dict mapping generated,
    or an empty dict if request fails or no genre classifications for that book.

    :param url: <String> full url with isbn to goodreads.com
    :return: <Dict> most popular shelved genres for book (name : tags)
    """
    genre_dict = {}

    # Request book page three times before throwing URL request error
    response = ""
    num_url_errors = 0
    while num_url_errors < 3:
        try:
            response = urlopen(url, timeout=3)
        except URLError as e:
            print(e)
            num_url_errors += 1
        else:
            break
    # Return empty genre dict if can't get page
    if num_url_errors >= 3:
        return genre_dict

    # Decode the HTML file into a list
    htmltxt = [i.decode('utf-8') for i in response.readlines()]

    # Skip ahead to the beginning of the genre section
    last_inserted = ""
    start_line = 0
    for line_num in range(start_line, len(htmltxt)):
        # <a href="/work/shelves/ denotes genre section beginning
        if "<a href=\"/work/shelves/" not in htmltxt[line_num]:
            continue
        else:
            start_line = line_num + 1
            break
    # If reached end of file, genre section not found for book, return empty dict
    if start_line == 0:
        return genre_dict

    # Parse through the genre section and extract genres and tag counts, storing in genre_dict
    for line_num in range(start_line, len(htmltxt)):
        # <a class="actionLinkLite bookPageGenreLink" denotes line with genre name
        if "<a class=\"actionLinkLite bookPageGenreLink\"" in htmltxt[line_num] and "&gt" in htmltxt[line_num]:
            continue
        if "<a class=\"actionLinkLite bookPageGenreLink\"" in htmltxt[line_num]:
            beginIndex = htmltxt[line_num].find("/genres/") + 7
            endIndex = htmltxt[line_num].find("\">", beginIndex)
            last_inserted = htmltxt[line_num][beginIndex+1:endIndex]
            genre_dict[last_inserted] = 0
        # <a title=" denotes line with number of people who shelved that genre
        if "<a title=\"" in htmltxt[line_num]:
            beginIndex = htmltxt[line_num].find("\"")
            endIndex = htmltxt[line_num].find(" ", beginIndex)
            num_tags = int(htmltxt[line_num][beginIndex+1:endIndex])
            genre_dict[last_inserted] = num_tags
        # <a class="actionLink right bookPageGenreLink__seeMoreLink" denotes end
        if "<a class=\"actionLink right bookPageGenreLink__seeMoreLink\"" in htmltxt[line_num]:
            break

    return genre_dict


def map_genres(genres):
    """
    Takes a dictionary of genres and maps them to a small subset of
    genres which are used in classification and verification. Each
    scraped genre maps to one other mapped genre.

    Genres:
        Fiction: Fantasy, Romance, Mystery-Horror, Poetry, Science Fiction,
                 Historical Fiction, General Fiction
        Non-Fiction: History, Biography, Other

    :param genres: <Dict> genre names mapped to number of times tagged
    :return: <String> genre type (fiction, non-fiction)
            , <Dict> Mapped genres only : max times tagged
    """
    mapped_genres = {}

    # Determine the overall classification for later classifications
    gtype = ""
    if "fiction" in genres:
        gtype = "fiction"
    elif "non-fiction" in genres:
        gtype = "non-fiction"

    # Map each genre in the dict to our selected genres based on predefined rules
    for k,v in sorted(genres.items(), key=lambda x: x[1]):
        # Map main genres directly
        if k in main_genres:
            mapped_genres[k] = v

        # Map other top 100 genres to their main genres if applicable
        elif k in genre_mapping:
            mapped_genres[genre_mapping[k]] = v

        # Handle the history/historical fiction special case
        elif k in history_genres and gtype is "fiction":
            mapped_genres["historical-fiction"] = v
        elif k in history_genres and gtype is "non-fiction":
            mapped_genres["history"] = v

        # Otherwise, classify as other
        elif k in other_genres and gtype is "fiction":
            mapped_genres["general-fiction"] = v
        elif k in other_genres and gtype is "non-fiction":
            mapped_genres["other"] = v

    return gtype, mapped_genres


def insert_into_db(conn, title, author, isbn, gtype, genres, mapped_genres):
    """
    Takes information about a book and inserts a new row into the database containing that
    information, updating the existing entry if the ISBN is already present.

    :param conn: <Connection> connection to the database to insert into
    :param title: <String> title of the book
    :param author: <String> author of the book
    :param isbn: <String> ISBN of the book (13 before 10 if available)
    :param gtype: <String> fiction or non-fiction
    :param genres: <Dict> genre name : frequency tagged, scraped from goodreads
    :param mapped_genres: <Dict> genre name : frequency tagged, mapped by map_genres
    :return:
    """
    c = conn.cursor()  # Cursor to database

    # Sanitize the passed in data
    isbn_key = int(isbn)

    # Insert the book data into the table if isbn not in table
    book_data = (isbn_key, title, author, gtype, str(genres), str(mapped_genres))
    c.execute("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, NULL)", book_data)

    conn.commit()  # Commit database changes


def process_raw_book(conn, direc, book):
    """
    Takes a file name and a directory of collapsed book files, gets the
    title and author from the file name, then uses these to query the
    isbn. Generates a url from the isbn to scrape genre classification
    information from goodreads and stores in a dict mapping genre name to
    frequency. Converts the text into bag of words format. Inserts all of
    the information gathered into a row in the database if no errors
    occurred during processing.

    :param conn: <Connection> connection to the database to insert into
    :param direc: <String> The directory path containing all collapsed books
    :param book: <String> The file name of the book to process in direc
    :return:
    """
    c = conn.cursor()  # Cursor to database

    # Check if file processed or skipped already and return if so
    if processed in book or skipped in book:
        if debug: print("Book already has been processed or skipped: " + book)
        return

    # Skip all data store files created by Mac OS
    if ".DS_" in book:
        if debug: print("Skipping file: .DS_Store")
        return

    # Query google books API for title to get ISBNs
    title = book.split(delim)[0].strip()
    author = book.split(delim)[1].replace(".txt", "").strip()
    isbns = get_isbn(title, author)
    # If no ISBN was found, skip the book and print out problem
    if not isbns:
        if debug: print("File skipped due to invalid ISBN: " + book)
        os.rename(direc + '/' + book, direc + '/' + skipped + book)
        return

    # Loop through all of the isbns obtained by get_isbn
    genres = {}
    isbn_used = ""
    for isbn in isbns:
        # Check if isbn exists in database before calling goodreads
        c.execute("SELECT * FROM books WHERE isbn=?", (isbn,))
        if c.fetchone():
            if debug: print("Book with this ISBN already exists in database: " + book)
            return
        # Use ISBN to append to https://www.goodreads.com/book/isbn/
        url = "https://www.goodreads.com/book/isbn/" + isbn
        # Grab genre information from this and store in scraped genres array
        genres = get_genre_info(url)
        # If we grabbed genre info successfully, break, otherwise check next ISBN
        if genres:
            isbn_used = isbn
            break

    # If we couldn't get genre information with any ISBN in list, invalid file so return
    if not genres:
        if debug: print("File skipped due to lack of genre information: " + book)
        os.rename(direc + '/' + book, direc + '/' + skipped + book)
        return

    # Insert the title, author, ISBN, and genre info into DB table
    gtype, mapped_genres = map_genres(genres)
    insert_into_db(conn, title, author, isbn_used, gtype, genres, mapped_genres)

    # Mark file as processed successfully by adding prefix
    os.rename(direc + '/' + book, direc + '/' + processed + book)


def process_all_books(direc):
    """
    Directory specified should be the one used as output from
    process_all_files and contain txt files with title%%name.txt format
    and stripped book text in the file.

    :param direc: <String> The directory path containing all collapsed books
    :return:
    """
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    # Create database table if necessary
    c.execute('''CREATE TABLE IF NOT EXISTS books
             (isbn int CONSTRAINT isbn_exists PRIMARY KEY, title varchar(255), author varchar(255),
             gtype varchar(255), genres_scraped varchar(255), genres_mapped varchar(255), genres_predicted varchar(255))''')

    # Run process_raw_book on each book in directory
    for (dirpath, dirnames, filenames) in os.walk(direc):
        for file in filenames:
            process_raw_book(conn, direc, file)
        break  # prevent traversing cache sub-directories

    conn.commit()
    conn.close()


"""
Testing Function
"""

def test_functions():
    """
    Function to test the outputs of the other individual functions in the pipeline.
    Results currently verified manually.

    :return:
    """
    # Test get_isbn function
    print(get_isbn("The Odyssey", "Homer"))
    print(get_isbn("Poems", "John Hay"))
    print(get_isbn("Fifty Shades of Grey", "E L James"))

    # Test get_genre_info function
    print(get_genre_info("https://www.goodreads.com/book/isbn/" + "1904633374"))
    print(get_genre_info("https://www.goodreads.com/book/isbn/" + "1511863560"))
    print(get_genre_info("https://www.goodreads.com/book/isbn/" + "0345803574"))

    # Test map_genres function
    odyssey = {'adventure': 613, 'poetry': 2880, 'mythology': 1557, 'epic': 412, 'read-for-school': 492, 'fiction': 3081, 'literature': 1051, 'school': 1118, 'classics': 12244, 'fantasy': 1077}
    poems = {}
    fifty_shades = {'contemporary-romance': 41, 'romance': 420, 'erotic-romance': 52, 'contemporary': 42, 'fiction': 134, 'love': 28, 'bdsm': 118, 'chick-lit': 32, 'adult': 69, 'erotica': 180}
    print(map_genres(odyssey))
    print(map_genres(poems))
    print(map_genres(fifty_shades))

    # Test database and process_raw_book function
    conn = sqlite3.connect(os.path.join('db', 'test_db.sqlite'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS books
             (isbn int CONSTRAINT isbn_exists PRIMARY KEY, title varchar(255), author varchar(255),
             gtype varchar(255), genres_scraped varchar(255), genres_mapped varchar(255), genres_predicted varchar(255))''')

    process_raw_book(conn, nameddir, "$$$War and Peace%%%Leo Tolstoy.txt")
    c.execute("SELECT * FROM books")
    print(c.fetchone())

    conn.commit()
    conn.close()


"""
Main Function
"""


if __name__ == "__main__":
    print("No function has been selected to run. Uncomment one in main.")
    # Note: Routines can be run independently or in sequence, except strip
    # routines which should only be run if needed

    # 1. Process All Files Routine
    # process_all_files(dir_list)
    # 2. Scrape Genre Listing Routine
    # genre_dict = scrape_genre_listing()
    # 3. Process All Books Routine
    # process_all_books(nameddir)

    # Other Routines
    # strip_processed_prefix(nameddir)
    # strip_skipped_prefix(nameddir)
