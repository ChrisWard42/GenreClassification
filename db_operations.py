"""Module to perform operations on the manybooks sqlite databases"""

import sqlite3


"""
Database Manipulation Functions
"""


def combine_databases(src_db, dst_db):
    """
    Extracts all non-duplicate information from src_db and inserts
    it into dst_db. Both src_db and dst_db must have the same
    table schema. Genres to combine on are currently hardcoded into
    the function, could be generalized later if necessary.

    :param src_db: <String> path to database to extract from
    :param dst_db: <String> path to database to copy to
    :return:
    """
    # Connect to both databases and create cursor objects
    src_conn = sqlite3.connect(src_db)
    dst_conn = sqlite3.connect(dst_db)
    src_c = src_conn.cursor()
    dst_c = dst_conn.cursor()

    # Grab all of the rows from src_db which match criteria
    genre = ("Science Fiction",)
    src_c.execute("SELECT * FROM books WHERE genre=?", genre)
    book_list = src_c.fetchall()

    # Directory to list in dest and separator used in src
    dest_dir = "/Users/nikolai_vogler/manybooks/Science Fiction/"
    src_dir_char = "\\"

    # Insert the fetched rows into dest_db
    # Note: Does not detect duplicates
    for book in book_list:
        # First switch the directory to desired location
        bookl = list(book)
        local_file = bookl[3]
        split_ind = local_file.rfind(src_dir_char)
        new_dir = dest_dir + book[3][split_ind+1:]
        ins_row = (bookl[0], bookl[1], bookl[2], new_dir, bookl[4], bookl[5])

        # Then insert into the database
        dst_c.execute("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)", ins_row)

    dst_conn.commit()
    dst_conn.close()
    src_conn.close()


"""
Main Function
"""


if __name__ == "__main__":
    print("No function has been selected to run. Uncomment one in main.")
    # combine_databases(os.path.join('manybooks2.sqlite'), os.path.join('db', 'manybooks.sqlite'))
    # combine_databases(os.path.join('db', 'manybooks.sqlite'), os.path.join('db', 'manybooks2.sqlite'))
