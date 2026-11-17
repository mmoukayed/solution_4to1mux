The following files are unchanged copies of the user's 17 November 2026
camera captures, retained for offline perception regression tests:

- `bin_false_person.png`: `collection_bin_20261117_075855_679741.png`
- `book_colours_live.png`: `book_row_20261117_075717_841958.png`
- `shelf_colours_live.png`: `shelf_column_20261117_075716_437802.png`

The saved frames include the application's green annotation and timestamp.
The annotated red spine is checked at the raw colour stage because its green
outline changes the surrounding colour ring; unannotated spines are checked
through the shelf-context stage. These files have no recorded depth, so depth,
confirmation timing, and ambiguity tests use synthetic data. The positive bin
image used by the tests is an existing simulator capture in `erc_images`.

These checks do not validate physical robot motion or placement.
