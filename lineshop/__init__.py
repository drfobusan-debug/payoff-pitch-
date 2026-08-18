"""Line-shop scanner: what the board is offering, not what a model thinks.

Every rating this repo has tested loses to the closing spread (NFL ridge 10.28
MAE vs the market's 9.91; CFB blends 12.31+ vs 12.13), so the money that is
actually on the table is in execution -- taking a side at a better number than
the consensus, crossing a key number, and the occasional middle that genuinely
pays. This package reads the multi-book board and prices those, using the
empirical joint distribution of closing numbers and results rather than an
opinion about who wins.
"""
