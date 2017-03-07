'''Zipf's law simply states that given some corpus (large and structured set of
   texts) of natural language utterances, the occurrence of the most frequent word
   will be approximately twice as often as the second most frequent word, three
   times as the third most frequent word, four times as the fourth most frequent
   word, and so forth.  '''



import re     # REGULAR EXPRESSION
import matplotlib.pyplot as plt

text_file = open('sample.txt', 'r')

text_file_string = (text_file.read())     # string variable

words = re.findall(r'(\b[A-Za-z][a-z]{1,9}\b)', text_file_string.lower())
# find all words that start with a letter (Upper or Lower case) followed by any sequence of letters
# at least 1 characters and not more than 9 characters (2 to 10 characters long)

import collections
from collections import Counter


word_frequency = Counter(words)    # finds the frequncy of each word in the 'words' list.
print(word_frequency)

values = list(reversed(sorted(word_frequency.values())))


plt.loglog(values)
plt.show()
