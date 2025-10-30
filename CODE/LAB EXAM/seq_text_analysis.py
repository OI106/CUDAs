# TASK 1:

import pandas as pd
from collections import Counter
import re
import time as Time

# Load the IMDB dataset and display the first 20,000 rows

df = pd.read_csv('IMDB Dataset.csv')

print(df.head(20000))

start_time = Time.time()

df_col = df["review"].astype(str)+ " "+ df["sentiment"].astype(str)

words = re.findall(r'\w+', ' '.join(df_col).lower())

words_count = Counter(words)

most_common_words = pd.DataFrame(words_count.most_common(20), columns=['Word', 'Frequency'])

most_common_words.to_csv('seq_output.csv', index=False)

end_time = Time.time()
print("Most common/frequent words:", most_common_words)

print("Time taken (in seconds):", end_time - start_time)



