import spacy
nlp = spacy.load("en_core_web_sm")


def load_data(corpuspath):
    import pandas as pd
    import numpy as np
    import os
    
    # start empty
    data = None
    
    # traverse directory tree - starts by year
    for year in sorted(os.listdir(corpuspath)):
        
        if year.startswith('.'): continue
        
        # filter out no-data directories
        if os.path.isfile(corpuspath+year) or year.startswith('.'):
            continue

        # filter out not wanted years
        if int(year) < 2013: continue
        if int(year) == 2021: continue

        # continue traverse - for each year, go over all subreddits
        for subreddit in sorted(os.listdir(corpuspath+year)):
            
            if subreddit.startswith('.'): continue

            # file path with submission data and load to pandas DF
            submissions_csv = corpuspath+year+'/'+subreddit+'/'+year+'-'+subreddit+'-submissions.csv'
            aux = pd.read_csv(submissions_csv)
            
            # fill-in subreddit and year info
            aux['subreddit'] = subreddit
            aux['year'] = year
            
            # store in larger structure
            if data is None: data = aux
            else: data = pd.concat([data, aux])
    
    # convert data types 
    data['num_comments'] = data['num_comments'].astype(float)
    data['score'] = data['score'].astype(float)
    data['year'] = data['year'].astype(int)
    
    # dropout anything that does not have useful text in the body
    data = data[
        (data['selftext'] != "") &
        (data['selftext'] != "[deleted]") &
        (data['selftext'] != "[removed]")
    ].dropna()
    
    data.reset_index(drop=True, inplace=True)
    return data


def token_substitution(s):
    """
    Substitute words or sequences with an identifying token
    for later processing.
    """
    import re
    
    s = re.sub(r"\[[\w ]*\]\(http\S+\)", " <URL> ", s)
    s = re.sub(r"http\S+",               " <URL> ", s)
    s = re.sub(r"!\[[\w ]*\]\(\S+\)",    " <IMAGE> ", s)
    s = re.sub(r"\S+@\S+",               " <EMAIL> ", s)
    s = re.sub(r"#\S+",                  " <HASHTAG> ", s)
    s = re.sub(r"r\/\w+",                " <SUBREDDIT> ", s)
    s = re.sub(r"reddit\.\w+",           " <SUBREDDIT> ", s)
    s = re.sub(r"u\/\S+",                " <USER> ", s)
    s = re.sub(r"[!?()\[\]\.,\/\:\;\-\_]+[\s]+", " <PUNCT> ", s)
    s = re.sub(r"[\s]+[!?()\[\]\.,\/\:\;\-\_]+", " <PUNCT> ", s)
    s = s.replace('/', ' <PUNCT> ')
    s = s.strip()
    return s



def remove_meta_tokens(s):
    return ' '.join([w for w in s.split(' ') if not w.startswith('<')])


def expand_contractions(s):
    """
    Expand contractions to full words.
    """
    import contractions
    return ' '.join([contractions.fix(w) for w in s.split()])


def clean_string(s):
    """
    Remove all garbage from the given string.
    Define garbage as styling tokens not related to semantics.
    """
    import re
    #s = s.lower()
    s = s.replace('\n', ' ')
    s = s.replace('\t', ' ')
    s = s.replace(' - ', ' ')
    s = s.replace(' -', ' ')
    s = s.replace('- ', ' ')
    s = s.replace('**', '')
    s = s.replace('*', '')
    s = s.replace('"', '')
    s = s.replace('&#x200b;', ' ')
    s = re.sub('\s+', ' ', s)
    return ' '+s+' '



def preprocess_pipeline(s):
    return remove_meta_tokens(
                token_substitution(
                    clean_string(s)
                )
            ).lower()



def remove_tril(X):
    """
    Remove main diag and lower triangle of square matrix.
    
    :param: X (square matrix): matrix to remove tril from
    
    :return: X (flat array): return other entries in X not removed
    """
    import numpy as np
    # ignore diagonal
    np.fill_diagonal(X, np.nan)
    # ignore lower triangle
    X[np.tril_indices(n=X.shape[0], m=X.shape[1])] = np.nan
    # flatten into a single dimension
    X = X.flatten()
    # remove nan
    X = X[~np.isnan(X)]
    return X


def get_topic_unique_counts(topic_top_words):
    topic_unique_counts = []
    for i, words in enumerate(topic_top_words):
        nunique = 0
        for w in words:
            is_w_unique = True
            for other_words in topic_top_words:
                if other_words != words and w in other_words:
                    is_w_unique = False
                    break
            if is_w_unique:
                nunique += 1
        topic_unique_counts.append(nunique)
    return topic_unique_counts


def print_topics(fi, vocabulary, nwords):
    for k in range(fi.shape[0]):
        print(f"Topic {k}:", ", ".join([vocabulary[i] for i in (-fi[k,:]).argsort()[:nwords]]))
        
        
def get_topics_top_words(fi, vocabulary, nwords):
    topic_top_words = []
    for k in range(fi.shape[0]):
        topic_top_words.append([vocabulary[i] for i in (-fi[k,:]).argsort()[:nwords]])
    return topic_top_words


def plot_wordcloud(word_importances_vector, vocabulary, title=None):
    wordcloud = WordCloud(width = 800, 
                          height = 800, 
                          background_color ='white', 
                          min_font_size = 10,
                          max_words=len(vocabulary))

    # generate importances dictionary
    importances = dict()
    for i, imp in enumerate(word_importances_vector):
        importances[vocabulary[i]] = imp

    # generate wordcloud object
    wordcloud.generate_from_frequencies(importances)

    # generate plot
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)
    plt.title(title)
    plt.show()