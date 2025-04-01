from collections import Counter
from distutils.command.config import dump_file
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn
from textblob import TextBlob
from nltk.corpus import stopwords

# The analysis available are piechart / histogram / wordcloud / kernal graph / time graph / scatterplot 

def getSubjectivity(text): # Returns the subjectivity from the library textblob
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text): # Returns the polarity from the library textblob
    return TextBlob(text).sentiment.polarity


def data_read_clean(df):
    cols = [0,1,6] # Specficy first 2 columns to drop
    df = df.drop(df.columns[cols], axis=1) # Drop first 2 columns which are unncessary
    df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
    df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column

    remove_words = ["healthcare", "Healthcare", "Worker", "worker", "healthcare worker", "workers", "never", "so", "before", 
    "healthcare workers", "the", "to", "and", "of", "for", "a", "in", "is", "are", "that", "on", "you", "with", "amp"] # Specify common words to be removed4

    rem = r'\b(?:{})\b'.format('|'.join(remove_words)) # Set parameters to remove this list of words from "Tweet" column
    df['Tweet'] = df['Tweet'].str.replace(rem, '') # Apply the removal to the pandas dataframe tweet

    df['Subjectivity'] = df['Tweet'].apply(getSubjectivity)  # Adding new column subjectivity from textblob
    df['Polarity'] = df['Tweet'].apply(getPolarity)  # Adding new column Polarity from textblob
    return df


# Functions below perform visualization with different charts / graphs

def pie_chart(df):     # Creates a pie chart to count % of each emotion
    emotions = df['Emotion'].value_counts().rename_axis('Emotion').reset_index(name = 'counts')
    wp={'linewidth':1, 'edgecolor': 'black'}
    explode = (0.1,0.1,0.1)
    colors = ({"Positive" : "Green", "Negative": "red", "Neutral" :"Blue"})
    emotions.set_index('Emotion').plot(kind='pie', y='counts', figsize=(6, 6), autopct = '%1.0f%%', shadow = True, wedgeprops = wp, explode = explode, label = ''
    , colors=[colors[c] for c in emotions["Emotion"]])
    plt.title("Polarity Distribution")
    plt.show()

    
def histo(df): # Creates a histogram based on the number of likes for each tweet
    df.groupby('Emotion')['Number of Likes'].sum().plot(kind='bar', figsize=(6, 6), color = ["red","blue","green"])
    plt.title("Number of Likes for each Emotion")
    plt.show()
    # fig = px.histogram(df, x="Emotion", y ="Number of Likes", title="Number of likes for each emotion", width=1200, height=1000) # Takes data from the column "Number of Likes"
    # fig.update_traces(textfont_size = 100,
    #                 marker_line_width=1, marker_color=["green", "red", "blue"])
    # fig.show()


def wordcloud(tweet, title, col): # Creating a wordcloud with different emotions
    # image = np.array(Image.open('hashtag.png'))
    stopwords = set(STOPWORDS)
    stopwords.update(["br", "href"])
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, 
                        background_color="white", stopwords=stopwords, min_font_size=10, colormap=col).generate(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, size = 20)
    plt.show()


def kernal_graph(df):# Kernal distribution graph
# Estimate density of the distribution of emotion / similar to a histogram
    num_neg = df[df['Emotion']=='Negative']['Tweet'].apply(lambda x: len(x.split()))
    num_neu = df[df['Emotion']=='Neutral']['Tweet'].apply(lambda x: len(x.split()))
    num_pos = df[df['Emotion']=='Positive']['Tweet'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12,6))
    sns.kdeplot(num_neg, fill=True, color = 'r').set_title('Distribution of number of words')
    sns.kdeplot(num_neu, fill=True, color = 'b')
    sns.kdeplot(num_pos, fill=True, color = 'g')

    plt.legend(labels=['Negative', 'Neutral','Positive'])
    plt.show()


def time_bar(start, end, df): # Graph to show sentiments over time
    df['Date Created'] = pd.to_datetime(df['Date Created'])
    df['Date Created'] = df['Date Created'].dt.date
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    mask = (df['Date Created'] > start) & (df['Date Created'] <= end)
    df2 = df.loc[mask]
    grouping = df2.groupby(by='Date Created')['Emotion'].value_counts()
    unstack_graph = grouping.unstack(level=1)
    print(unstack_graph)
    pd.DataFrame(unstack_graph).plot.bar(color={'Positive': 'green','Negative': 'red','Neutral': 'blue'})
    plt.title("Count of Likes by Emotion")
    plt.show()


def scatter_plot(df): # Scatter plot between subjectivity & polarity
    df.plot.scatter(x="Polarity", y="Subjectivity", c="DarkBlue", colormap="viridis")
    plt.show()
    

def most_common(df): # Barplot to show the count of popular words
    sentences = [] # Create a new list to store sentences from tweets
    for word in df['Tweet']:
        sentences.append(word)
    sentences
    # print(sentences[:10])

    lines = []
    for line in sentences:
        words = line.split() # Split this sentences up to get each word
        for w in words:
            lines.append(w) # Append the words to a new list
    # print(lines[:10])

    stop_words = set(stopwords.words('english'))
    print(stop_words)
    new = []
    for w in lines:
        if w not in stop_words: # Remove unwanted words using stoplist
            new.append(w)

    # print(new[:10])

    df_count = pd.DataFrame(new)
    # Further removal of punctuations
    df_count.drop(df_count[df_count[0] == '.'].index, inplace=True)
    df_count.drop(df_count[df_count[0] == ','].index, inplace=True)
    df_count.drop(df_count[df_count[0] == '&;'].index, inplace=True)
    df_count.drop(df_count[df_count[0] == '-'].index, inplace=True)
    df_count = df_count[0].value_counts() # Count the occurence of each word


    df_count = df_count[:20] # Take the first 20 words with the most number of count
    plt.figure(figsize=(10,5))
    sns.barplot(df_count.values, df_count.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.xlabel('Count of words', fontsize=12)
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.show() 


df = data_read_clean(pd.read_csv('data.csv'))  # Read data from csv and drop duplicates from column "Tweet"


# pie_chart(df)

histo(df)

# kernal_graph(df)

# Word Cloud 
df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion
def_neutral = df.loc[df['Emotion'] == "Neutral"] # Selecting columns with neutral emotion
# wordcloud_p = wordcloud(df_positive, "Positive Word Cloud", "Greens")
# wordcloud_n = wordcloud(df_negative, "Negative Word Cloud", "Reds")
# wordcloud_neu = wordcloud(def_neutral, "Neutral Word Cloud", "Blues")

# time_bar('2019-01-01','2020-01-01', df)

# scatter_plot(df)

# most_common(df)