from platform import machine
import PySimpleGUI as sg
import snscrape.modules.twitter as sntwitter # Importing of Scraper API
import re
import pandas as pd  # Pandas for importing data into DF
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Vader analysis returns the polarity of the comment
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn
from textblob import TextBlob
from nltk.corpus import stopwords as sp
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from functools import partial
from sklearn.metrics import accuracy_score
# nltk.download("punkt")
# nltk.download('stopwords')
# Eg how neg or pos a comment this and the compound is the overall 

# nltk.downloader.download('vader_lexicon') # Remember to uncomment this to install lexicon file before running scraper

stop_words = set(sp.words('english'))
def search_profile(counts, profiles): # Using Snscrape API to scrape profile data on twitter
    attributes_container = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(profiles).get_items()):
        if i>counts:
            break
        attributes_container.append([tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    tweets_df = pd.DataFrame(attributes_container, columns=["Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
    return tweets_df


def search_results(counts, topic_time): # Using Snscrape API to scrape search result data
    attributes_container = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(topic_time).get_items()): # For each tweet in given topic / time
        if i>counts:
            break
        attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
    # Creating a dataframe to load the list
    tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
    return tweets_df


def clean_text(texts): # Remove special characters and https
    texts = re.sub(r'@[A-Za-z0-9]+', '', texts) 
    texts = re.sub(r'#', '', texts) 
    texts = re.sub(r'RT\s+', '', texts)
    texts = re.sub(r'https?:\S+', '', texts)
    texts = re.sub(r':', '', texts)
    texts = re.sub(r'_','', texts)
    texts = re.sub(r'@','', texts)
    texts = re.sub(r'"', '', texts)
    texts = " ".join(texts.split())
    texts = ''.join([c for c in texts if ord(c) < 128]) # Only look for ASCII characters
    return texts


def popular(text):
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(text)


def emotion(score):
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def export_csv(data):  # Using pandas to export data to csv
    data.to_csv("data_sample.csv")


def append_csv(data): # Append data to csv
    data.to_csv("data_test.csv", mode='w', header=False)


def clear_csv(file):
    f = open(file, 'w')
    f.truncate()
    f.close()

# Updating the dataframe with subjectivity and polarity with textblob

def getSubjectivity(text): # Returns the subjectivity from the library textblob
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text): # Returns the polarity from the library textblob
    return TextBlob(text).sentiment.polarity


def data_read_clean(df): # Perform further data cleaning
    cols = [0] # Specficy first column to drop
    df = df.drop(df.columns[cols], axis=1) # Drop first 2 columns which are unncessary
    df['Tweet'] = df['Tweet'].str.lower()  # Convert tweets to lower caps
    df = df.drop_duplicates(subset=['Tweet'], keep='last') # Drop duplicates from tweet column

    remove_words = ["healthcare", "Healthcare", "Worker", "worker", "healthcare worker", "workers", "never", "so", "before", 
    "healthcare workers", "the", "to", "and", "of", "for", "a", "in", "is", "are", "that", "on", "you", "with", "amp"] # Specify common words to be removed4

    rem = r'\b(?:{})\b'.format('|'.join(remove_words)) # Set parameters to remove this list of words from "Tweet" column
    df['Tweet'] = df['Tweet'].str.replace(rem, '') # Apply the removal to the pandas dataframe tweet

    return df

# Machine Learning Functions

def tokenize(d):
    return word_tokenize(d)


def plot_confusion_matrix(y_test, y_predicted, title='Confusion Matrix'):
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True, fmt='.20g')
    print('Accuracy: %.3f' % accuracy_score(y_test, y_predicted))
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_avg_vector(sent, w):
    vector = np.zeros(100)
    total_words = 0
    for word in sent.split():        
        if word in w.wv.index_to_key:
            vector += w.wv.word_vec(word)
            total_words += 1
    if total_words > 0:
        return vector / total_words
    else:
        return vector


def score_metrics(y_test, y_predicted): 
    accuracy = accuracy_score(y_test, y_predicted) 
    precision = precision_score(y_test, y_predicted,average= 'macro') 
    recall = recall_score(y_test, y_predicted,average='macro') 
    print("accuracy = %0.3f, precision = %0.3f, recall = %0.3f" % (accuracy, precision, recall)) 

    
# Visualizations with different charts / graphs

def pie_chart(df):     # Creates a pie chart to count % of each emotion
    emotions = df['Emotion'].value_counts().rename_axis('Emotion').reset_index(name = 'counts')
    wp={'linewidth':1, 'edgecolor': 'black'}
    explode = (0.1,0.1,0.1)
    colors = ({"Positive" : "green", "Negative": "red", "Neutral" :"blue"})
    emotions.set_index('Emotion').plot(kind='pie', y='counts', figsize=(6, 6), autopct = '%1.0f%%', shadow = True, wedgeprops = wp, explode = explode, label = ''
    , colors=[colors[c] for c in emotions["Emotion"]])
    plt.title("Polarity Distribution")
    plt.show()

    
def histo(df): # Creates a histogram based on the number of likes for each tweet
    df.groupby('Emotion')['Number of Likes'].sum().plot(kind='bar', figsize=(6, 6), color = ["red","blue","green"])
    plt.title("Number of Likes for each Emotion")
    plt.show()
    

def wordcloud(tweet, title, col): # Creating a wordcloud with different emotions
    # image = np.array(Image.open('hashtag.png'))
    words = " ".join(tweets for tweets in tweet.Tweet)
    wordcloud = WordCloud(width=1000, height=800, 
                        background_color="white",min_font_size=10, colormap=col, stopwords=None).generate(words)
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
    pd.DataFrame(unstack_graph).plot.bar(color={'Positive': 'green','Negative': 'red','Neutral': 'blue'})
    plt.title("Count of Likes by Emotion")
    plt.show()


def scatter_plot(df): # Scatter plot between subjectivity & polarity
    df.plot.scatter(x="Polarity", y="Subjectivity", c="DarkBlue", colormap="viridis")
    plt.title("Relationship between Subjectivity and Polarity")
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

    # print(stop_words)
    new = []
    for w in lines:
        if w not in stop_words: # Remove unwanted words using stoplist
            new.append(w)

    new1 = []
    for f in new:
        if f not in list_keyword:
            new1.append(f)

    # print(new[:10])

    df_count = pd.DataFrame(new1)
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


def open_data_window(df): # Open window for data analysis
    df['Tweet'] = df['Tweet'].apply(clean_text)  # Cleaning of tweets
    df = data_read_clean(df)  # Read data from csv and drop duplicates from column "Tweet"
    df['Subjectivity'] = df['Tweet'].apply(getSubjectivity)  # Adding new column subjectivity from textblob
    df['Polarity'] = df['Tweet'].apply(getPolarity)  # Adding new column Polarity from textblob
    print("Cleaning text")
    print(df) # Show the new data after cleaning

    # Defining the buttons and layout
    layout = [[sg.Text("Data Analysis Window\n\nPlease select one of the analysis below :", font=('_20'))], 
    [sg.Button("Piechart"),sg.Button("Histogram"), sg.Button("Kernal Graph")],
    [sg.Button("Positive Word Cloud"), sg.Button("Negative Word Cloud"), sg.Button("Neutral Word Cloud")],
    [sg.Button("Scatter", key="scatter"), sg.Button("Time Graph"), sg.Button("Most Common Words")]]

    window = sg.Window("Analysis", layout, modal=True, size=(600,200)) # Unable to interact with main window until you close second window
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        # Setting the functionality of each button
        elif event == "Piechart": 
            pie_chart(df)
        
        elif event == "scatter":
            scatter_plot(df)

        elif event == "Histogram":
            histo(df)
        elif event == "Kernal Graph":
            kernal_graph(df)

        elif event == "Positive Word Cloud":
            df_positive = df.loc[df['Emotion'] == "Positive"] # Selecting columns with positive emotion
            wordcloud(df_positive, "Positive Word Cloud", "Greens")
        elif event == "Negative Word Cloud":
            df_negative = df.loc[df['Emotion'] == "Negative"] # Selecting columns with negative emotion
            wordcloud(df_negative, "Negative Word Cloud", "Reds")
        elif event == "Neutral Word Cloud":
            def_neutral = df.loc[df['Emotion'] == "Neutral"] # Selecting columns with neutral emotion
            wordcloud(def_neutral, "Neutral Word Cloud", "Blues")

        elif event == "Time Graph":
            time_bar('2019-01-01','2022-01-01', df)

        elif event == "Most Common Words":
            most_common(df)

    window.close()


def open_data_frame(df): # Open window for to view data frame
    data_list = df.values.tolist() # Converting data frame back to list
    layout = [[sg.Text("Scraped Data Frame")],
    [sg.Table(values=data_list, headings=list_dataframe, max_col_width=35,
    auto_size_columns=True,display_row_numbers=True,justification="right",
    num_rows=20, key='-TABLE-',
    row_height=35)]]
    
    window = sg.Window("Data Frame", layout, modal=True, resizable=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()


def similar_words_frame(lst): # Display the words that the model has determined to be similar
    df2 = pd.DataFrame(lst, columns = ['Words', 'Precision'])
    pos_lists = df2.values.tolist()
    layout = [[sg.Text("Scraped Data Frame")],
    [sg.Table(values=pos_lists, headings=similar_words, max_col_width=35,
    auto_size_columns=True,display_row_numbers=True,justification="right",
    num_rows=10, key='-TABLE-',
    row_height=35)]]
    
    window = sg.Window("Data Frame", layout, modal=True, resizable=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()


def machine_learning(df, keywords): # Open window for machine learning
    df['Tweet'] = df['Tweet'].apply(clean_text)  # Cleaning of tweets
    df = data_read_clean(df)  # Read data from csv and drop duplicates from column "Tweet"

    X = df.Tweet
    y = df.Emotion
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
    texts_w2v = df.Tweet.apply(tokenize).to_list() # The use of tokenize function to tokenize the tweets
    w2v = Word2Vec(sentences = texts_w2v, window = 3, vector_size = 100, min_count = 5, workers = 4, sg = 1) # Defining the Word2Vec Model

    

    layout = [[sg.Text("W2Vec / Logistic Regression Model Used:\n ", font=('_20'))], 
    [sg.DD(similar_list, key = "positive_negative", size=(10,10)),sg.Button("Similar Words")],
    [sg.Button("Confusion Matrix")],
    [sg.Text("Accuracy of Model is printed in the terminal", font=('_5'))]]

    window = sg.Window("Machine Learning", layout, modal=True, size=(600,200)) # Unable to interact with main window until you close second window
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        elif event == "Similar Words":
            if values["positive_negative"] == "positive":
                similar_words_frame(w2v.wv.most_similar(positive=keywords))
            elif values["positive_negative"] == "negative":
                similar_words_frame(w2v.wv.most_similar(negative=keywords))

        elif event == "Confusion Matrix":
            # df['w2v_vector'] = df['Tweet'].map(get_avg_vector())
            df['w2v_vector'] = list(map(lambda sent: get_avg_vector(sent, w=w2v), df['Tweet'])) # Apply get_avg_vector function to df
            word2vec_X = df['w2v_vector']
            y = df['Emotion']
            X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(word2vec_X, y,test_size = 0.2)
            word2vec_lr = LogisticRegression(random_state=42,solver = "liblinear")
            word2vec_lr.fit(np.stack(X_train_word2vec), y_train_word2vec)
            y_predicted_word2vec_lr = word2vec_lr.predict(np.stack(X_test_word2vec))
            plot_confusion_matrix(y_test, y_predicted_word2vec_lr)
            

date_time = "since:2020-02-01 until:2020-05-01" # Sample date
list_keyword = ["covid", "covid19", "doctor", "healthcare", "healthcaresystem", "healthcareworker", "hospital", "nurse", "pandemic"] # List of keywords
list_dataframe = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet", "Polarity", "Emotion"] # Headers for the dataframe
list_yearfrom = ["2020","2021","2022"] # Range of years
list_monthfrom = ["01","02","03","04","05","06","07","08","09","10","11","12"] # Range of months
similar_list = ["positive", "negative"]
similar_words = ["Word", "Precision"]

layout = [[sg.Text('Please select the Keyword / Date / Amount to scrape from twitter\n\n', font='_25')], # Designing buttons and layout
 [sg.Text('Keyword :', font="_15"), sg.DD(list_keyword, key = "key_word", size=(50,50))],
 [sg.Text('Date from :', font='_15'), sg.DD(list_yearfrom, key = "year_start", size=(10,10)), sg.DD(list_monthfrom, key = "month_start", size=(10,10))
 ,sg.Text('    Date Until :', font='_15'), sg.DD(list_yearfrom, key = "year_end", size=(10,10)), sg.DD(list_monthfrom, key = "month_end", size=(10,10))],

 [sg.Text('Amount :', font="_15"), sg.InputText(key = "number", size=(20,50))],
    [sg.Button("Scrape Data"), sg.Button("Export to CSV")],
    [sg.Button("Data Analysis"), sg.Button("Data Frame"), sg.Button("Machine Learning"), sg.Exit()]]


window = sg.Window("Python Analysis", layout, size=(700,280), resizable=True)

while True:
    try:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        elif event == "Scrape Data": # Scrapes data and stores it in search_data
            search_data = search_results(int(values["number"]), values["key_word"]+ " near:'Singapore'" + " since:" + values["year_start"] + "-" + values["month_start"]+ "-" + "01" + 
            " until:" + values["year_end"] + "-" + values["month_end"] + "-" + "01") # Takes user-input for time / keyword and amount

            search_data['Polarity'] = search_data['Tweet'].apply(popular)  # Adding new column polarity using VaderSentiment Analysis
            search_data['Emotion'] = search_data['Polarity'].apply(emotion) # Use polarity to get the emotion
            if search_data.empty:
                sg.popup_auto_close("Scrape unsuccessful")
            elif not search_data.empty:
                sg.popup_auto_close("Scrape successful")

        search_data = search_data

        if event == "Export to CSV": # Allows one to export search_data to a csv file
            if search_data.empty:
                sg.popup_auto_close("Export unsuccessful")
            elif not search_data.empty:
                sg.popup_auto_close("Export successful")
                export_csv(search_data)

        if event == "Data Analysis": # Data must be scraped first before Data Analysis can be opened
            open_data_window(search_data)

        if event == "Data Frame":
            open_data_frame(search_data) # Function to display data

        if event == "Machine Learning":
            machine_learning(search_data, values["key_word"])
    except:
        sg.popup_auto_close("Error encountered, please try again") 

window.close()
