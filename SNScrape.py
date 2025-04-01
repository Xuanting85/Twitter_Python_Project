from attr import attributes
import snscrape.modules.twitter as sntwitter # Importing of Scraper API
import re
import pandas as pd  # Pandas for importing data into DF
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Vader analysis returns the polarity of the comment
# Eg how neg or pos a comment this and the compound is the overall 
# nltk.downloader.download('vader_lexicon') # Remember to uncomment this to install lexicon file before running scraper

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


def clean_text(text): # Remove special characters and https
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#', '', text) 
    text = re.sub(r'RT\s+', '', text)
    text = re.sub(r'https?:\S+', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'_','', text)
    text = re.sub(r'@','', text)
    text = re.sub(r'"', '', text)
    text = " ".join(text.split())
    text = ''.join([c for c in text if ord(c) < 128]) # Only look for ASCII characters


    return text


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
    data.to_csv("data.csv")


def append_csv(data): # Append data to csv
    data.to_csv("data.csv", mode='a', header=False)


def clear_csv(file):
    f = open(file, 'w')
    f.truncate()
    f.close()



search_data = search_results(100, "healthcare workers since:2019-01-01 until:2020-01-01")
profile_data = search_profile(0, 'from:sporeMOH')

# Applying function to extract the polarity / subjectivity and find the emotion of different tweets
# Applying functions to clean data
search_data['Polarity'] = search_data['Tweet'].apply(popular)  # Adding new column polarity using VaderSentiment Analysis
search_data['Tweet'] = search_data['Tweet'].apply(clean_text)  # Cleaning of tweets
search_data['Emotion'] = search_data['Polarity'].apply(emotion) # Use polarity to get the emotion


profile_data['Polarity'] = profile_data['Tweet'].apply(popular)  # Adding new column polarity
profile_data['Tweet'] = profile_data['Tweet'].apply(clean_text)  # Cleaning of tweets
# profile_data['Emotion'] = profile_data['Polarity'].apply(emotion) # Use polarity to get the emotion

# Prints output
# print(search_data)
# print(profile_data)

# Create and export to csv file
export_csv(search_data)

# Clear csv file
# clear_csv("data.csv")