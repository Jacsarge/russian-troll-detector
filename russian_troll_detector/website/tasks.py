import twitter
import json
import pandas as pd
import datetime
import math
import pycld2 as cld2
import re
api = twitter.Api(consumer_key='CwmoUS5LGaMy8hEv8GXZdvBSo',
                  consumer_secret='TNkhK3MWXi5l6vZftPXY0X6a9pIfmk7eAUN81AyA9f58ol0Rg5',
                  access_token_key='947384818559475714-21egVCNzbgRPLztJNkDeuqjpfLOy4EE',
                  access_token_secret='8wb1FbZRHQBoZy9IgBVUx9OXICtoycfAlDTgPo9z2Tv71')

def is_valid(t):
    return not (t[0] == '@' or
                t[0] == '#' or
                t.startswith('http') or
                t.startswith('www'))


def get_lang(tweet):
    cleaned = ' '.join([x for x in tweet.split() if is_valid(x)])
    try:
        lang = cld2.detect(cleaned)[2][0][1]
    except:
        return 'unk'
    if lang == 'un' or lang == 'xxx':
        return 'unk'
    return lang


def is_link(tweet):
    # check for no link
    return re.search("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))", tweet.text)


def get_tweets(username):

    # try:
    user = api.GetUser(None, username, True, True)
    # except Exception as err:
    #    print(err)

    user_df = pd.DataFrame()

    print(user)
    if user["verified"] is True:
        print("User is verified, so it's highly likely this person is not a troll.")
        return
    # 1.user id
    user_id = user["id_str"]

    # 2. 200 tweets
    author_tweets = api.GetUserTimeline(screen_name=username, count=200)
    print(len(author_tweets))

    # 3. follower
    follower = user["followers_count"]

    # 4. following
    following = user["friends_count"]

    return user_id, author_tweets, follower, following



def calculate_features(user_id, author_tweets, follower, following):
    feature_dict = {}

    # the dictionary that keeps track of tweets made on each day
    author_daily_tweets = {}
    text = ''
    for tweet in author_tweets:
        # get the text
        text += tweet.text + '\n'

        # construct the daily tweet dictionary
        # Fri Mar 15 00:35:15 +0000 2019
        tweet_datetime = datetime.datetime.strptime(tweet.created_at, '%a %b %d %H:%M:%S +0000 %Y')
        # print(tweet_datetime)

        tweet_day = datetime.date(year=tweet_datetime.year, month=tweet_datetime.month, day=tweet_datetime.day)
        if tweet_day in author_daily_tweets:
            author_daily_tweets[tweet_day] += 1
        else:
            author_daily_tweets[tweet_day] = 1.0

    # 15.text
    feature_dict['text'] = text

    # 1. average number of tweets per user per day
    feature_dict["avg_num_tweets_per_day"] = sum(author_daily_tweets.values()) / len(author_daily_tweets)

    # 2.sd number of tweets per user per day
    diff_sqrt_sum = 0.0
    for day, daily_tweet_num in author_daily_tweets.items():
        diff_sqrt_sum += (daily_tweet_num - feature_dict["avg_num_tweets_per_day"]) ** 2
    feature_dict['sd_num_tweets_per_day'] = math.sqrt(diff_sqrt_sum / len(author_daily_tweets))

    # 3. Average length of tweets (number of characters) per user
    total_length = 0.0
    for tweet in author_tweets:
        total_length += len(tweet.text)
    feature_dict["avg_length_tweets"] = total_length / len(author_tweets)

    # 4. Retweet ratio (retweet rates) per user
    author_retweets = 0.0
    for tweet in author_tweets:
        if re.match("RT ", tweet.text):
            author_retweets += 1
    feature_dict["retweet_rate"] = author_retweets / len(author_tweets)

    # 5. Average number of hashtags per user (word)
    # 6. Average number of hashtags per user (character)
    hashtag_word_count = 0.0
    hashtag_char_count = 0.0
    for tweet in author_tweets:
        # print(tweet)
        hashtags = re.findall(r"#(\w+)", tweet.text)
        hashtag_word_count += len(hashtags)
        hashtag_char_count += sum([len(ht) for ht in hashtags])
    feature_dict["avg_num_hashtags_words"] = hashtag_word_count / len(author_tweets)
    feature_dict["avg_num_hashtags_chars"] = hashtag_char_count / len(author_tweets)

    # 7. ratio of retweets that contain link among all tweets
    retweet_link_count = 0.0
    for tweet in author_tweets:
        if tweet.retweeted_status and is_link(tweet):
            retweet_link_count += 1
    feature_dict["retweet_link_rate"] = retweet_link_count / len(author_tweets)

    # 8. ratio of tweets that contain link among all tweets
    all_link_count = 0.0
    for tweet in author_tweets:
        if is_link(tweet):
            all_link_count += 1
    feature_dict["all_link_rate"] = 1.0 * all_link_count / len(author_tweets)

    # 10. features based on language of tweets -> speaksRussian and num_uniq_languages
    # 11. language distribution
    speaksRussian = 0
    unique_languages = set()
    lang_count = {}

    """ The fixed list of languages we used for the model """
    with open("model_info/fixed_lang.json") as file:
        fixed_languages = json.load(file)

    for tweet in author_tweets:
        curr_language = get_lang(tweet.text)
        if curr_language == "unk":
            # unknown language, as returned by the get_lang() function
            continue
        unique_languages.add(curr_language)
        if curr_language == "ru":
            speaksRussian = 1

        if curr_language in lang_count:
            lang_count[curr_language] += 1
        else:
            lang_count[curr_language] = 1.0

    feature_dict["speaks_russian"] = speaksRussian
    feature_dict["num_uniq_languages"] = len(unique_languages)

    total_lang_usage_count = sum(lang_count.values())

    for lang in fixed_languages:
        if lang in lang_count:
            feature_dict[lang] = lang_count[lang] * 1.0 / total_lang_usage_count
        else:
            feature_dict[lang] = 0

    # 12.follower
    # 13.following
    # 14.ratio of follower/following
    feature_dict["max_followers"] = follower
    feature_dict["max_following"] = following

    if following == 0.0:
        following = 0.5
    feature_dict["ratio_followers_following"] = float(follower) / float(following)

    feature_pd = pd.DataFrame([feature_dict], index=[user_id])

    return feature_pd

