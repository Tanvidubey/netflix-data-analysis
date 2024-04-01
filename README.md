# Netflix Viewer Analytics and Prediction

# About Data

### This dataset contains data collected from Netflix of different TV shows and movies from the year 2008 to 2021.

- type: Gives information about 2 different unique values one is TV Show, and another is movie.
- title: Gives information about the title of Movie or TV Show.
- director: Gives information about the director who directed the Movie or TV Show.
- cast: Gives information about the cast who plays role in Movie or TV Show.
- release_year: Gives information about the year when Movie or TV Show was released.
- rating: Gives information about the Movie or TV Show are in which category (eg like the movies are only for students, or adults, etc).
- duration: Gives information about the duration of Movie or TV Show.
- listed_in: Gives information about the genre of Movie or TV Show.
- description: Gives information about the description of Movie or TV Show.

# Data Collection and Importing:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
custom_palette =['#E50914', '#221f1f', '#141414', '#B20710']
['#221f1f', '#b20710', '#e50914','#f5f5f1']

# Set the custom palette
sns.set_palette(custom_palette)
```

```python
netflix_orignal =pd.read_csv('netflix.csv')
```

```python
#creating a copy of original data
netflix = netflix_orignal.copy()
```

## Overview of dataset

```python
#top 5 row of data
netflix.head()
```

|  | show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | s1 | Movie | Dick Johnson Is Dead | Kirsten Johnson | NaN | United States | September 25, 2021 | 2020 | PG-13 | 90 min | Documentaries | As her father nears the end of his life, filmm... |
| 1 | s2 | TV Show | Blood & Water | NaN | Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban... | South Africa | September 24, 2021 | 2021 | TV-MA | 2 Seasons | International TV Shows, TV Dramas, TV Mysteries | After crossing paths at a party, a Cape Town t... |
| 2 | s3 | TV Show | Ganglands | Julien Leclercq | Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi... | NaN | September 24, 2021 | 2021 | TV-MA | 1 Season | Crime TV Shows, International TV Shows, TV Act... | To protect his family from a powerful drug lor... |
| 3 | s4 | TV Show | Jailbirds New Orleans | NaN | NaN | NaN | September 24, 2021 | 2021 | TV-MA | 1 Season | Docuseries, Reality TV | Feuds, flirtations and toilet talk go down amo... |
| 4 | s5 | TV Show | Kota Factory | NaN | Mayur More, Jitendra Kumar, Ranjan Raj, Alam K... | India | September 24, 2021 | 2021 | TV-MA | 2 Seasons | International TV Shows, Romantic TV Shows, TV ... | In a city of coaching centers known to train I... |

```python
#last 5 rows of data
netflix.tail()
```

|  | show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8802 | s8803 | Movie | Zodiac | David Fincher | Mark Ruffalo, Jake Gyllenhaal, Robert Downey J... | United States | November 20, 2019 | 2007 | R | 158 min | Cult Movies, Dramas, Thrillers | A political cartoonist, a crime reporter and a... |
| 8803 | s8804 | TV Show | Zombie Dumb | NaN | NaN | NaN | July 1, 2019 | 2018 | TV-Y7 | 2 Seasons | Kids' TV, Korean TV Shows, TV Comedies | While living alone in a spooky town, a young g... |
| 8804 | s8805 | Movie | Zombieland | Ruben Fleischer | Jesse Eisenberg, Woody Harrelson, Emma Stone, ... | United States | November 1, 2019 | 2009 | R | 88 min | Comedies, Horror Movies | Looking to survive in a world taken over by zo... |
| 8805 | s8806 | Movie | Zoom | Peter Hewitt | Tim Allen, Courteney Cox, Chevy Chase, Kate Ma... | United States | January 11, 2020 | 2006 | PG | 88 min | Children & Family Movies, Comedies | Dragged from civilian life, a former superhero... |
| 8806 | s8807 | Movie | Zubaan | Mozez Singh | Vicky Kaushal, Sarah-Jane Dias, Raaghav Chanan... | India | March 2, 2019 | 2015 | TV-14 | 111 min | Dramas, International Movies, Music & Musicals | A scrappy but poor boy worms his way into a ty... |

```python
#shape of data
netflix.shape
```

(8807, 12)

```python
netflix.columns
```

Index(['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
'release_year', 'rating', 'duration', 'listed_in', 'description'],
dtype='object')

```python
netflix.info()
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/a232664a-8aef-4c18-969a-848dba6a5989)


```python
netflix.describe()
```

|  | release_year |
| --- | --- |
| count | 8807.000000 |
| mean | 2014.180198 |
| std | 8.819312 |
| min | 1925.000000 |
| 25% | 2013.000000 |
| 50% | 2017.000000 |
| 75% | 2019.000000 |
| max | 2021.000000 |

### 📌Insights

- **The dataset comprises 8,807 rows and 12 columns.**
- **The movie release years in the dataset range from a minimum of 1925 to a maximum of 2021, indicating that it covers movies from 1925 to 2021.**

# Data Preprocessing

1. Check for Errors and Null Values
2. Replace Null Values with appropriate values
3. Drop down features that are incomplete and are not too relevant for analysis
4. Create new features that can would help to improve prediction

```python
#checking row wise duplicate records
netflix[netflix.duplicated()]
```

|  | show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

```python
#check for null values
miss=netflix.isnull().sum()
miss_percent= round((netflix.isnull().sum()/len(netflix))*100,2)
miss_data=pd.concat([miss,miss_percent],axis=1,keys=['Total','%'])
print(miss_data)
```

   `           Total      %
show_id           0   0.00
type              0   0.00
title             0   0.00
director       2634  29.91
cast            825   9.37
country         831   9.44
date_added       10   0.11
release_year      0   0.00
rating            4   0.05
duration          3   0.03
listed_in         0   0.00
description       0   0.00`

- **The `director` column contains approximately 30% null values.**
- **The `cast` column has 9.37% null values.**
- **The columns `country`, `date_added`, `rating`, and `duration` have minimal missing values.**
- **There are no duplicate rows in this dataset.**
- For missing values in country we will replace with the most common country(mode)
- For cast and director will just add "No Data"

```python
netflix['country'] = netflix['country'].fillna(netflix['country'].mode()[0])
```

```python
netflix['cast'].replace(np.nan,'No Data',inplace=True)
netflix['director'].replace(np.nan,'No Data',inplace=True)
```

## New Feature:

- extract the day and month from the 'date_added' column to make it easier to analyze trends over time.
- Create a new column for genre to extract insights.
- Convert the 'date_added' column to datetime format.
- Drop unnecessary columns

```python
# Convert 'date_added' to date time format with 'coerce' to handle and replace invalid dates.
netflix['date_added'] = pd.to_datetime(netflix['date_added'],errors='coerce')
```

```python
# Extracted month,month_name,day  information from the 'date_added' column
netflix['month_added'] = netflix['date_added'].dt.month
netflix['month_name_added'] = netflix['date_added'].dt.month_name()
netflix['day_added'] = netflix['date_added'].dt.day_name()
```

```python
netflix.head(1)
```

|  | show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description | month_added | month_name_added | day_added |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | s1 | Movie | Dick Johnson Is Dead | Kirsten Johnson | No Data | United States | 2021-09-25 | 2020 | PG-13 | 90 min | Documentaries | As her father nears the end of his life, filmm... | 9.0 | September | Saturday |

```python
netflix['listed_in']
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/976d21b2-f43d-453e-a4c7-3016b46fd234)


```python
#Create a new column for genre
genres = netflix['listed_in'].str.split(', ')

netflix['genre1'] = genres.str[0]
netflix['genre2'] = genres.str[1]  
netflix['genre3'] = genres.str[2]  
```

```python
netflix.head(1)
```

|  | show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description | month_added | month_name_added | day_added | genre1 | genre2 | genre3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | s1 | Movie | Dick Johnson Is Dead | Kirsten Johnson | No Data | United States | 2021-09-25 | 2020 | PG-13 | 90 min | Documentaries | As her father nears the end of his life, filmm... | 9.0 | September | Saturday | Documentaries | NaN | NaN |

```python
#unique values in duration column
netflix['duration'].value_counts()
```
![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/2f150218-dbfb-4074-92e1-fd804d7d2636)


## 📊🔍 EDA: Exploratory Data Analysis

**What different types of shows or movies are uploaded on Netflix?**

```python
#data after cleaning
netflix.head()
```

|  | show_id | type | title | director | cast | country | date_added | release_year | rating | duration | listed_in | description | month_added | month_name_added | day_added | genre1 | genre2 | genre3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | s1 | Movie | Dick Johnson Is Dead | Kirsten Johnson | No Data | United States | 2021-09-25 | 2020 | PG-13 | 90 min | Documentaries | As her father nears the end of his life, filmm... | 9.0 | September | Saturday | Documentaries | NaN | NaN |
| 1 | s2 | TV Show | Blood & Water | No Data | Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban... | South Africa | 2021-09-24 | 2021 | TV-MA | 2 Seasons | International TV Shows, TV Dramas, TV Mysteries | After crossing paths at a party, a Cape Town t... | 9.0 | September | Friday | International TV Shows | TV Dramas | TV Mysteries |
| 2 | s3 | TV Show | Ganglands | Julien Leclercq | Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi... | United States | 2021-09-24 | 2021 | TV-MA | 1 Season | Crime TV Shows, International TV Shows, TV Act... | To protect his family from a powerful drug lor... | 9.0 | September | Friday | Crime TV Shows | International TV Shows | TV Action & Adventure |
| 3 | s4 | TV Show | Jailbirds New Orleans | No Data | No Data | United States | 2021-09-24 | 2021 | TV-MA | 1 Season | Docuseries, Reality TV | Feuds, flirtations and toilet talk go down amo... | 9.0 | September | Friday | Docuseries | Reality TV | NaN |
| 4 | s5 | TV Show | Kota Factory | No Data | Mayur More, Jitendra Kumar, Ranjan Raj, Alam K... | India | 2021-09-24 | 2021 | TV-MA | 2 Seasons | International TV Shows, Romantic TV Shows, TV ... | In a city of coaching centers known to train I... | 9.0 | September | Friday | International TV Shows | Romantic TV Shows | TV Comedies |

## Movies vs TV Shows:

```python
netflix['type'].value_counts()
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/6f8c6c42-b606-4b23-9f89-e486c87ffb5d)


```python
#create a pie chart
netflix['type'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title('Movies vs TV Shows')
```

Text(0.5, 1.0, 'Movies vs TV Shows')

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/8627c497-7af3-4647-8868-37a43387f7eb)


✅Netflix content primarily consists of 69.6% movies and 30.4% TV shows. It is evident that there are more Movies on Netflix than TV shows.

## What different types of shows or movies are uploaded on Netflix?

```python
netflix.groupby(['type','genre1'])['genre1'].value_counts()
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/8e4498c3-9d3b-4466-bcbf-a0de48246751)


## Top Genres in TV Shows and Movies

```python
#top shows and movies
top_tv_genre = netflix[netflix['type']=='TV Show']['genre1'].value_counts().head(10)
top_movie_genre = netflix[netflix['type']=='Movie']['genre1'].value_counts().head(10)
top_tv_genre.values
```

array([774, 399, 388, 253, 221, 176, 120, 120,  67,  40])

```python
#bar graph
fig,axes = plt.subplots(1,2,figsize=(14,6))

sns.barplot(x=top_tv_genre.values, y=top_tv_genre.index,palette=custom_palette,ax=axes[0])
axes[0].set_title("Top TV Genre")

sns.barplot(x=top_movie_genre.values, y=top_movie_genre.index,palette=custom_palette,ax=axes[1])
axes[1].set_title("Top  Movie")
```

Text(0.5, 1.0, 'Top  Movie')

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/2ca88c77-4197-46ea-a310-04ef23b82a3c)

✅ `International TV Shows` dominate the genre preferences among the top TV shows on Netflix, while `Drama` stands out as the primary choice for movies

## Top content creating Countries

```python
top_country = netflix['country'].value_counts().head(10)
```

```python
sns.countplot(y="country", data=netflix, order=top_country.index,palette=custom_palette)
```

<Axes: xlabel='count', ylabel='country'>

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/226694ba-968a-43f2-bf5d-8e863d8cb8d4)

✅The United States, India, and the United Kingdom are the top three countries contributing the most content to Netflix `🥇🇺🇸🥈🇮🇳🥉UK`

## What's the best month to release content?

```python
month_counts = netflix['month_name_added'].value_counts()
month_counts
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/ae79d5b9-6bbf-49cd-b6bd-fca5517a86be)


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.barplot(x=month_counts.index, y=month_counts, palette=custom_palette)

# Adding labels and title
plt.xlabel("Months")
plt.ylabel("Count")
plt.title("Distribution of Content Added by Month")

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

plt.show()
```
![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/77340b56-3110-4023-b0fb-5485f2bd29a3)

✅The months of July, December, and September stand out as the top three periods for content releases on Netflix.

```python
[netflix.info](http://netflix.info/)()
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/769c8308-d7b5-4c65-9748-2aa6a4c07880)


## Year Wise Analysis

```python
year = netflix['release_year'].value_counts().index[0:15]
year
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/dee74a1a-475a-43b6-b3f8-93f698202042)

```python
plt.figure(figsize=(12,10))
ax = sns.countplot(y="release_year", data=netflix, palette=custom_palette, order=year)
```
![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/ba92a177-2376-4b05-b33e-1a761aa3ca7c)

✅2018 was the year when most of the movies were released.

## Viewer Ratings Analysis

```python
rating = netflix['rating'].value_counts().index[0:15]
rating
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/d25ff7a9-23ea-4738-a5c2-baf2e0c0984e)


```python
plt.figure(figsize=(12,10))
ax = sns.countplot(x="rating", data=netflix, palette=custom_palette, order=rating)
```

![image](https://github.com/Tanvidubey/netflix-data-analysis/assets/92937290/7abf55f1-af2a-404b-8c27-69aafefb0c9e)

✅`TV-MA` is the most common viewer rating on Netflix, indicating a strong presence of content suitable for mature audiences, followed by `TV-14` and `TV-PG` with 'R' also available.

# 📌Insight and Conclusion

- **Diverse Content Mix**: Netflix offers a diverse range of content, with 69.6% movies and 30.4% TV shows.
- **Genre Insights**: International TV Shows are a hit among the top TV shows, while Drama rules the movie category.
- **Top Content Providers**: The United States, India, and the United Kingdom are the primary content producers on Netflix.
- **Best Release Months**: For optimal content releases, focus on July, December, and September – they attract the most viewers.
- **Year-wise Peak**: 2018 marked a peak in Netflix movie releases. It's an ideal reference for planning content.
- **Viewer Ratings**: Netflix's most common rating is 'TV-MA,' indicating a significant collection for mature audiences. There's also content for 'TV-14,' 'TV-PG,' and 'R' fans.
- **Recommendation**: To maximize engagement, prioritize International TV Shows, Drama movies, and mature content. Strategically release in July, December, or September, and monitor audience trends for consistent success.
