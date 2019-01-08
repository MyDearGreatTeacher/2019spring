```
Pandas Cookbook
Theodore Petrou
October 2017
```


# DataFrame components--the index, columns, and data


```
>>> movie = pd.read_csv('data/movie.csv')
>>> index = movie.index
>>> columns = movie.columns
>>> data = movie.values


Display each component's values:

>>> index
RangeIndex(start=0, stop=5043, step=1)

>>> columns
Index(['color', 'director_name', 'num_critic_for_reviews',
       ...
       'imdb_score', 'aspect_ratio', 'movie_facebook_likes'],
       dtype='object')

>>> data
array([['Color', 'James Cameron', 723.0, ..., 7.9, 1.78, 33000],
       ..., 
       ['Color', 'Jon Gunn', 43.0, ..., 6.6, 1.85, 456]],
       dtype=object)

```
