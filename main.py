import pandas as pd
from fastai.collab import CollabDataLoaders, collab_learner

ratings = pd.read_csv('ml-100k/u.data', delimiter='\t', header=None,
                      usecols=(0,1,2), names=['user','movie','rating'])
ratings.head()

movies = pd.read_csv('ml-100k/u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()

ratings = ratings.merge(movies)
ratings.head()
data = CollabDataLoaders.from_df(ratings, item_name='movie', bs=64)

data.show_batch()

learn = collab_learner(data, n_factors=50,y_range=(0, 5.5))
learn.fit_one_cycle(5, wd=0.1)

user_id = 919

movies_to_predict_for = [i for i in range(len(movies['movie'].unique()))]

df = pd.DataFrame({
    'user': [user_id] * len(movies_to_predict_for),
    'movie': movies_to_predict_for
})

dl = learn.dls.test_dl(df)
preds = learn.get_preds(dl=dl)

preds_df = pd.DataFrame({
    'movie': [movies['title'][i] for i in movies_to_predict_for],
    'prediction': preds[0].numpy().flatten()
})

best_movie_for_user = ratings.loc[(ratings['user'] == user_id)].drop(['user', 'movie'], axis=1).sort_values(by='rating', ascending=False)[:10]

print(f"\n10 the best movies for user with id{user_id}:\n")
print(best_movie_for_user.to_string(index=False))
print(f"\nReccomendations: \n")
print(preds_df.sort_values(by='prediction', ascending=False)[:10].to_string(Index=False))
