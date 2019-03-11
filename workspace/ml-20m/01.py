from pyspark.mllib.recommendation import ALS,MatrixFactorizationModel, Rating
import os

# 加载数据，成rdd
data = sc.textFile("../large_files/movielens-20m-dataset/small_rating.csv")

# 提取数据
header = data.first()
data = data.filter(lambda row: row != header)

# 转化成rating对象
ratings = data.map(
    lambda l: l.split(',')
).map(
    lambda l: Rating(int(l[0]), int (l[1]),float(l[2]))
)

# 分成train和test
train, test = ratings.randomSplit([0.8, 0.2])


# train the model
K = 10
epochs = 10
model = ALS.train(train, K, epochs)

# train evaluate
x = train.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(p)
# joins on first item: (user_id, movie_id)
# each row of result is: ((user_id, movie_id), (rating, prediction))
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("train mse: %s" % mse)

# test evaluate
x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ()).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] -r[1][1]) ** 2).mean()
print("test mse: %s" % mse)