import pandas as pd
from sklearn.linear_model import LinearRegression

df_train = pd.read_csv("./matrix_train.csv", header=None)
df_train_x = df_train.iloc[:,1:52]
df_train_y = df_train.iloc[:,-1]
doc_id_train = df_train.iloc[:,0]

regressor = LinearRegression(fit_intercept=True, normalize=True)
regressor.fit(df_train_x, df_train_y)

df_test = pd.read_csv("./matrix_test.csv", header=None)
df_test_x = df_test.iloc[:,1:52]
df_test_y = df_test.iloc[:,-1]

# y_pred = regressor.predict(df_train_x)
# df = pd.DataFrame({'Doc_Id': doc_id_train, 'Actual': df_train_y, 'Predicted': y_pred})

# matched = 0
# for index, row in df.iterrows():
    # if row['Actual'] == row['Predicted']:
        # matched += 1

# accuracy = (matched / df.shape[0]) * 100
# print("Training accuracy: {0}".format(accuracy))

# y_pred = regressor.predict(df_test_x)
doc_id_test = df_test.iloc[:,0]

# doc_id_test_distinct = set(doc_id_test)

# df = pd.DataFrame({'Doc_Id': doc_id_test, 'Actual': df_test_y, 'Predicted': y_pred})
# print(df)
# matched = 0
# for index, row in df.iterrows():
    # if row['Actual'] == row['Predicted']:
        # matched += 1

# accuracy = (matched / df.shape[0]) * 100
# print("Test accuracy: {0}".format(accuracy))

y_pred = regressor.predict(df_test_x)
df = pd.DataFrame({'Doc_Id': doc_id_test, 'Actual': df_test_y, 'Predicted': y_pred})
sorted_df = df.sort_values(by=['Predicted'], ascending=False)

with open("test_eval.txt", "w") as test_eval:
    rank = 1
    for index, row in sorted_df.iterrows():
        string = "{0} {1} {2}\n".format(row['Doc_Id'], rank, row['Predicted'])
        test_eval.write(string)
        if rank == 1000:
            rank = 0
        rank += 1

