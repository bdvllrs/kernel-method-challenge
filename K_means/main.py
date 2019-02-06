import matplotlib.pyplot as plt
from utils.utils import X_train_matrix_0,X_train_matrix_1,X_train_matrix_2,\
X_train_0, X_train_1, X_train_2,Y_train_0,Y_train_1,Y_train_2, X_test_0\
,X_test_1,X_test_2,X_test_matrix_0,X_test_matrix_1,X_test_matrix_2, accuracy_score, train_test_split

import numpy as np
from K_means.model import K_Means

colors = 10*["g","r","c","b","k"]

X_train_full = np.concatenate((X_train_matrix_0,X_train_matrix_1,X_train_matrix_2))
Y_train_full = np.concatenate((Y_train_0,Y_train_1,Y_train_2)).reshape(-1)
X_test_full= np.concatenate((X_test_matrix_0,X_test_matrix_1,X_test_matrix_2))
X_train, X_val,y_train,y_val = train_test_split(X_train_full,Y_train_full,test_size=0.1)




clf = K_Means()
clf.fit(X_train)

y_pred_val =[]
y_pred =[]

correct = 0
for i in range(len(X_val)):
    predict_me = np.array(X_val[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    y_pred_val.append(prediction)
    if prediction == y_val[i]:
        correct += 1

unique, counts = np.unique(np.array(y_pred_val), return_counts=True)
print(correct/len(X_val))


with open("submission_k_means.csv", 'w') as f:
    f.write('Id,Bound\n')
    for i in range(len(y_pred)):
        f.write(str(i)+','+str(y_pred[i])+'\n')