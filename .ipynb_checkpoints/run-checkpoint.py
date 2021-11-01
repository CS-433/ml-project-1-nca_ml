from implementations import *
from proj1_helpers import *
seed = 1

y, x, ids = load_csv_data("project1/data/train.csv")

# Divide the dataset in subsets in which the same variables are defined,
# and remove the undefined variables in each group :
tx, ids = divide_in_groups(x)


shape01 = tx[0].shape[1]
shape12 = tx[3].shape[1]

# Add a constant feature with value 1 to allow the separating hyperplane to shift from the origin :
usable_tx01 = add_offset(polynomial_embedding(tx[0][:,0:shape01-2],3))
usable_tx11 = add_offset(polynomial_embedding(tx[1],3))
usable_tx02 = add_offset(polynomial_embedding(tx[2],3))
usable_tx12 = add_offset(polynomial_embedding(tx[3][:,0:shape12-2],3))
usable_tx03 = add_offset(polynomial_embedding(tx[4],3))
usable_tx13 = add_offset(polynomial_embedding(tx[5],3))

# Cross validation of the regularization parameter :
lambda01 = cross_validation_demo(y[ids[0]],usable_tx01, seed)
lambda11 = cross_validation_demo(y[ids[1]],usable_tx11, seed)
lambda02 = cross_validation_demo(y[ids[2]],usable_tx02, seed)
lambda12 = cross_validation_demo(y[ids[3]],usable_tx12, seed)
lambda03 = cross_validation_demo(y[ids[4]],usable_tx03, seed)
lambda13 = cross_validation_demo(y[ids[5]],usable_tx13, seed)

# Cross validation of the degree of the polynomial feature agmentation :
degrees01 = bias_variance_demo(y[ids[0]],tx[0][:,0:shape01-2],lambda01)
degrees11 = bias_variance_demo(y[ids[1]],tx[1],lambda11)
degrees02 = bias_variance_demo(y[ids[2]],tx[2],lambda02)
degrees12 = bias_variance_demo(y[ids[3]],tx[3][:,0:shape12-2],lambda12)
degrees03 = bias_variance_demo(y[ids[4]],tx[4],lambda03)
degrees13 = bias_variance_demo(y[ids[5]],tx[5],lambda13)

# Polynomial embedding of the data, uding the degree that resulted from the relevant cross validation :
usable_tx01 = add_offset(polynomial_embedding(tx[0][:,0:shape01-2],degrees01))
usable_tx11 = add_offset(polynomial_embedding(tx[1],degrees11))
usable_tx02 = add_offset(polynomial_embedding(tx[2],degrees02))
usable_tx12 = add_offset(polynomial_embedding(tx[3][:,0:shape12-2],degrees12))
usable_tx03 = add_offset(polynomial_embedding(tx[4],degrees03))
usable_tx13 = add_offset(polynomial_embedding(tx[5],degrees13))

# Linear regression on the augmented data :
w01,loss01 = ridge_regression(y[ids[0]],usable_tx01, lambda01)
w11,loss11 = ridge_regression(y[ids[1]],usable_tx11, lambda12)
w02,loss02 = ridge_regression(y[ids[2]],usable_tx02,lambda02)
w12,loss12 = ridge_regression(y[ids[3]],usable_tx12,lambda12)
w03,loss03 = ridge_regression(y[ids[4]],usable_tx03,lambda03)
w13,loss13 = ridge_regression(y[ids[5]],usable_tx13,lambda13)

y_test, x_test, ids_init = load_csv_data("project1/data/test.csv")

# Dividing the test set into groups with the same variables defined :
tx_test, ids = divide_in_groups(x_test)

# Add offset
usable_tx01_test = add_offset(polynomial_embedding(tx_test[0][:,0:tx_test[0].shape[1]-2],degrees01))
usable_tx11_test = add_offset(polynomial_embedding(tx_test[1],degrees11))
usable_tx02_test = add_offset(polynomial_embedding(tx_test[2],degrees02))
usable_tx12_test = add_offset(polynomial_embedding(tx_test[3][:,0:tx_test[3].shape[1]-2],degrees12))
usable_tx03_test = add_offset(polynomial_embedding(tx_test[4],degrees03))
usable_tx13_test = add_offset(polynomial_embedding(tx_test[5],degrees13))

# Predictions using our previously trained models :
y_pred01 = predict_labels(w01,usable_tx01_test)
y_pred11 = predict_labels(w11,usable_tx11_test)
y_pred02 = predict_labels(w02,usable_tx02_test)
y_pred12 = predict_labels(w12,usable_tx12_test)
y_pred03 = predict_labels(w03,usable_tx03_test)
y_pred13 = predict_labels(w13,usable_tx13_test)

# Putting back together the predictions of the 6 data subsets, and creating a submission file :
y_pred = np.zeros(x_test.shape[0])
y_pred[ids[0]]= y_pred01
y_pred[ids[1]]= y_pred11
y_pred[ids[2]]= y_pred02
y_pred[ids[3]]= y_pred12
y_pred[ids[4]]= y_pred03
y_pred[ids[5]]= y_pred13
create_csv_submission(ids_init,y_pred,"test_20")