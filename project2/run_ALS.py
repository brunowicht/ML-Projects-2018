"""
	Main run file. Will run an ALS either with a short or long execution. This can be changed with the SHORT_EXECUTION flag below. 
	The long execution can be very long so use at your own discretion..
	The short execution loads previously computed X and Y and only runs one iterations. 
"""
from utilities import *
DATAPATH = "data/"

## load the data. 
dataset = pd.read_csv(DATAPATH+"cleaned_data_train.csv", names="user_id,movie_id,prediction".split(','))
dataset['user_id'] = dataset['user_id']-1
dataset['movie_id'] = dataset['movie_id']-1


# Make a matrix. 
table=pd.pivot_table(dataset,values='prediction',index=['user_id'],columns=['movie_id'])

# fill the na values with 0
matrix=table.fillna(0).values


## Split the data into test and train. 
label_positions=np.logical_not(matrix==0)
test_size=0.1
where=np.where(label_positions.reshape(label_positions.size,))[0]
test_where=random.sample(where.tolist(),int(len(where)*test_size))
train_where=list(set(where.copy().tolist())-set(test_where))
train_matrix=np.zeros(matrix.size,)
train_matrix[train_where]=matrix.reshape((matrix.size,))[train_where]
test_matrix=np.zeros(matrix.size,)
test_matrix[test_where]=matrix.reshape((matrix.size,))[test_where]

train_matrix=train_matrix.reshape(matrix.shape)
test_matrix=test_matrix.reshape(matrix.shape)

train_label_positions=np.logical_not(train_matrix==0)
test_label_positions=np.logical_not(test_matrix==0)

## Short or long execution flag. 
SHORT_EXECUTION=True


dataset_to_predict = pd.read_csv(DATAPATH+"cleaned_sample.csv", names="user_id,movie_id,prediction".split(','))
dataset_to_predict ['user_id'] = dataset_to_predict ['user_id']-1
dataset_to_predict ['movie_id'] = dataset_to_predict ['movie_id']-1
X=0
Y=0
res_matrix=0

if not SHORT_EXECUTION:
## this is for long execution. it will rund ALS 10 times. 
    #initialization with SVD
    A,B,C=np.linalg.svd(matrix,full_matrices=False)
    nb_feature=3
    B_reduced=np.sqrt(np.diag(B[:nb_feature]))



    X=A[:,:nb_feature].dot(B_reduced)
    Y=B_reduced.dot(C[:nb_feature,:])
    
    for i in range(10):
        X,Y=ALS(X,Y,matrix,label_positions,10.,10.)

else:
# runs a single iteration of ALS: 
    X=pd.read_csv('intermediateX.csv').values[:,1:]
    Y=pd.read_csv('intermediateY.csv').values[:,1:]
    print(X[:3,:])
    print(Y[:3,:])
    X,Y=ALS(X,Y,matrix,label_positions,10.,10.)


    


## now that we have ran the ALS we can predict. 
res_matrix=np.round(np.dot(X,Y))
res_matrix[res_matrix>5]=5
res_matrix[res_matrix<1]=1
matrix_pd = pd.DataFrame(res_matrix)
matrix_pd.index.name = "user_id"
matrix_pd.columns.name="movie_id"
name='submissions.csv'


res=matrix_pd.stack().to_frame().reset_index(level=["user_id","movie_id"])
res.columns=["user_id","movie_id","final_predictions"]
# get the predictions. 
predictors= pd.merge(dataset_to_predict,res,right_on=['user_id','movie_id'],left_on=['user_id','movie_id'],how='inner')

predictors=predictors.drop(columns=['prediction'])
# save it under name file. 
create_submission(predictors,name)
