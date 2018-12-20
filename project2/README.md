# Machine Learning project 2
## Authors
Nicolas Freundler, Johan Lanzrein, Bruno Wicht
## Contents 
+run_als.py : file to run als
+run_nnmf.py : file to run the non-negative matrix factorization
+run_nn.py : file to run the neural net method. 
+data/ : a folder containing some data to speed up the als reproducibility. 
## Required library
+numpy, pandas, matplotlib
+scikit
+tensorflow and keras
## How to run : 
After installing the required library, run the file of your choice. For convenience, you can either do a fast ALS, which loads some iterations and speeds up the process. Or run it normally. 
The files will produce ready to use submissions. 
The run time of each method varies : 
- Non-negative matrix factorization : ~30min
- Neural net : ~1h20
- ALS with fast mode : ~30min
- ALS in normal mode : ~4h

## Results : 
These results are based on the CrowdAI website, for more training error you can refer to the report. 
- Non-negative matrix factorization : RMSE 1.042
- Neural net : RMSE 1.042
- ALS : RMSE 1.029