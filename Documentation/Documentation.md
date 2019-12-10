## collaborative.py
    This file is used to predict movie based on item-item collaborative filtering. 20 Closest neighbors were considered while calculating rating.

### Functions Used:

    - pearsoncoff(x, y)
        This function calculates pearson coefficient between two items x and y.
        Returns a float pearson coefficient.
    - rating(x, i)
        This function predicts the rating which user x would give to movie i. This function implments item-item filtering. Returns a float predicted rating.
    - populate(fname)
        This function is used to make nested dictionary from the dataset.
    - recommend(user)
        This function returns a list of top 10 movie recommendations for user.

## collaborative_baseline.py
    This file is used to predict movie based on collaborative filtering with baseline approach.

### Functions Used:

    - pearsoncoff(x, y)
        This function calculates pearson coefficient between two items x and y.
        Returns a float pearson coefficient.
    - rating(x,i,total_mean)
        This function predicts the rating which user x would give to movie i. This function implments item-item filtering. total_mean is also passed as parameter. Returns a float predicted rating.
    - populate(fname)
        This function is used to make nested dictionary from the dataset.
    - recommend(user)
        This function returns a list of top 10 movie recommendations for user.
    - calcmean()
        This function is used to calculate overall mean of the matrix.
    
## svd.py
    This file is used to predict movie based on SVD Decomposition Matrix approach.
    
### Functions Used:
    
    - populate(fname)
        This function is used to make nested dictionary from the dataset.
    - normalize(M)
        This function is used to normalize the matrix.
        Input- Matrix M.
        Output- Normalized Matrix M, avg.
    - SVD(ratings_array)
        Function to perform SVD decomposition of a matrix.
        Returns-
        U(2D numpy array): U matrix
        V_t(2D numpy array): Transpose of V matrix
        S(2D numpy array): Sigma Matrix
    - RMSE(M,T)
        This function is used to compute Root Mean Square Error between matrices M and T.
    - SE(M,P)
        This function is used to calculate spearman correlation.
    - kpre(M,T)
        Calculating K-precision. To do this, we will assume that any true rating above 3 corresponds to a relevant item and any true rating below 3 is irrelevant.
    - ninety()
        This function is used for SVD with 90% retained energy.
        
## cur.py
    This file is used to predict movie based on CUR Decomposition Matrix approach.
        
### Functions Used:
        
        - populate(fname)
            This function is used to make nested dictionary from the dataset.
        - normalize(M)
            This function is used to normalize the matrix.
            Input- Matrix M.
            Output- Normalized Matrix M, avg.
        - SVD(ratings_array)
            Function to perform SVD decomposition of a matrix.
            Returns-
            U(2D numpy array): U matrix
            V_t(2D numpy array): Transpose of V matrix
            S(2D numpy array): Sigma Matrix
        - RMSE(M,T)
            This function is used to compute Root Mean Square Error between matrices M and T.
        - SE(M,P)
            This function is used to calculate spearman correlation.
        - kpre(M,T)
            Calculating K-precision. To do this, we will assume that any true rating above 3 corresponds to a relevant item and any true rating below 3 is irrelevant.
        - cur(matrix,rank)
            The main function which is used for calculating CUR Matrices.
        - cur_ninety()
            This function is used for SVD with 90% retained energy.
        
