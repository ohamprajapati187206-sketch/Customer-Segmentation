from kneed import KneeLocator
from sklearn.cluster import KMeans

def get_optimum_k(df):
    #i)find wcss
    wcss=[]
    k_values=range(1,11)
    for i in k_values:
        model=KMeans(n_clusters=i)
        model.fit(df)
        wcss.append(model.inertia_)

    #ii)knee locator 
    KL=KneeLocator(k_values,wcss,curve='convex',direction='decreasing')
    return KL.elbow