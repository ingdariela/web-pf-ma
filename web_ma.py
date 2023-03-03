import pandas as pd
import streamlit as st
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
import pickle


st.set_option('deprecation.showPyplotGlobalUse', False)

# Page Configuration
st.set_page_config(page_title="Clustering Brokers - MA",
                   page_icon=":o:",
                   layout="wide")

st.title(":o: Clustering Brokers")


## Clustering

st.markdown("##")

# Create a function to train the model
def train_model(data):

    data0 = data.copy()

    #Groupby
    data["Quantity_TL"] = data["Type"].apply(lambda x: np.where(x == "TL",1,0))
    data["Quantity_TL"] = data["Quantity_TL"] *data["Quantity"]
    data["Quantity_LTL"] = data["Type"].apply(lambda x: np.where(x == "LTL",1,0))
    data["Quantity_LTL"] = data["Quantity_LTL"] *data["Quantity"]
    data.drop(['StateOrigin','Route','Quantity','Type','WeeksAgo','TruckType'],axis=1,inplace=True)
    data = data.groupby(['Company'],as_index=False).agg({'Distance_mi':'mean','Rate_mi':'mean','Weight_k':'mean',
                                                                'Size_ft':'mean','Order_Antc_Time':'median','Quantity_TL':'sum','Quantity_LTL':'sum'})

    # Separate numeric and categorical variables
    num_cols = ((data.dtypes == 'float64') | (data.dtypes == 'int64'))
    num_cols = list(num_cols[num_cols].index) 
    object_cols = (data.dtypes == 'object') 
    object_cols = list(object_cols[object_cols].index) 

    # Preprocessing
    data_num = pd.DataFrame(preprocessing.scale(data[num_cols]))
    data_num.columns = num_cols
    data_obj = data[object_cols]
    data_pp = data_obj.join(data_num,how='inner')
    obj_cols = (data_pp.dtypes == 'object') 
    obj_index = [i for i in range(len(obj_cols)) if obj_cols[i] == True]

    # Load model from file
    with open('kpt_model.pkl', 'rb') as f:
        kpt = pickle.load(f)

    modelo_kpt = kpt.fit_predict(data_pp,categorical=obj_index)
    data_pp['Cluster_KPrototypes'] = modelo_kpt

    # Load model from file
    with open('km_model.pkl', 'rb') as f:
        km = pickle.load(f)

    mod_km = km.fit(data_num)
    data_pp['Cluster_KMeans'] = mod_km.labels_

    data_sg = pd.merge(data0, data_pp[['Company','Cluster_KPrototypes','Cluster_KMeans']],how='left',on='Company')

    #Groupby
    data_sg["Quantity_TL"] = data_sg["Type"].apply(lambda x: np.where(x == "TL",1,0))
    data_sg["Quantity_TL"] = data_sg["Quantity_TL"] *data_sg["Quantity"]
    data_sg["Quantity_LTL"] = data_sg["Type"].apply(lambda x: np.where(x == "LTL",1,0))
    data_sg["Quantity_LTL"] = data_sg["Quantity_LTL"] *data_sg["Quantity"]
    data_sg.drop(['StateOrigin','Route','Quantity','Type','WeeksAgo','TruckType'],axis=1,inplace=True)

    cnt_kpt = pd.DataFrame(data_pp.groupby('Cluster_KPrototypes').count()['Company'])
    cnt_km = pd.DataFrame(data_pp.groupby('Cluster_KMeans').count()['Company'])

    # Devolver los resultados del modelo
    return data_pp, data_sg, cnt_kpt, cnt_km

# Create a function to upload data
def upload_data():
    data = st.file_uploader("Upload CSV data file", type=["csv"])
    if data:
        return pd.read_csv(data)
    else:
        return None
    
# Upload the data
loads = upload_data()

if loads is not None:

    # Mostrar los resultados
    df_model_clust, df_sg, df_ckpt, df_ckm = train_model(loads)
    st.write("Results of the k-prototypes and k-means model:")
    st.dataframe(df_model_clust)

    st.markdown("##")

    st.write("Profiles - KPrototypes:")
    g1 = df_sg.groupby('Cluster_KPrototypes').agg({'Distance_mi':'mean','Rate_mi':'mean','Weight_k':'mean',
                                                                            'Size_ft':'mean','Order_Antc_Time':'median','Quantity_TL':'sum','Quantity_LTL':'sum'})
    st.dataframe(df_ckpt)
    st.dataframe(g1)

    st.markdown("##")

    st.write("Profiles - KMeans:")
    g2 = df_sg.groupby('Cluster_KMeans').agg({'Distance_mi':'mean','Rate_mi':'mean','Weight_k':'mean',
                                                                            'Size_ft':'mean','Order_Antc_Time':'median','Quantity_TL':'sum','Quantity_LTL':'sum'})
    st.dataframe(df_ckm)
    st.dataframe(g2)
    




