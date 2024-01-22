import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import pickle 
import sklearn


# Load the saved machine learning model
# model_LR = pickle.load(open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\ai_elite_10\laptop_deployment\models\LR_model.sav","rb"))
# model_DT = pickle.load(open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\ai_elite_10\laptop_deployment\models\Decision_Tree_Model.sav","rb"))
model_KNN = pickle.load(open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\ai_elite_10\laptop_deployment\models\KNN_model.sav","rb"))
# model_RF = pickle.load(open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\ai_elite_10\laptop_deployment\models\RF_model.sav","rb"))
encoder_OHE=pickle.load(open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\ai_elite_10\laptop_deployment\models\encoder.sav","rb"))
std_scaler = pickle.load(open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\ai_elite_10\laptop_deployment\models\std_scaler.sav","rb"))


st.title('Laptop prediction')

# Define categories for select boxes
brand_names = ['ASUS','Lenovo', 'HP', 'DELL', 'acer','RedmiBook','MSI','Infinix','APPLE','realme','ALIENWARE',
                 'SAMSUNG','Ultimus','Vaio','GIGABYTE','Nokia']

processor_names =['Intel Core i5','Intel Core i3','Ryzen 9 Octa Core','Ryzen 7 Octa Core','Dual Core','Intel Core i7','Ryzen 5 Hexa Core','Ryzen 5 Quad Core','Intel Core i9','M1',
                  'M1 Pro','Intel Platinum','M2','Ryzen 3 Quad Core','Intel Celeron Quad Core','Qualcomm Snapdragon 7c','M1 Max','Ryzen 7 Quad Core','Ryzen 3 Hexa Core']
ram_names = ['4 GB','8 GB','16 GB','32 GB']
ram_type_names = ['DDR4','DDR5','LPDDR4X','Unified Memory','LPDDR5','LPDDR4','LPDDR3']
storage_names = ['512 GB SSD','1000 GB SSD','256 GB SSD','1000 GB HDD','64 GB EMMC','2000 GB SSD','128 GB SSD','128 GB EMMC','32 GB EMMC'] 
rating_names = [4.2, 4.3, 4.4, 4.1, 4.6, 4.5] 
# selected_model = [model_LR, model_DT, model_KNN, model_RF]

brand = st.selectbox('Select Brand', brand_names)
processor_name = st.selectbox('Select Processor Name', processor_names)
ram_gb = st.selectbox('Select RAM (in GB)', ram_names)
ram_type = st.selectbox('Select RAM_type',ram_type_names)
storage = st.selectbox('Select Storage Type', storage_names)
rating = st.selectbox('Select laptop with corresponding rating',rating_names)
rating = st.selectbox('Select Model to Predict',rating_names)

# if selected_model == 'Linear Regression':
#     selected_model = model_LR
# elif selected_model == 'Decision Tree':
#     selected_model = model_DT
# elif selected_model == 'K-Nearest Neighbors':
#     selected_model = model_KNN
# elif selected_model == 'Random Forest':
#     selected_model = model_RF

def predict_price(brand, processor_name, ram_gb, ram_type, storage,rating):
    processor_mapping = {'Dual Core':1,'Intel Celeron Quad Core':2,'Intel Platinum':3,'Intel Core i3':4,'Intel Core i5':5,'Intel Core i7':6,'Intel Core i9':7,'Ryzen 3 Quad Core':8,'Ryzen 5 Quad Core':9,'Ryzen 7 Quad Core':10,'Ryzen 3 Hexa Core':11,'Ryzen 5 Hexa Core':12,'Ryzen 7 Octa Core':13,'Ryzen 9 Octa Core':14,'Qualcomm Snapdragon 7c':15,'M1':16,'M2':17,'M1 Pro':18,'M1 Max':19}
    processor_name = processor_mapping.get(processor_name,0)

    ram_mapping = {'4 GB': 1, '8 GB': 2, '16 GB': 3, '32 GB': 4}
    ram_gb = ram_mapping.get(ram_gb, 0)

    ram_type_mapping = {'Unified Memory': 1, 'DDR4': 2, 'DDR5': 3, 'LPDDR3': 4, 'LPDDR4': 5, 'LPDDR4X': 6, 'LPDDR5': 7}
    ram_type = ram_type_mapping.get(ram_type, 0)

    storage_mapping = {'1000 GB HDD': 0, '128 GB SSD': 1, '256 GB SSD': 2, '512 GB SSD': 3, '1000 GB SSD': 4,
                       '2000 GB SSD': 5, '32 GB EMMC': 6, '64 GB EMMC': 7, '128 GB EMMC': 8}
    
    storage = storage_mapping.get(storage, 0) 

    brand_enc = encoder_OHE.transform([[brand]])[0]
    rating_scaling = std_scaler.transform([[rating]])[0]

    input_data = np.concatenate(([brand_enc, processor_name, ram_gb, ram_type, storage, rating_scaling]), axis=None)
    input_data = input_data.reshape(1, -1)

    # predicted_price = st.title(int(np.exp(model_LR.predict(input_data))))
    predicted_price = model_KNN.predict(input_data)
    return predicted_price[0]

if st.button('Predict Price'):
    predicted_price = predict_price(brand, processor_name, ram_gb, ram_type, storage,rating)
    st.write(f'Predicted Price: Rs: {predicted_price:.2f}/-')
