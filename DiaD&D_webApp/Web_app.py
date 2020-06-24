#Written by Ali Mehrnezhad, June 2020
#DiaD&D(Diabetes Detection & Direction) WebApp for prediction of Diabetes and Prediabetes risk

#Importing libraries
import sys
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import altair as alt
import streamlit as st
import pickle
import numpy as np
from sklearn import preprocessing
from PIL import Image, ImageDraw, ImageFont 

st.title('Diabetes Detection & Direction')

if sys.version_info[0] < 3:
    reload(sys) # noqa: F821 pylint:disable=undefined-variable
    sys.setdefaultencoding("utf-8")
    
st.sidebar.title(' Inputs:')

#Inputs to be collected from user
Age=st.sidebar.slider("How old are you?",18,100,40)
Height=st.sidebar.slider("Height (cm)?",120,210,180)
Weight=st.sidebar.slider("Weight (Kg)?",25,200,85)
BMI=Weight/((Height)*(Height))*10000
Waist=st.sidebar.slider("Waist circ.(cm)?",55,170,90)
Thigh=st.sidebar.slider("Thigh circ.(cm)?",25,95,50)           
Calf=st.sidebar.slider("Calf circ.(cm)?",20,62,35)
Arm=st.sidebar.slider("Arm circ.(cm)?",17,60,32)
            
Gender = st.sidebar.selectbox('Gender? (0: Male, 1: Female)',[0,1],0)
Fa_hist = st.sidebar.selectbox('Family History of Diabetes? (0: No, 1: Yes)',[0,1],0)
BP_hist = st.sidebar.selectbox('Blood Pressure History(yourself)? (0: No, 1: Yes)',[0,1],0)
SMQ = st.sidebar.selectbox('Smoked >100 Cigarette? (0: No, 1: Yes)',[0,1],0)
HHS = st.sidebar.selectbox('Houshold size >2 people? (0: No, 1: Yes)',[0,1],0)

#Collecting datasets
X=pd.read_pickle('X_train.pickle')
y=pd.read_pickle('Y_train.pickle')
X_bc=pd.read_pickle('X_bc.pickle')

                   
# Reading Models for predicting body composition
with open('Total_Fat_model.pkl', 'rb') as file:  
    Total_Fat_model = pickle.load(file)   
with open('Trunk_Fat_model.pkl', 'rb') as file:  
    Trunk_Fat_model = pickle.load(file)    
with open('Arm_Fat_model.pkl', 'rb') as file:  
    Arm_Fat_model = pickle.load(file)     
with open('Leg_Fat_model.pkl', 'rb') as file:  
    Leg_Fat_model = pickle.load(file)     
with open('Total_Lean_model.pkl', 'rb') as file:  
    Total_Lean_model = pickle.load(file) 


#Predicting Body composition of user  
input_bc = [[Age, Weight, Height, BMI, Waist, Thigh, Arm, Calf, Gender]]
df_bc=pd.DataFrame(input_bc, columns = ['RIDAGEYR', 'bmxwt', 'bmxht', 'bmxbmi', 'bmxwaist', 'BMXTHICR',
       'BMXARMC', 'BMXCALF', 'RIAGENDR_2'])
df_bc_ar= preprocessing.StandardScaler().fit(X_bc).transform(df_bc)
total_fat=(Total_Fat_model.predict(df_bc_ar)/1000).item()
trunk_fat=(Trunk_Fat_model.predict(df_bc_ar)/1000).item()
arm_fat=(Arm_Fat_model.predict(df_bc_ar)/1000).item()
leg_fat=(Leg_Fat_model.predict(df_bc_ar)/1000).item()
lean_mass=(Total_Lean_model.predict(df_bc_ar)/1000).item()
fat_ratio=trunk_fat/total_fat

#Predicting Body composition of user  
df_bc_ar_to= preprocessing.StandardScaler().fit(X_bc).transform(X_bc)
total_fat_to=(Total_Fat_model.predict(df_bc_ar_to)/1000)
trunk_fat_to=Trunk_Fat_model.predict(df_bc_ar_to)/1000
arm_fat_to=(Arm_Fat_model.predict(df_bc_ar_to)/1000)
leg_fat_to=(Leg_Fat_model.predict(df_bc_ar_to)/1000)
lean_mass_to=(Total_Lean_model.predict(df_bc_ar_to)/1000)
fat_ratio_to=trunk_fat_to/total_fat_to

#Inputs for predicting Diabetes and Prediabetes risk  
inputs = [[Age, Weight, Height, BMI, Waist, total_fat, trunk_fat, fat_ratio,arm_fat, leg_fat
           ,lean_mass, Gender,Fa_hist, BP_hist, SMQ, HHS ]] 
dff = pd.DataFrame(inputs, columns = ['RIDAGEYR', 'bmxwt', 'bmxht', 'bmxbmi', 'bmxwaist', 'tot_fat', 'tr_fat',
   'tr_tot_fat', 'arm_fat', 'leg_fat', 'lean_fat', 'RIAGENDR_2',
   'Fa_hist_1', 'Bl_pres_1', 'SMQ_1', 'HHS_1'])
dff_ar= preprocessing.StandardScaler().fit(X).transform(dff)

# Reading Model for predicting Diabetes and Prediabetes risk
with open('LR_model1.pkl', 'rb') as file:  
    Pickled_RF_Model = pickle.load(file)

Risk=Pickled_RF_Model.predict_proba(dff_ar)[:,1]
XX= preprocessing.StandardScaler().fit(X).transform(X)
rf_probs = Pickled_RF_Model.predict_proba(XX)
rf_diab_prob=rf_probs[:,1]

# Visualization of user diabetes risk  
st.subheader('Your Diabetes Risk Compared to the US Population')
Risk=np.round_(Risk,decimals = 2)
Risk=(Risk).item()
st.markdown(f">### Your Risk of Diabetes: {Risk*100}% \n {Risk*100} out of 100 people who are similar to you are diabetic or prediabetic ")

#Histogram of diabetes risk over the whole population
DIS=pd.DataFrame()
hist_values = np.histogram(rf_diab_prob,bins=100, range=(0,1),normed=True)

DIS['Freq']=pd.Series(hist_values[0])
DIS['Prob']=pd.Series(hist_values[1])
DIS['Risk'] = Risk

chart_one = alt.Chart(DIS).mark_bar().encode(
    alt.X('Prob:Q', scale=alt.Scale(domain=(0.1, 0.9)), axis=alt.Axis(format='%', title='Diabetes Risk')),
    alt.Y("Freq",axis=alt.Axis(title='Relative Frequency'))
)

chart_two = alt.Chart(DIS).mark_rule(color='red').encode(x='Risk',size=alt.value(2))

(chart_one+chart_two).properties(width=600)
st.altair_chart(chart_one+chart_two)

###Visualization of body composition
st.subheader('Your Regional Body Fat:')
st.markdown('(Red Oval: High Regional Fat)')
output_bc=[[total_fat, trunk_fat ,arm_fat, leg_fat, fat_ratio, lean_mass ]]           
output_bc=pd.DataFrame(output_bc, columns = ['Total Fat(kg)','Trunk Fat(kg)','Arm Fat(kg)','Leg Fat(kg)','Trunk fat/Total fat','Lean mass(kg)']) 
#st.write(output_bc) -> To show as a table 

image = Image.open("body.jpg")
draw  = ImageDraw.Draw(image)
if trunk_fat > np.percentile(trunk_fat_to, 50):
    fill1='red'
else:
    fill1='blue'
    
if arm_fat > np.percentile(arm_fat_to, 50):
    fill2='red'
else:
    fill2='blue'
    
if leg_fat > np.percentile(leg_fat_to, 50):
    fill3='red'
else:
    fill3='blue'    
    
draw.ellipse((65, 75, 95, 145), fill=fill1, outline =None)  #Trunk
draw.ellipse((40, 80, 50, 100), fill=fill2, outline =None)  #Arm
draw.ellipse((108, 80, 118, 100), fill=fill2, outline =None) #Arm
draw.ellipse((58, 165, 70, 200), fill=fill3, outline =None)  #Leg
draw.ellipse((94, 165, 106, 200), fill=fill3, outline =None)  #Leg

st.image(image)

st.markdown(f"> ##### Total body fat:  {np.round_(total_fat, decimals=2)} kg \n > ##### Total body lean mass:  {np.round_(lean_mass, decimals=2)} kg \n > ##### Total Leg fat :  {np.round_(leg_fat, decimals=2)} kg \n > ##### Upper body fat :  {np.round_(trunk_fat, decimals=2)} kg")

#Show a warning because of high upper body fat
if fat_ratio>=0.52:
    st.markdown(" #### Your high upper body fat puts you at more risk for diabetes.")

#Decreasing weight and new prediction of diabetes risk   
st.subheader('Set Goals to Reduce Your Risk:')
st.markdown("#### Weight loss in Kg")

W_loss = st.slider("",0,50,0)

#By weight loss some other body composition and measurments are modified by a multiplier based on linear regression with weight
input_rec= [[Age, (Weight-W_loss),Height,(Weight-W_loss)/Weight*BMI,(Waist-(1.17*W_loss)),
             total_fat-(0.35*W_loss), trunk_fat-(0.17*W_loss), fat_ratio,arm_fat-(0.02*W_loss),
             leg_fat-(0.06*W_loss),lean_mass-(0.6*W_loss), Gender,Fa_hist, BP_hist, SMQ, HHS ]]
df_rec= pd.DataFrame(input_rec, columns = ['RIDAGEYR','bmxwt','bmxht','bmxbmi','bmxwaist','tot_fat','tr_fat',
'tr_tot_fat', 'arm_fat', 'leg_fat', 'lean_fat', 'RIAGENDR_2',
'Fa_hist_1', 'Bl_pres_1', 'SMQ_1', 'HHS_1'])
df_rec_ar= preprocessing.StandardScaler().fit(X).transform(df_rec)
Risk_rec=Pickled_RF_Model.predict_proba(df_rec_ar)[:,1]
Risk_rec=(Risk_rec).item()
Risk_rec=np.round_(Risk_rec,decimals = 2)

st.markdown(f">#### If you lose {W_loss} kg weight, your diabetes risk will reduce to : {(Risk_rec)*100}%")

