import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
import base64
from streamlit_folium import st_folium

import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd

import numpy as np



#Model Type
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Evaluation de Model
from sklearn.metrics import accuracy_score

#Importer les NLTK et fonctions
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download("punkt")
import spacy

#NLP Features
from sklearn.feature_extraction.text import TfidfVectorizer
import string

punkt = string.punctuation


st.set_page_config(
    page_title="WELCOME TO Appli_ADN_Tourisme",
    page_icon="🧊",
    layout="wide"
)



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('arriere_plan.webp')
col1, col2, col3 = st.columns([1,6,1])

with col1:
    
    st.image("logo-adn-retina.png")
    
with col2:
    st.write("<h1 style='text-align: center; color: #67d4cc;'>Welcome to Appli_ADN_Tourisme 📲</h1>", unsafe_allow_html=True)

with col3:
    st.image("Data-Tourisme.jpg")

@st.cache(allow_output_mutation=True)
def get_my_data():
# chargement des donnees
    df=pd.read_csv(r"C:\Users\rachi\Desktop\Projet4\ADN_tourisme\df_400POI_clean.csv",error_bad_lines=False, engine="python")
    df.drop(df.iloc[:, 0:1], inplace=True, axis=1)
    df['description_fr']=df['description_fr'].fillna(df['description_fr_'])
    df= df.drop(df.iloc[:, [5,6,8]], axis=1)
    df=df.fillna('Information non renseignée')
    df=df.replace(['valeur manquante'], 'Information non renseignée')
    return df

   

          
  
st.write('')
st.write('')

selected = option_menu(None, ["HOME","POI_Comparator","Dashboard","Category_Predictor"], 
    icons=['bank2','binoculars-fill','bi-bar-chart-line-fill','cloud-check-fill'], default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#67d4cc"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "orange"},
        "nav-link-selected": {"background-color": "#f92a63ff"},
    })
    

if selected == "HOME":
    
    st.write("**ADN Tourisme est née le 11 mars 2020 du regroupement des trois fédérations historiques des acteurs institutionnels du tourisme, Offices de Tourisme de France, Tourisme & Territoires et Destination Régions.**")
    st.write('')
    
    st.markdown("Prennons quelques minutes pour regarder la video ci-joint afin de mieux comprendre les valeurs et les objectifs d'ADN Tourisme ⏰⏰📺📺⬇️⬇️")
    st.video('https://youtu.be/uM3QA9XQeLU')
    st.write('')
    st.markdown("<h6 style='text-align: justify; color: black;'>L'Appli_ADN_Tourisme permet de rechercher un établissement dans la base de données d'ADN Tourisme, et retourner l’ensemble des informations associées à cet établissement.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify; color: black;'>Le but est que les propriétaires  puissent accéder facilement aux informations concernant leur établissement.</h6>", unsafe_allow_html=True)
    

    st.caption("Application created by Lucile, Laurence, Sadja, Zied & Victor")

elif selected == "POI_Comparator":
    
    st.write('')
    st.caption(f'<h1 style="color:#67d4cc;font-size:24px;">{"Select your Point Of Interest 🤳"}</h1>', unsafe_allow_html=True)
    st.write('')

    with st.sidebar:

       
        df = get_my_data() 
        mylist_region=sorted(list(df['region_label'].unique()))
        option1 = st.selectbox( 
            '**Choose your region**',mylist_region)
        st.write('')

        for region in (mylist_region):
            if region==option1:
                st.image(region+".jpg")
                df=df[df['region_label']==option1]

                mylist_departement=sorted(list(df['departement_label'].unique()))        
                st.write("")
                option2 = st.selectbox(
                '**Choose your department**',mylist_departement)
                st.write('')

                
                for departement in (mylist_departement):
                    if departement==option2:
                        df=df[df['departement_label']==option2].reset_index().drop(columns='index')

                    
                        
                        mylist=sorted(list(df['POI_label'].unique()))
                        topic = st.selectbox('**Select your interest point**',mylist)

                        for index,label in enumerate(mylist):
                            if df['POI_label'][index]==topic:
                                df_bis=df.iloc[index]
                                df1 = pd.DataFrame(data=df_bis.index, columns=['Name'])
                                df2 = pd.DataFrame(data=df_bis.values, columns=['Information'])
                                df_select = pd.merge(df1, df2, left_index=True, right_index=True)
              
                                st.write("")
                                mylist_bis=sorted(list(df_select['Name']))
                                topic_ = st.multiselect('**Select your information**',mylist_bis)

                                
    col1, col2= st.columns(2)
        
    with col1:
        for index,label in enumerate(mylist):
            if label==topic:
                df_=df[df['POI_label']==topic].reset_index().drop(columns='index')
            
                map = folium.Map(location=[df_.lat, df_.long], zoom_start=14, control_scale=True)
                
                for index, location_info in df_.iterrows():
                    folium.Marker(
                    location=[location_info["lat"], location_info["long"]],
                    popup=folium.Popup ('vous êtes ici', max_width=300, show= True),
                    icon=folium.Icon(
                    icon='info-sign'
                    )
                ).add_to(map)

                for index, location_info in df.iterrows():
                    folium.Marker([location_info["lat"], location_info["long"]], popup=location_info["POI_label"],icon=folium.Icon(color='Brown', icon='flag', prefix='fa')).add_to(map)
                    folium.CircleMarker([df_.lat.mean(), df_.long.mean()],  radius =100).add_to(map)
                    folium.CircleMarker([df_.lat.mean(), df_.long.mean()], radius =15).add_to(map)
                st_data=st_folium(map, height=490, width=500)


    with col2:
        if topic_:
            filtered_df = df_select[df_select["Name"].isin(topic_)]
            dict_from_list = dict(zip(filtered_df["Name"].to_list(), filtered_df["Information"].to_list()))
            st.write(dict_from_list)
        
        else:
             
            st.spinner("Generating first plot...")
            #create a dataframe with region labels and number of POI by regions
            poi_counts = df.groupby(['region_label'])['region_label'].count().reset_index(name='count')
            
            # download the regions geodata
            url = 'https://france-geojson.gregoiredavid.fr/repo/regions.geojson'
            regions = gpd.read_file(url)
            regions = regions.rename(columns={'nom': 'region_label'})
            
            # merge the regions data with the POI counts
            regions = regions.merge(poi_counts, on='region_label', how='left')
            
            width, height = 800, 400
            
            fig_1 = folium.Figure(width=width, height = height)
            
            # create the map
            m = folium.Map(location=[48.85, 2.35], zoom_start=5)
            
            # create the choropleth
            folium.Choropleth(
                geo_data=regions,
                name='choropleth',
                data=regions,
                columns=['region_label', 'count'],
                key_on='feature.properties.region_label',
                fill_color='YlGn',
                fill_opacity=0.7,
                line_opacity=0.7,
                legend_name='Number of POIs'
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            fig_1.add_child(m)
            
            st_folium(fig_1,width=700, height=490)
                
elif selected == "Dashboard":
    st.write("")
    @st.cache(allow_output_mutation=True)
    def get_my_data_():
    # chargement des donnees
        data = pd.read_csv(r'C:\Users\rachi\Desktop\Projet4\ADN_tourisme\df_400POI_2.csv', low_memory = False)
        return data
    
    df = get_my_data_()

    ####################
    ### CLEANING ###
    ####################

    df['type_0'] = df['type_0'].str.replace("schema:", "")
    df['type_0'] = df['type_0'].str.replace("olo:", "")
    df['type_1'] = df['type_1'].str.replace("schema:", "")
    df['type_1'] = df['type_1'].str.replace("olo:", "")


    df['description_or_com_en'] = df['description_en'].fillna(df['description_en_'])
    df['description_or_com_fr'] = df['description_fr'].fillna(df['description_fr_'])
    df['description_or_com_ALL'] = df['description_or_com_en'].fillna(df['description_or_com_fr'])
    df['description_or_com_ALL'].isna().sum()
    df['new'] = df['description_or_com_ALL'].apply(lambda x:0 if pd.isna(x) else 1)
    df['new'].value_counts()


    ####################
    ### SHAPEMAP ###
    ####################

    col1, col2 = st.columns(2)

    with col1:
        st.spinner("Generating first plot...")
        
        top_10_poi = df.groupby(by='type_0')['POI_label'].count().sort_values(ascending=False).head(10)
        total = top_10_poi.sum()

        # Plot the bar chart
        fig_3 = px.bar(top_10_poi, x=top_10_poi.values[::-1], y=top_10_poi.index[::-1],
                       color_discrete_sequence =['#003366']*len(top_10_poi), 
                    labels={'x': 'Type 0', 'y': 'Number of POIs'}, orientation = 'h',title="<b>Top 10 by category</b>")
        fig_3.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig_3.update_layout(title_font_color="#67d4cc",
                            title_x=0.5)

        # Add annotations with the percentage values
        annotations = []
        for i, v in enumerate(top_10_poi.values[::-1]):
            annotations.append(dict(x=(v + 0) / 2, y=top_10_poi.index[::-1][i], 
                                    text=str(round(v/total*100, 1)) + "%", 
                                    showarrow=False, font=dict(size=12, color = 'white')))

        fig_3.update_layout(annotations=annotations)

        # Fix x axis range
        fig_3.update_xaxes(fixedrange=True)
        
        st.plotly_chart(fig_3, use_container_width=True)
        
             
    ####################
    ### TREEMAP ###
    ####################

    with col2:
        
        # Aggregate the data by type_0 and type_1
        agg_df = df.groupby(['type_0', 'type_1'])['POI_label'].count().sort_values(ascending=False).reset_index()
        
        # Filter top 5 type_0 for lighter visualisation
        tree_map_df = agg_df[(agg_df['type_0'] == 'Accommodation')|(agg_df['type_0'] == 'LocalBusiness')|(agg_df['type_0'] == 'FoodEstablishment')|(agg_df['type_0'] == 'Event')|(agg_df['type_0'] == 'CulturalSite')]
        
        # plot the treemap
       
        fig_2 = px.treemap(tree_map_df, path=['type_0', 'type_1'], values='POI_label',color='POI_label',
                  color_continuous_scale='RdBu', 
                  title= "<b>Points of interest by category</b>")
        fig_2.update_layout(
        title_font_color="#67d4cc",
        title_x=0.5)
                  
        fig_2.update_layout(margin = dict(t=50, l=25, r=25, b=25))

        

        st.plotly_chart(fig_2,width=500, height=500)



    col3, col4 = st.columns(2)

    with col3:
        # no description by info supplier
        no_desc_supplier = df[df['new'] == 0].groupby(['Information_supplier'])['Information_supplier'].count().sort_values(ascending=False).head(10)

        # Calculate the total number of POI without description
        total_no_desc = len(df[df['new'] == 0])

        # Calculate the percentage of cities without descriptions
        no_desc_supplier_pct = no_desc_supplier / total_no_desc * 100

        # Plot the bar chart
        fig_5 = px.bar(no_desc_supplier_pct, x=no_desc_supplier_pct.values[::-1], y=no_desc_supplier_pct.index[::-1], 
                    color_discrete_sequence =['#850606']*len(no_desc_supplier_pct),
                    labels={'x': 'Percentage of cities', 'y': 'Region label'}, orientation='h',
                    title="<b>Absence of description by supplier</b>")
        fig_5.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig_5.update_layout(title_font_color="#67d4cc",
                            title_x=0.5)
        
        # Add annotations with the percentage values
        annotations = []
        for i, v in enumerate(no_desc_supplier_pct.values[::-1]):
            annotations.append(dict(x=(v + 0) / 2, y=no_desc_supplier_pct.index[::-1][i], text=str(round(v, 1)) + "%", showarrow=False, font=dict(size=12, color='white'), xanchor='center'))

        fig_5.update_layout(annotations=annotations)

        # Fix x axis range
        fig_5.update_xaxes(fixedrange=True)

        st.plotly_chart(fig_5, use_container_width=True,height=500,width=500)

    ####################
    ### TOP_10_CITIES ###
    ####################
        
    with col4:
        st.spinner("Generating first plot...")
        top_10_city = df.groupby(by='locality')['POI_label'].count().sort_values(ascending=False).head(10)
        total_4 = top_10_city.sum()

        # Plot the bar chart
        fig_4 = px.bar(top_10_city, x=top_10_city.values[::-1], y=top_10_city.index[::-1],
                       color_discrete_sequence =['#FFA07A']*len(top_10_poi), 
                       labels={'x': 'locality', 'y': 'Number of cities'}, orientation = 'h',title="<b>Top 10 by cities</b>")
        fig_4.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        fig_4.update_layout(
            title_font_color="#67d4cc",
            title_x=0.5)
        # Add annotations with the percentage values
        annotations = []
        for i, v in enumerate(top_10_city.values[::-1]):
            annotations.append(dict(x=(v + 0) / 2, y=top_10_city.index[::-1][i], text=str(round(v/total_4*100, 1)) + "%", showarrow=False, font=dict(size=12, color='white'), xanchor='center'))
        
        fig_4.update_layout(annotations=annotations)
        
        # Fix x axis range
        fig_4.update_xaxes(fixedrange=True)
        
        st.plotly_chart(fig_4, use_container_width=True)

    ####################
    ### NO DESCRIPTION ###
    ####################
    

    
                                              

else:
    with st.sidebar:
    
        st.markdown("<h2 style='text-align: left; color: white;'>Machine learning time 📱</h2>", unsafe_allow_html=True)
                    #st.subheader("*This application identifies the category of a Place of Interest*")
            
            
        st.markdown("<h4 style='text-align: justify; color: black;'>The objective of this machine learning model is to verify the category and subcategories of a point of interest according to DATAtourism Ontology</h4>", unsafe_allow_html=True)
        st.write("")
        
        st.markdown("<h4 style='text-align: justify; color: black;'>The category is predicted using the establishment's name and description as input features for a supervised machine learning model (Logistic Regression)</h4>", unsafe_allow_html=True)
        st.write("")
        
        st.write("<h6 style='text-align: justify; color: black;'>This prototype has been deployed for accommadation and foodestablishment categories</h6>", unsafe_allow_html=True)
        
        st.image("Gif.gif")

    nlp = spacy.load('fr_core_news_md') 

    @st.cache(allow_output_mutation=True)
    def dataframe(url_csv):
        df = pd.read_csv(url_csv)
        return df
    
    #Tokenizing and Suppression of stopwords
    def clean_lemma (text):
        
        for punc in list(string.punctuation):
            if punc in text:
                text = text.replace(punc,'')
        text = nltk.word_tokenize(text.lower())
        #print(text)
        tokens_clean = []
        for words in text:
            if words not in nltk.corpus.stopwords.words("french"):
                tokens_clean.append(words)
        clean_text = ' '.join(tokens_clean)
        #nlp = spacy.load('en_core_web_sm')
        sent_tokens = nlp(clean_text)
        clean_text_lemma =[word.lemma_ for word in sent_tokens]
        lemma_text = " ".join(clean_text_lemma)
        return str(lemma_text)
    
        #Function for evaluating the Logistic Regression prediction model

    def give_category (my_establishment, my_description):
        my_data = my_establishment+' '+my_description
        my_data_ = clean_lemma(my_data)
        tfidf = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2))
        features = tfidf.fit(X_train['POI_label_desc_fr_lemma'])
        my_data_pred = tfidf.transform([my_data_])
        return modelLR.predict(my_data_pred)
    
    df_nouvelle_aquitaine=dataframe(r"C:\Users\rachi\Desktop\Projet4\ADN_tourisme\df_nouvelle_aquitaine_lemma.csv")

    # Training and Testing Sets
    X_category = df_nouvelle_aquitaine[['POI_label_desc_fr_lemma']]
    y_category = df_nouvelle_aquitaine['type_0']
    X_train, X_test, y_train, y_test = train_test_split(X_category, y_category, train_size = 0.80, random_state = 42, shuffle = True)


    #ML Modeling and TFIDF Vectorizing
    tfidf = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2))
    features = tfidf.fit(X_train['POI_label_desc_fr_lemma'])
    X_train_CV = tfidf.transform(X_train['POI_label_desc_fr_lemma'])
    X_test_CV = tfidf.transform(X_test['POI_label_desc_fr_lemma'])


    modelLR = LogisticRegression(max_iter=200, multi_class='ovr', solver='lbfgs')
    modelLR.fit(X_train_CV, y_train)

    y_pred =modelLR.predict(X_test_CV)
    accuracy = accuracy_score(y_test, y_pred)


    #Machine Learning Results
    st.markdown("<h4 style='text-align: center; color: #67d4cc;'>Verify the category of the Point Of Interest 📱</h4>", unsafe_allow_html=True)
    st.write("")
    
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("ML.png")

    with col3:
        st.write("")
    
    st.caption("<h6 style='text-align: center; color: black;'>Please,Enter the name and description of your establishment in French</h2>", unsafe_allow_html=True)
    

        
    col1,col2 = st.columns(2)
    with col1:
        my_establishment = st.text_input('Name:')
    with col2:
        my_description = st.text_input('Description :')

    category_prediction = give_category(my_establishment, my_description)
    st.write("") 

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")

    with col2:
        if st.button('Click to see the category'):
            st.write("")
            st.markdown("<h4 style='text-align: justify; color: #67d4cc;'>The main category of the POI is:</h4>", unsafe_allow_html=True)
            
            #category_prediction
            st.subheader(f"{category_prediction}")

    with col3:
        st.write("")  
    
        #st.write(f"This is the predicted category {category_prediction}") 

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("equipe.png")

    with col3:
        st.write("")