import streamlit as st
import pandas as pd
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="CreditAI", 
                   page_icon="💳", 
                   layout="wide")


st.sidebar.markdown("""
<style>
@keyframes pulse {
  0% {transform: scale(1); opacity: 0.6;}
  50% {transform: scale(1.05); opacity: 1;}
  100% {transform: scale(1); opacity: 0.6;}
}

.sidebar-status-container {
  display: flex;
  justify-content: center;
  gap: 10px;
  flex-wrap: nowrap;
  margin-top: 10px;
}

.status-badge {
  width: 36px;              /* Largeur = hauteur */
  height: 36px;             /* Taille circulaire */
  border-radius: 50%;       /* Cercle parfait */
  font-size: 18px;          /* Taille texte/picto */
  font-weight: 700;
  color: white;
  display: flex;            /* Centrer texte verticalement */
  align-items: center;      
  justify-content: center;  /* Centrer texte horizontalement */
  box-shadow: 0 3px 8px rgba(0,0,0,0.15);
  cursor: default;
  user-select: none;
  transition: all 0.3s ease-in-out;
  text-align: center;
  line-height: 1;
}

/* Vert clignotant */
.badge-green {
  background: linear-gradient(145deg, #2ecc71, #27ae60);
  animation: pulse 2s infinite;
}

/* Rouge statique */
.badge-red {
  background: linear-gradient(145deg, #e74c3c, #c0392b);
}
</style>

<div class="sidebar-status-container">
  <div class="status-badge badge-green" title="Connecté"></div>
  <div class="status-badge badge-red" title="Alerte"></div>
</div>
""", unsafe_allow_html=True)



#color_discrete_sequence=["#C0E0CD"])
st.markdown("""
<style>
.box {
    background-color: #ff9c33;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 30px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}
.boxe {
    background-color: #8186DE; 
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 30px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}          
h3 {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Lottie animation setup ---
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv('donnees_traitees.csv')
    return data
data = load_data()

@st.cache_data
def load_data1():
    data1 = pd.read_csv('AER_credit_card_data.csv')
    return data1
data1 = load_data1()

with st.sidebar:
    selection = option_menu(
        menu_title="Menu",
        options=["Accueil","Donnees Brutes", "Statistiques Descriptives", "Visualisation","Prediction","À propos","Contact Us"],
        icons=["house","square", "info", "circle","file-earmark-text", "info-circle","envelope"],
        menu_icon="globe",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "blue", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e0e0e0"
            },
            "nav-link-selected": {
                "background-color": "#ff9c33",
                "color": "white",
                "font-weight": "bold"
            }
        }
    )

            # Charger le fichier Lottie localement
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_animation = load_lottie_file("Animation1.json")
lot2= load_lottie_file("Animation2.json")
lot3= load_lottie_file("Animation3.json")

if selection == "Accueil":
    # Titre de l'application
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: sky-blue;'>💳 CreditAI</h1></div>""", unsafe_allow_html=True)




    col1, col2= st.columns(2)
    with col1:
        st.markdown("""
    <div class="box">
    <h3>💡Bienvenue dans CreditAI </h3>
    </div>
    """, unsafe_allow_html=True)
        
        # Affichage de l'animation Lottie centrée
        st_lottie(lottie_animation, speed=1, width=300, height=200, key="lottie1")

        st.markdown("""
    <div class="box">
    <h4>💡 Contrôlez votre avenir financier.</h4>
    </div>
    """, unsafe_allow_html=True)
        st_lottie(lot2, speed=1, width=300, height=200, key="lottie4")
    with col2:
        st_lottie(lot2, speed=1, width=300, height=200, key="lottie2")

        st.markdown("""
    <div class="box">
    <h4>💡 Anticipez vos dépenses.</h4>
    </div>
    """, unsafe_allow_html=True)
        st_lottie(lottie_animation, speed=1, width=300, height=200, key="lottie3")
    
    st.info("🟨 CreditIA est une application intelligente qui vous aide à prédire vos dépenses mensuelles en fonction de vos habitudes d’utilisation de carte de crédit. Grâce à l’intelligence artificielle, vous obtenez des estimations personnalisées pour mieux gérer votre budget.")

elif selection== "Donnees Brutes":
    st.markdown("""
    <div class="box">
    <h3>📊 Affichage des donnees </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="boxe"">
    <h3>Données Brutes sans traitement </h3>
    </div>""", unsafe_allow_html=True)
    st.info("Les données brutes sont issues du jeu de données AER Credit Card Data. Elles contiennent des informations sur les utilisateurs de cartes de crédit, y compris leur âge, revenu, dépenses mensuelles, et d'autres caractéristiques.")
    st.dataframe(data1.style.background_gradient(cmap='Blues'), height=300)

    st.markdown("""<div class="boxe", style="text-align: center;">
    <h3>Explications des Variables </h3>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center;">
    <p>card : Indique si la personne possède une carte de crédit (yes ou no).</p>
    <p>reports : Nombre de signalements négatifs dans l’historique de crédit (ex : retards de paiement).</p>
    <p>age : Age du demandeur (en années).</p>
    <p>income : Revenu annuel du demandeur (en milliers de dollars).</p>
    <p>share : Proportion du revenu mensuel dépensée avec une carte de crédit.</p>
    <p>expenditure : Dépenses mensuelles moyennes (en dollars).</p>
    <p>owner : Indique si la personne possède un bien immobilier (yes ou no).</p>
    <p>selfemp : Indique si la personne est travailleuse indépendante (yes ou no).</p>
    <p>dependents : Nombre de personnes à charge.</p>
    <p>months : Ancienneté du dossier de crédit, en mois.</p>
    <p>majorcards : Nombre de cartes de crédit détenues auprès de grandes institutions.</p>
    <p>active : Nombre d’achats réalisés avec une carte de crédit lors du dernier mois </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="boxe", style="text-align: center;">
    <h3>Traitements Effectues </h3>
    </div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Donnees Dupliquees", f"{data.duplicated().sum()}")
    col2.metric("Donnees Absentes", f"{data.isnull().sum().sum()}")
    col3.metric("Dimension", f"{data.shape[0]} lignes, {data.shape[1]} colonnes")
    st.markdown("""
    <div style="text-align: center;">
    <p>card : Encodage categotielle.</p>
    <p>age : Changement de type en entier.</p>
    <p>owner : Encodage categotielle.</p>
    <p>selfemp : Encodage categotielle.</p>
    <p>La Gestion des Outliers.</p>
    <p>Verification des donnees dupliquees.</p>
    <p>Verification des donnees absentes.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="boxe", style="text-align: center;">
    <h3>Données Brutes Traitees </h3>
    </div>""", unsafe_allow_html=True)
    st.dataframe(data.style.background_gradient(cmap='Blues'), height=300)  





elif selection == "Statistiques Descriptives":
    #st.title(' Statistiques')
    st.markdown("""<div class="box">
    <h3>🔢 Statistiques de chaque variable. </h3>
    </div>""", unsafe_allow_html=True)
    st.dataframe(data.describe().style.background_gradient(cmap='Greens'))


elif selection== "Visualisation":
    #st.title('📈 Visualisation')
    st.markdown("""<div class="box", style="text-align: center;">
    <h3> 📈 Visualisation </h3>
    </div>""", unsafe_allow_html=True)
    # Sélection des colonnes pour l'analyse
    cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Distribution", "Relations", "Corrélations"])

    with tab1:
        st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Distribution des Variables </h3>
        </div>""", unsafe_allow_html=True)
        #st.subheader("Distribution des Variables")
        selected_col = st.selectbox("Sélectionnez une variable", cols, index=cols.index('expenditure') if 'expenditure' in cols else 0)
        
        fig = px.histogram(data, x=selected_col, nbins=50, 
                        title=f'Distribution de {selected_col}',
                        color_discrete_sequence=["#FAAC63"])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        #st.subheader("Relations entre Variables")
        st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Relations entre Variables </h3>
        </div>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox('Variable X', cols, index=cols.index('income') if 'income' in cols else 0)
        with col2:
            y_axis = st.selectbox('Variable Y', cols, index=cols.index('expenditure') if 'expenditure' in cols else 1)
        
        fig = px.scatter(data, x=x_axis, y=y_axis, 
                        color='card' if 'card' in data.columns else None,
                        title=f'{y_axis} vs {x_axis}',
                        trendline="lowess")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        #st.subheader("Matrice de Corrélation")
        st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Matrice de correlation </h3>
        </div>""", unsafe_allow_html=True)
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        corr = numeric_data.corr()
        
        fig = px.imshow(corr,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        title='Matrice de Corrélation',
                        width=800, height=800)
        st.plotly_chart(fig, use_container_width=True)


elif selection== "Prediction":
    #st.title('Faire une prediction')
    st.markdown("""<div class="box", style="text-align: center;">
        <h3> 🧮 Prediction </h3>
        </div>""", unsafe_allow_html=True)
    # Sidebar pour les options
    st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Choix du Modele </h3>
        </div>""", unsafe_allow_html=True)
    st.info("Sélectionnez le modèle de prédiction que vous souhaitez utiliser pour estimer les dépenses mensuelles.")
    st.info("Les modèles disponibles sont : Random Forest, Linear Regression et Gradient Boosting.")
    selected_model = st.selectbox('Modèle de prédiction', 
                                     ['Random Forest', 'Linear Regression', 'Gradient Boosting'],
                                     index=0)
    # Préparation des données
    X = data.drop(['expenditure'], axis=1)
    y = data['expenditure']

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Sélection du modèle
    if selected_model == 'Random Forest':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    elif selected_model == 'Linear Regression':
        from sklearn.linear_model import LinearRegression
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=42))
        ])

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Evaluation du modele </h3>
        </div>""", unsafe_allow_html=True)
    st.info("Les métriques suivantes sont calculées pour évaluer la performance du modèle de prédiction choisi.")

    # Affichage des métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
    <div style="background-color: #ff9c33; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h4 style="color:#e0e0e0; font-weight: bold;text-align:center;">RMSE</h4>   
                 </div>  """, unsafe_allow_html=True)

        st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                {rmse:.2f}
            </div>
            """,
            unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background-color: #ff9c33; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h4 style="color:#e0e0e0; font-weight: bold;text-align:center;">MAE</h4>   
                 </div>  """, unsafe_allow_html=True)

        st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                {mae:.2f}
            </div>
            """,
            unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="background-color: #ff9c33; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h4 style="color:#e0e0e0; font-weight: bold;text-align:center;">R² Score</h4>   
                 </div>  """, unsafe_allow_html=True)

        st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                {r2:.2f}
            </div>
            """,
            unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div style="background-color: #ff9c33; border-radius: 15px; padding: 20px; 
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1); margin: 0px 0;">
        <h4 style="color:#e0e0e0; font-weight: bold;text-align:center;">Score Test</h4>   
                 </div>  """, unsafe_allow_html=True)

        st.markdown(f"""<div style='background-color:#fff3cd;border-left:8px solid #fd7e14;
                padding:20px;
                margin:20px 0;
                border-radius:10px;
                font-size:75px;
                font-weight:bold;
                color:#7a4600;'>
                {model.score(X_test, y_test):.1%}
            </div>
            """,
            unsafe_allow_html=True)


    st.info("RMSE (Root Mean Squared Error) : Mesure de l'erreur quadratique moyenne entre les valeurs prédites et réelles.")
    st.info("MAE (Mean Absolute Error) : Mesure de l'erreur absolue moyenne entre les valeurs prédites et réelles.")    
    st.info("R² Score : Indique la proportion de la variance des dépenses mensuelles expliquée par le modèle.")
    st.info("Score du modèle : Indique la performance globale du modèle sur les données de test.")
    # Visualisation des prédictions vs réelles
    st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Comparaison Prédictions vs Réelle </h3>
        </div>""", unsafe_allow_html=True)
    #st.subheader('Comparaison Prédictions vs Réelles')
    fig = px.scatter(x=y_test, y=y_pred, 
                    labels={'x': 'Valeurs Réelles', 'y': 'Prédictions'},
                    title='Prédictions vs Valeurs Réelles',
                    trendline="ols")
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max(),
                line=dict(color="Red", width=2, dash="dot"))
    st.plotly_chart(fig, use_container_width=True)

    # Importance des caractéristiques
    if hasattr(model.named_steps[model.steps[-1][0]], 'feature_importances_'):

        st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> Importance des variables</h3>
        </div>""", unsafe_allow_html=True)
        #st.subheader('Importance des Caractéristiques')
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.named_steps[model.steps[-1][0]].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                    title='Importance des Caractéristiques',
                    color='Importance',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # Prédiction pour de nouvelles données
    st.markdown("""<div class="boxe", style="text-align: center;">
        <h3> 🎯 Faire une Nouvelle Prédiction </h3>
        </div>""", unsafe_allow_html=True)
    
    st.info("Utilisez le formulaire ci-dessous pour entrer les caractéristiques d'un nouveau client et prédire ses dépenses mensuelles.")
    #st.header('🎯 Faire une Nouvelle Prédiction')


    input_data = {}
    cols = st.columns(3)
        
        # Liste des colonnes catégorielles
    categorical_features = ['card', 'reports', 'owner', 'selfemp', 'dependents', 'majorcards']
        
    for i, feature in enumerate(X.columns):
        if feature in categorical_features:
            options = sorted(data[feature].unique())
            input_data[feature] = cols[i % 3].selectbox(
                label=feature,
                options=options,
                key=f"{feature}_selectbox"
                )
        elif data[feature].dtype == 'int64':
            input_data[feature] = cols[i % 3].number_input(
                label=feature,
                min_value=int(data[feature].min()),
                max_value=int(data[feature].max()),
                    value=int(data[feature].median()),
                    step=1,
                    key=f"{feature}_intinput"
                )
        else:
            input_data[feature] = cols[i % 3].number_input(
                label=feature,
                    min_value=float(data[feature].min()),
                    max_value=float(data[feature].max()),
                    value=float(data[feature].median()),
                    key=f"{feature}_floatinput"
                )

        
    if st.button('Prédire les Dépenses'):
        input_df = pd.DataFrame([input_data])
            
        prediction = model.predict(input_df)

        lower_bound = prediction[0] - rmse
        upper_bound = prediction[0] + rmse

        st.markdown(
                    f"""
                    <div style="margin-top: 20px; margin-bottom: 20px;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 22px; text-align: left;">
                            <tr style="background-color: #bbd99d; color: #1b5e20;">
                                <th style="padding: 12px;">Dépenses prédites</th>
                                <td style="padding: 12px; font-weight: bold;">{prediction[0]:.2f} USD</td>
                            </tr>
                            <tr style="background-color: #93f2fd; color: #0d47a1;">
                                <th style="padding: 12px;">Intervalle de confiance (approximatif)</th>
                                <td style="padding: 12px; font-weight: bold;">{lower_bound:.2f} - {upper_bound:.2f}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            



elif selection== "À propos":
    #st.title('A propos')
    st.markdown("""<div class="box", style="text-align: center;">
        <h3> ℹ️ À propos </h3>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #C0E0CD; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🧾 Données utilisées</h3>
        <p style="color: #34495e; font-size: 16px;">Le modèle est entraîné sur le jeu de données AER Credit Card Data, qui comprend :</p>
        <ul style="color: #34495e; font-size: 16px;">
            <li>L'âge, le revenu et les dépenses mensuelles,</li>
            <li>Le statut de propriétaire et d’emploi indépendant,</li>
            <li>L’historique de crédit (nombre de rapports négatifs),</li>
            <li>Le nombre de personnes à charge, la durée du crédit, etc..</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #C0E0CD; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🧠 Fonctionnement</h3>
        <p style="color: #34495e; font-size: 16px;">
Notre application repose sur un algorithme de classification binaire. En entrant les informations d'un individu, l'application prédit la probabilité qu'il ou elle possède une carte de crédit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #C0E0CD; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🔍 Objectif</h3>
        <p style="color: #34495e; font-size: 16px;">L’objectif principal est d’illustrer comment les données personnelles et financières peuvent être utilisées pour :</p>
        <ul style="color: #34495e; font-size: 16px;">
            <li>Aider les institutions financières à évaluer les profils clients,</li>
            <li>Sensibiliser les utilisateurs à leur profil de crédit,</li>
            <li>Montrer l’application de l’intelligence artificielle à la finance.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #C0E0CD; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🛠 Technologies</h3>
        <ul style="color: #34495e; font-size: 16px;">
            <li>Python & Streamlit pour l’interface,</li>
            <li>Pandas pour la gestion des données,</li>
            <li>Algorithme de prediction: Random Forest, Linear Regression, Gradient Boosting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif selection== "Contact Us":
    st.markdown("""<div class="box", style="text-align: center;">
        <h3>  📧 Contactez-nous </h3>
        </div>""", unsafe_allow_html=True)
    st.warning("Si vous avez des questions ou suggestions, envoyez-nous un e-mail.")

    col1, col2= st.columns(2)
    with col1:
        # Affichage de l'animation Lottie centrée
        st_lottie(lot3, speed=1, width=380, height=380, key="lottie1")
    with col2:
        st.markdown("""
    <div style="background-color: #C0E0CD; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">📧 Email:        hiroshiyewo@gmail.com</h3>
        <h3 style="color:#2c3e50; font-weight: bold;">Noms:         Yewo Feutchou Hiroshi</h3>
        <h3 style="color:#2c3e50; font-weight: bold;">Statut: Master 2 IABD</h3>
        <h3 style="color:#2c3e50; font-weight: bold;">Institution: Keyce Informatique</h3>
        <h3 style="color:#2c3e50; font-weight: bold;">Projet: Application de prédiction de comportement lié à la carte de crédit</h3>
        
    </div>""", unsafe_allow_html=True)