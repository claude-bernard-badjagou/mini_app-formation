import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

st.title("Bienvenus sur votre application de la moyenne")
st.text("Cette application est un exemple de création d'application de data science avec streamlit")

#### ---------------PROJET-------------------####


# Prédire la moyenne en fonction des notes dans chaque compétence
# Prédire la note de SQL en fonction de la note en maths (regression linéaire multiple)

# Lecture du fichier
df = pd.read_csv("my_data_upgrage.csv")

if st.button("Voir un apperçu des données"):
	st.write(df)

st.markdown("#### Pour faire une prédiction, renseigner les champs ci-dessous avec vos ou les notes de l'étudiant. Enfin, cliquer sur le bouton *Prédire*")
# Feature sélection
# x = df[colonnes_numeriques] # les features 
x = df[['sql', 'excel', 'maths', 'python', 'poo', 'tableau',
         'data_exploration', 'data_manipulation', 'data_viz',
         'data_transformation', 'data_modelisation', 'data_deployement',
         'pack_office', 'result_presentation']]

# La cible à prédire
y = df['moyenne']

# Découpage des données
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instanciation du modèle
reg_line_multi = LinearRegression()

# Entrainement du modèle
reg_line_multi.fit(x_train, y_train)



# Evaluation du modèle
#mse = mean_squared_error(y_test, y_pred)
#st.write("Mean Squared Error:", mse)
#mae = mean_absolute_error(y_test, y_pred)
#st.write("Mean Absolute Error:", mae)
#r2 = r2_score(y_test, y_pred)
#st.write("R-squared:", r2)


# Prendre les données de l'utilisateur pour faire des prédictions
sql  = st.number_input("Note en sql sur 100 : ", 0, 100, 0)
excel = st.number_input("Note en excel sur 100 : ", 0, 100, 0)
maths = st.number_input("Note en maths sur 100 : ", 0, 100, 0)
python = st.number_input("Note en python sur 100 : ", 0, 100, 0)
poo = st.number_input("Note en poo sur 100 : ", 0, 100, 0)
tableau = st.number_input("Note en tableau sur 100 : ", 0, 100, 0)
data_exploration = st.number_input("Note en data exploration sur 100 : ", 0, 100, 0)
data_manipulation = st.number_input("Note en data manipulation sur 100 : ", 0, 100, 0)
data_viz = st.number_input("Note en data viz sur 100 : ", 0, 100, 0)
data_transformation = st.number_input("Note en data transformation sur 100 : ", 0, 100, 0)
data_modelisation = st.number_input("Note en data modelisation sur 100 : ", 0, 100, 0)
data_deployement = st.number_input("Note en data deployement sur 100 : ", 0, 100, 0)
pack_office = st.number_input("Note en pack office sur 100 : ", 0, 100, 0)
result_presentation = st.number_input("Note en result presentation sur 100 : ", 0, 100, 0)


notes = {"sql" : sql,
		"excel" :excel, 
		"maths" : maths,
		"python" : python, 
		"poo" : poo,
		"tableau" : tableau,
		"data_exploration" : data_exploration,
		"data_manipulation" : data_manipulation,
		"data_viz" : data_viz,
		"data_transformation" : data_transformation,
		"data_modelisation" : data_modelisation,
		"data_deployement" : data_deployement,
		"pack_office" : pack_office,
		"result_presentation" : result_presentation}
		

df_notes= pd.DataFrame(notes, index=[0])

st.success("Voici les vos informations renseignées pour faire la prédiction :")
st.write(df_notes)

if st.button("Prédire"):
	# Prédiction du modèle
	moyenne_prédite = reg_line_multi.predict(df_notes)
	st.write("Avec ses notes dans les différentes compétences, la moyenne sera : ", moyenne_prédite)
	if moyenne_prédite < 50:
		st.error("Vous n'aurez pas la moyenne de passage")
	if moyenne_prédite >= 50 and moyenne_prédite < 60 :
		st.write("Vous aurez la mention *Passable*")
	if moyenne_prédite > 60 and moyenne_prédite < 70 :
		st.write("Vous aurez la mention *Assez-bien*")
	if moyenne_prédite > 70 and moyenne_prédite < 80:
		st.info("Vous aurez la mention *Bien*")
	if moyenne_prédite > 80 and moyenne_prédite < 90 :
		st.success("Vous aurez la mention *Très Bien*")
	if moyenne_prédite >= 90:
		st.success("Vous aurez la mention *Excellente*")