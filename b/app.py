import streamlit as webbui
import pandas as analysis
import joblib

model = joblib.load("model.pkl")

webbui.title("Låneprediktion med AI")

webbui.write("Fyll i värden för att testa modellen.")

income = webbui.number_input("Inkomst", min_value=0.0, value=30000.0)
age = webbui.number_input("Ålder", min_value=18, value=30)
credit_score = webbui.number_input("Kreditpoäng", min_value=0.0, value=600.0)

if webbui.button("Gör prediktion"):

    # Skapa DataFrame med samma features som modellen tränades på
    input_data = analysis.DataFrame({
        "income": [income],
        "age": [age],
        "credit_score": [credit_score]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        webbui.success("Lånet godkänns")
    else:
        webbui.error("Lånet avslås")