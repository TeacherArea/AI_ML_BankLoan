import streamlit as webbui
import pandas as analysis
import joblib

# Ladda modellerna
model = joblib.load("model.pkl")
compare_model = joblib.load("compare.pkl")

webbui.title("Låneprediktion med AI")

webbui.write("Jämför två AI-modeller för låneprediktion.")

income = webbui.number_input("Inkomst", min_value=0.0, value=30000.0)
age = webbui.number_input("Ålder", min_value=18, value=30)
credit_score = webbui.number_input("Kreditpoäng", min_value=0.0, value=600.0)

if webbui.button("Gör prediktion"):

    input_data = analysis.DataFrame({
        "income": [income],
        "age": [age],
        "credit_score": [credit_score]
    })

    original_prediction = model.predict(input_data)[0]

    compare_prediction = compare_model.predict(input_data)[0]

    webbui.subheader("Originalmodell")

    if original_prediction == 1:
        webbui.success("Lånet godkänns")
    else:
        webbui.error("Lånet avslås")

    webbui.subheader("Kompletterande modell")

    if compare_prediction == 1:
        webbui.success("Lånet godkänns")
    else:
        webbui.error("Lånet avslås")

    webbui.subheader("Jämförelse")

    if original_prediction == compare_prediction:
        webbui.info("Modellerna är överens.")
    else:
        webbui.warning("Modellerna är oense.")