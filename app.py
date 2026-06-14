# -------------------------------------
# IMPORT LIBRARIES
# -------------------------------------
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from datetime import datetime

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(page_title="Oil Pump Dashboard", layout="wide")

# -------------------------------------
# LOAD MODEL
# -------------------------------------
with open("model/random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------------
# HEADER
# -------------------------------------
st.title("⚙ Oil Pump Failure Monitoring Dashboard")
st.markdown("### AI-Based Predictive Maintenance System")
st.divider()

# -------------------------------------
# LAYOUT (LEFT + RIGHT)
# -------------------------------------
main_col, info_col = st.columns([3, 1])

# ================= LEFT SIDE =================
with main_col:

    st.subheader("🔧 Enter Sensor Parameters")

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        process_temp = st.number_input("Process Temperature (K)", value=310.0)
        rot_speed = st.number_input("Rotational Speed (rpm)", value=1500)

    with col2:
        torque = st.number_input("Torque (Nm)", value=40.0)
        tool_wear = st.number_input("Tool Wear (min)", value=100)
        pump_type = st.selectbox("Pump Type", ["L", "M", "H"])

    type_L = 1 if pump_type == "L" else 0
    type_M = 1 if pump_type == "M" else 0

    st.divider()

    if st.button("🚀 Predict Failure"):

        input_data = np.array([[air_temp, process_temp, rot_speed,
                                torque, tool_wear, type_L, type_M]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100

        st.session_state["prob"] = probability
        st.session_state["pred"] = prediction[0]
        st.session_state["inputs"] = {
            "Air Temp": air_temp,
            "Process Temp": process_temp,
            "RPM": rot_speed,
            "Torque": torque,
            "Tool Wear": tool_wear,
            "Type": pump_type
        }

    # -------- RESULTS --------
    if "prob" in st.session_state:

        probability = st.session_state["prob"]
        prediction = st.session_state["pred"]

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Failure Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # Risk
        if probability < 30:
            risk = "LOW"
        elif probability < 70:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        st.write(f"### Risk Level: {risk}")

        if prediction == 1:
            st.error("⚠ Pump Failure Predicted")
        else:
            st.success("✅ Pump Operating Normally")

        # Chart
        sensor_df = pd.DataFrame({
            "Sensor": ["Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear"],
            "Value": [st.session_state["inputs"]["Air Temp"],
                      st.session_state["inputs"]["Process Temp"],
                      st.session_state["inputs"]["RPM"],
                      st.session_state["inputs"]["Torque"],
                      st.session_state["inputs"]["Tool Wear"]]
        })

        fig = px.bar(sensor_df, x="Sensor", y="Value", color="Value")
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------
        # PDF GENERATION (ATTRACTIVE)
        # -------------------------------------
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer,
                                rightMargin=50,
                                leftMargin=50,
                                topMargin=50,
                                bottomMargin=40)

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('title', parent=styles['Title'], textColor=colors.darkblue)
        heading = ParagraphStyle('heading', parent=styles['Heading2'], textColor=colors.blue)

        elements = []

        # Title
        elements.append(Paragraph("Oil Pump Prediction Report", title_style))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        # Table
        table_data = [["Parameter", "Value"]]
        for k, v in st.session_state["inputs"].items():
            table_data.append([k, str(v)])
        table_data.append(["Probability", f"{probability:.2f}%"])
        table_data.append(["Risk Level", risk])

        table = Table(table_data)

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('GRID', (0, 0), (-1, -1), 1.2, colors.grey),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.whitesmoke, colors.lightgrey]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.4 * inch))

        # Reasons & Precautions
        if risk == "HIGH":
            reasons = ["High torque causing stress.", "Severe wear in components."]
            precautions = [
                "Stop the pump immediately to avoid further damage.",
                "Inspect all major components carefully.",
                "Replace damaged or worn-out parts.",
                "Ensure proper lubrication before restart."
            ]
        elif risk == "MEDIUM":
            reasons = ["Moderate wear detected.", "Increasing load conditions."]
            precautions = [
                "Reduce operational load gradually.",
                "Monitor system parameters frequently.",
                "Schedule maintenance inspection soon.",
                "Avoid continuous heavy usage."
            ]
        else:
            reasons = ["System operating normally.", "No abnormal conditions detected."]
            precautions = [
                "Continue regular monitoring.",
                "Maintain proper lubrication.",
                "Follow routine maintenance schedule.",
                "Perform periodic inspection."
            ]

        elements.append(Paragraph("Reasons", heading))
        elements.append(ListFlowable([ListItem(Paragraph(r, styles["Normal"])) for r in reasons]))

        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("Precautions", heading))
        elements.append(ListFlowable([ListItem(Paragraph(p, styles["Normal"])) for p in precautions]))

        doc.build(elements)

        st.download_button("📄 Download Report", buffer.getvalue(), "Pump_Report.pdf", "application/pdf")

# ================= RIGHT SIDE =================
with info_col:
    st.markdown("""
    <div style='background-color:#e8f0fe;
                padding:20px;
                border-radius:10px;
                border-left:6px solid #0066cc;'>

    <h4>📘 About This Project</h4>

    This system uses a Random Forest model  
    to predict oil pump failure using sensor data.  
    It helps in early fault detection and maintenance planning.

    <hr>

    <b>Model:</b> Random Forest  
    <b>Accuracy:</b> 98.45%

    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("<center>Developed by Samyuktha | MCA Project</center>", unsafe_allow_html=True)