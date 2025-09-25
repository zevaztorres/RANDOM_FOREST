import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configurar la página
st.set_page_config(
    page_title="Dataset Médico - Información",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar el directorio padre al path para importar model_trainer
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_trainer import MedicalDiagnosisModel

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.75rem 0;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
    }
    
    .info-card h3 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        font-size: 0.9rem;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #1f77b4;
        margin: 0.3rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-card h4 {
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-card h2 {
        font-size: 1.5rem;
        margin: 0.2rem 0;
    }
    
    .metric-card p {
        font-size: 0.8rem;
        margin: 0;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        margin: 0.3rem 0;
    }
    
    .feature-card h4 {
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    
    .feature-card p {
        font-size: 0.85rem;
        margin: 0.2rem 0;
    }
    
    .interpretation-box {
        background: #e8f4fd;
        padding: 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #0066cc;
        margin: 0.75rem 0;
    }
    
    .interpretation-box h4 {
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }
    
    .interpretation-box p {
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #ffc107;
        margin: 0.75rem 0;
    }
    
    .warning-box h4 {
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }
    
    .warning-box p {
        font-size: 0.85rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📊 Dataset Médico Sintético</h1>', unsafe_allow_html=True)

# Información general del dataset
st.markdown("""
<div class="info-card">
    <h3>🔬 Información General del Dataset</h3>
    <p>Este dataset sintético ha sido diseñado específicamente para simular datos médicos reales, 
    permitiendo el entrenamiento y evaluación de modelos de machine learning en un entorno controlado 
    y ético, sin comprometer la privacidad de pacientes reales.</p>
</div>
""", unsafe_allow_html=True)

# Crear instancia del modelo para generar datos de ejemplo
@st.cache_data
def generate_sample_dataset():
    """Genera un dataset de muestra para análisis"""
    model = MedicalDiagnosisModel()
    df = model.create_enhanced_dataset(n_samples=500)
    return df, model

# Generar dataset de muestra
with st.spinner("Generando dataset de muestra..."):
    df_sample, model = generate_sample_dataset()

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Resumen General", 
    "🔍 Características", 
    "📊 Distribuciones", 
    "🎯 Interpretación"
])

with tab1:
    st.subheader("📈 Resumen General del Dataset")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📋 Total de Muestras</h4>
            <h2>{len(df_sample)}</h2>
            <p>Registros sintéticos generados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔢 Características</h4>
            <h2>{len(model.feature_names)}</h2>
            <p>Variables predictoras</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Clases</h4>
            <h2>{len(df_sample['diagnostico'].unique())}</h2>
            <p>Diagnósticos posibles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚖️ Balance</h4>
            <h2>{(df_sample['diagnostico'].value_counts().std() / df_sample['diagnostico'].value_counts().mean() * 100):.1f}%</h2>
            <p>Coeficiente de variación</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribución de diagnósticos
    st.subheader("🎯 Distribución de Diagnósticos")
    
    diag_counts = df_sample['diagnostico'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_pie = px.pie(
            values=diag_counts.values,
            names=diag_counts.index,
            title="Distribución de Diagnósticos en el Dataset",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Estadísticas por Clase")
        for diag, count in diag_counts.items():
            percentage = (count / len(df_sample)) * 100
            st.markdown(f"""
            <div class="feature-card">
                <strong>{diag}</strong><br>
                <span style="color: #666;">Muestras: {count} ({percentage:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Información sobre el balance del dataset
    st.markdown("""
    <div class="interpretation-box">
        <h4>📊 Interpretación del Balance</h4>
        <p><strong>Dataset Balanceado:</strong> Todas las clases tienen una representación similar, 
        lo que es ideal para el entrenamiento de modelos de machine learning. Esto evita sesgos 
        hacia clases mayoritarias y asegura que el modelo aprenda patrones de todas las condiciones médicas.</p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.subheader("🔍 Características del Dataset")
    
    # Información sobre las características
    st.markdown("""
    <div class="interpretation-box">
        <h4>🏥 Características Médicas Simuladas</h4>
        <p>Cada característica representa un síntoma o indicador médico normalizado entre 0 y 1, 
        donde valores más altos indican mayor intensidad o presencia del síntoma.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar estadísticas descriptivas
    st.subheader("📊 Estadísticas Descriptivas")
    
    # Crear DataFrame con estadísticas
    feature_stats = df_sample[model.feature_names].describe().round(3)
    st.dataframe(feature_stats, use_container_width=True)
    
    # Explicación de cada característica
    st.subheader("📋 Descripción de Características")
    
    feature_descriptions = {
        'respira': {
            'descripcion': 'Capacidad de respirar normalmente',
            'rango': '0 (no respira) - 1 (respira normalmente)',
            'interpretacion': 'Valor 0 indica emergencia crítica'
        },
        'dolor_pecho': {
            'descripcion': 'Presencia de dolor en el pecho',
            'rango': '0 (sin dolor) - 1 (dolor presente)',
            'interpretacion': 'Valor 1 puede indicar problema cardíaco'
        },
        'tos': {
            'descripcion': 'Presencia de tos',
            'rango': '0 (sin tos) - 1 (tos presente)',
            'interpretacion': 'Valor 1 puede indicar problema respiratorio'
        },
        'fiebre': {
            'descripcion': 'Presencia de fiebre',
            'rango': '0 (sin fiebre) - 1 (fiebre presente)',
            'interpretacion': 'Valor 1 indica proceso infeccioso o inflamatorio'
        },
        'fatiga': {
            'descripcion': 'Presencia de cansancio extremo',
            'rango': '0 (sin fatiga) - 1 (fatiga presente)',
            'interpretacion': 'Valor 1 puede indicar múltiples condiciones'
        },
        'mareos': {
            'descripcion': 'Presencia de mareos o vértigo',
            'rango': '0 (sin mareos) - 1 (mareos presentes)',
            'interpretacion': 'Valor 1 puede indicar problema circulatorio'
        },
        'nauseas': {
            'descripcion': 'Presencia de náuseas',
            'rango': '0 (sin náuseas) - 1 (náuseas presentes)',
            'interpretacion': 'Valor 1 puede indicar malestar general'
        },
        'sudoracion': {
            'descripcion': 'Presencia de sudoración excesiva',
            'rango': '0 (sin sudoración) - 1 (sudoración presente)',
            'interpretacion': 'Valor 1 puede indicar estrés fisiológico'
        },
        'palpitaciones': {
            'descripcion': 'Presencia de palpitaciones cardíacas',
            'rango': '0 (sin palpitaciones) - 1 (palpitaciones presentes)',
            'interpretacion': 'Valor 1 puede indicar problema cardíaco'
        },
        'dificultad_respirar': {
            'descripcion': 'Presencia de dificultad para respirar',
            'rango': '0 (respiración normal) - 1 (dificultad presente)',
            'interpretacion': 'Valor 1 puede indicar problema respiratorio o cardíaco'
        }
    }
    
    # Mostrar características en columnas
    col1, col2 = st.columns(2)
    
    for i, (feature, info) in enumerate(feature_descriptions.items()):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h4>🔸 {feature.replace('_', ' ').title()}</h4>
                <p><strong>Descripción:</strong> {info['descripcion']}</p>
                <p><strong>Rango:</strong> {info['rango']}</p>
                <p><strong>Interpretación:</strong> {info['interpretacion']}</p>
                <p><strong>Promedio en dataset:</strong> {df_sample[feature].mean():.3f}</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.subheader("📊 Distribuciones y Correlaciones")
    

    
    # Matriz de correlación
    st.subheader("🔗 Matriz de Correlación")
    
    correlation_matrix = df_sample[model.feature_names].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlaciones entre Características",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    
    fig_corr.update_layout(
        width=800,
        height=600
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Interpretación de correlaciones
    st.markdown("""
    <div class="interpretation-box">
        <h4>🔗 Interpretación de Correlaciones</h4>
        <p><strong>Correlaciones Positivas (Azul):</strong> Cuando una característica aumenta, 
        la otra también tiende a aumentar. Por ejemplo, fiebre y dolor de cabeza suelen aparecer juntos.</p>
        <p><strong>Correlaciones Negativas (Rojo):</strong> Cuando una característica aumenta, 
        la otra tiende a disminuir.</p>
        <p><strong>Sin Correlación (Blanco):</strong> Las características son independientes entre sí.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Box plots por diagnóstico
    st.subheader("📦 Distribución por Diagnóstico")
    
    # Seleccionar característica para análisis detallado
    selected_feature = st.selectbox(
        "Selecciona una característica para análisis detallado:",
        model.feature_names,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    fig_box = px.box(
        df_sample,
        x='diagnostico',
        y=selected_feature,
        title=f"Distribución de {selected_feature.replace('_', ' ').title()} por Diagnóstico",
        color='diagnostico'
    )
    
    fig_box.update_layout(
        xaxis_title="Diagnóstico",
        yaxis_title=selected_feature.replace('_', ' ').title(),
        showlegend=False
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

with tab4:
    st.subheader("🎯 Interpretación y Uso del Dataset")
    
    # Información sobre el propósito
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Propósito del Dataset Sintético</h3>
        <p>Este dataset ha sido creado para demostrar técnicas de machine learning en el ámbito médico 
        sin utilizar datos reales de pacientes, respetando así la privacidad y las regulaciones médicas.</p>
    </div>
    """, unsafe_allow_html=True)
    

    # Interpretación de resultados
    st.subheader("📈 Cómo Interpretar los Resultados")
    
    st.markdown("""
    <div class="interpretation-box">
        <h4>🔍 Interpretación de Predicciones</h4>
        <ul>
            <li><strong>Probabilidades:</strong> El modelo proporciona probabilidades para cada diagnóstico</li>
            <li><strong>Confianza:</strong> Probabilidades altas (>0.7) indican mayor confianza del modelo</li>
            <li><strong>Incertidumbre:</strong> Probabilidades similares entre clases indican incertidumbre</li>
            <li><strong>Contexto:</strong> Siempre considerar el contexto clínico completo</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
# Footer con información adicional
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 <strong>Dataset Sintético Generado</strong> | 
    📊 <strong>Solo para Fines Educativos</strong> | 
    🏥 <strong>No para Uso Clínico Real</strong></p>
</div>
""", unsafe_allow_html=True)