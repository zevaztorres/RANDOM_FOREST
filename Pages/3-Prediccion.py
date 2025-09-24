import streamlit as st
import joblib
import pandas as pd
import os
import sys
from pathlib import Path

# Agregar el directorio Pages al path para imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_trainer import MedicalDiagnosisModel

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico Médico IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .emergency-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar el modelo
@st.cache_resource
def load_medical_model():
    """Carga el modelo médico mejorado"""
    model = MedicalDiagnosisModel()
    
    # Definir la ruta del modelo (buscar en Pages/ primero, luego en directorio actual)
    model_path = None
    possible_paths = [
        current_dir / 'enhanced_rf_model.pkl',  # En Pages/
        Path.cwd() / 'Pages' / 'enhanced_rf_model.pkl',  # Desde raíz
        Path.cwd() / 'enhanced_rf_model.pkl'  # En directorio actual
    ]
    
    for path in possible_paths:
        if path.exists():
            model_path = str(path)
            break
    
    # Intentar cargar modelo existente, si no existe, entrenar uno nuevo
    if model_path and model.load_model(model_path):
        st.success(f"Modelo cargado desde: {model_path}")
    else:
        st.warning("Modelo no encontrado. Entrenando nuevo modelo...")
        df = model.create_enhanced_dataset()
        model.train_model(df)
        # Guardar en el directorio Pages
        save_path = str(current_dir / 'enhanced_rf_model.pkl')
        model.save_model(save_path)
        st.success("Modelo entrenado y guardado exitosamente!")
    
    return model

# Cargar modelo
medical_model = load_medical_model()

# Mapeo de síntomas para la interfaz
symptom_questions = {
    'respira': "¿El paciente respira normalmente?",
    'dolor_pecho': "¿Presenta dolor en el pecho?",
    'tos': "¿Tiene tos persistente?",
    'fiebre': "¿Presenta fiebre?",
    'fatiga': "¿Experimenta fatiga extrema?",
    'mareos': "¿Tiene mareos o vértigo?",
    'nauseas': "¿Presenta náuseas o vómitos?",
    'sudoracion': "¿Tiene sudoración excesiva?",
    'palpitaciones': "¿Siente palpitaciones cardíacas?",
    'dificultad_respirar': "¿Tiene dificultad para respirar?"
}

# Información adicional sobre síntomas
symptom_info = {
    'respira': "Respiración normal sin obstrucciones",
    'dolor_pecho': "Dolor, presión o molestia en el área del pecho",
    'tos': "Tos que persiste por más de unos días",
    'fiebre': "Temperatura corporal elevada (>38°C)",
    'fatiga': "Cansancio extremo que no mejora con descanso",
    'mareos': "Sensación de inestabilidad o pérdida de equilibrio",
    'nauseas': "Sensación de malestar estomacal o ganas de vomitar",
    'sudoracion': "Sudoración anormal sin causa aparente",
    'palpitaciones': "Sensación de latidos cardíacos irregulares o acelerados",
    'dificultad_respirar': "Sensación de falta de aire o respiración laboriosa"
}

# Estado de sesión para flujo secuencial
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.respuestas = {}
    st.session_state.current_symptoms = []

# Título principal
st.markdown('<h1 class="main-header">🏥 Sistema de Diagnóstico Médico con IA</h1>', unsafe_allow_html=True)

# Sidebar con información
with st.sidebar:
    
    st.markdown("---")
    st.header("📋 Progreso")
    total_symptoms = len(symptom_questions)
    completed = len(st.session_state.respuestas)
    progress = completed / total_symptoms if st.session_state.step != 'resultado' else 1.0
    st.progress(progress)
    st.write(f"Síntomas evaluados: {completed}/{total_symptoms}")

# Función para mostrar información del síntoma
def show_symptom_info(symptom_key):
    with st.expander(f"ℹ️ Más información sobre este síntoma"):
        st.write(symptom_info[symptom_key])

# Lógica principal de la aplicación
if st.session_state.step == 0:
    st.markdown('<h2 class="step-header">🚨 Evaluación Inicial Crítica</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Esta es la pregunta más importante. Una respuesta negativa requiere atención inmediata.**")
        respira = st.radio(
            symptom_questions['respira'],
            ("Sí", "No"),
            help="Evalúe si el paciente puede respirar sin asistencia"
        )
    
    with col2:
        show_symptom_info('respira')
    
    if st.button("🔄 Continuar Evaluación", type="primary"):
        st.session_state.respuestas['respira'] = 1 if respira == "Sí" else 0
        if st.session_state.respuestas['respira'] == 0:
            st.session_state.step = 'emergencia'
        else:
            st.session_state.step = 1
        st.rerun()

elif st.session_state.step == 1:
    st.markdown('<h2 class="step-header">❤️ Evaluación de Síntomas Cardíacos</h2>', unsafe_allow_html=True)
    
    cardiac_symptoms = ['dolor_pecho', 'fatiga', 'palpitaciones', 'sudoracion', 'mareos']
    
    for symptom in cardiac_symptoms:
        col1, col2 = st.columns([2, 1])
        with col1:
            response = st.radio(
                symptom_questions[symptom],
                ("No", "Sí"),
                key=f"cardiac_{symptom}",
                help=f"Evalúe: {symptom_info[symptom]}"
            )
            st.session_state.respuestas[symptom] = 1 if response == "Sí" else 0
        
        with col2:
            show_symptom_info(symptom)
    
    if st.button("🔄 Continuar a Síntomas Respiratorios", type="primary"):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.markdown('<h2 class="step-header">🫁 Evaluación de Síntomas Respiratorios</h2>', unsafe_allow_html=True)
    
    respiratory_symptoms = ['tos', 'fiebre', 'dificultad_respirar', 'nauseas']
    
    for symptom in respiratory_symptoms:
        col1, col2 = st.columns([2, 1])
        with col1:
            response = st.radio(
                symptom_questions[symptom],
                ("No", "Sí"),
                key=f"respiratory_{symptom}",
                help=f"Evalúe: {symptom_info[symptom]}"
            )
            st.session_state.respuestas[symptom] = 1 if response == "Sí" else 0
        
        with col2:
            show_symptom_info(symptom)
    
    if st.button("🔍 Realizar Diagnóstico", type="primary"):
        st.session_state.step = 'resultado'
        st.rerun()

# Manejo de casos especiales
if st.session_state.step == 'emergencia':
    st.markdown("""
    <div class="emergency-box">
        <h2>🚨 EMERGENCIA MÉDICA DETECTADA</h2>
        <h3>ACCIONES INMEDIATAS REQUERIDAS:</h3>
        <ul>
            <li><strong>1. Llamar inmediatamente a servicios de emergencia (911)</strong></li>
            <li><strong>2. Verificar vías respiratorias</strong></li>
            <li><strong>3. Iniciar RCP si es necesario</strong></li>
            <li><strong>4. Mantener al paciente en posición de recuperación</strong></li>
            <li><strong>5. No dejar solo al paciente</strong></li>
        </ul>
        <p><strong>⚠️ Esta es una situación que pone en peligro la vida. Actúe inmediatamente.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Reiniciar Evaluación"):
        st.session_state.step = 0
        st.session_state.respuestas = {}
        st.rerun()

elif st.session_state.step == 'resultado':
    st.markdown('<h2 class="step-header">📊 Resultado del Diagnóstico</h2>', unsafe_allow_html=True)
    
    try:
        # Realizar predicción con el modelo mejorado
        result = medical_model.predict(st.session_state.respuestas)
        
        # Mostrar resultado principal
        diagnosis = result['diagnosis']
        probability = result['probability']
        
        if result['diagnosis_code'] == 3:  # Emergencia
            st.markdown(f"""
            <div class="emergency-box">
                <h3>🚨 {diagnosis}</h3>
                <p><strong>Probabilidad: {probability:.1%}</strong></p>
                <p>Se requiere atención médica inmediata.</p>
            </div>
            """, unsafe_allow_html=True)
        elif result['diagnosis_code'] in [1, 2]:  # Problemas cardíacos o respiratorios
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ {diagnosis}</h3>
                <p><strong>Probabilidad: {probability:.1%}</strong></p>
                <p>Se recomienda consultar con un médico pronto.</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # Normal
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ {diagnosis}</h3>
                <p><strong>Probabilidad: {probability:.1%}</strong></p>
                <p>Los síntomas no indican problemas graves inmediatos.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar todas las probabilidades
        st.subheader("📈 Análisis Detallado de Probabilidades")
        prob_df = pd.DataFrame([
            {"Diagnóstico": diag, "Probabilidad": f"{prob:.1%}"}
            for diag, prob in result['all_probabilities'].items()
        ]).sort_values("Probabilidad", ascending=False)
        
        st.dataframe(prob_df, use_container_width=True)
        
        # Mostrar síntomas reportados
        st.subheader("📋 Resumen de Síntomas Reportados")
        symptoms_present = [
            symptom_questions[key] for key, value in st.session_state.respuestas.items() 
            if value == 1
        ]
        
        if symptoms_present:
            for symptom in symptoms_present:
                st.write(f"• {symptom}")
        else:
            st.write("No se reportaron síntomas significativos.")
        
        # Disclaimer importante
        st.markdown("""
        ---
        ### ⚠️ IMPORTANTE - Disclaimer Médico
        
        **Este sistema es únicamente una herramienta de apoyo y NO reemplaza el criterio médico profesional.**
        
        - 🔬 **Basado en IA:** Utiliza algoritmos de aprendizaje automático
        - 📊 **Datos sintéticos:** Entrenado con datos simulados para demostración
        - 👨‍⚕️ **Consulte a un médico:** Siempre busque atención médica profesional
        - 🚨 **En emergencias:** Llame inmediatamente a servicios de emergencia
        
        **No tome decisiones médicas basándose únicamente en este diagnóstico.**
        """)
        
    except Exception as e:
        st.error(f"Error al realizar el diagnóstico: {str(e)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Nueva Evaluación", type="primary"):
            st.session_state.step = 0
            st.session_state.respuestas = {}
            st.rerun()
    
    with col2:
        if st.button("📥 Descargar Reporte"):
            # Aquí podrías implementar la descarga de un reporte en PDF
            st.info("Funcionalidad de descarga en desarrollo")