import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
import os

# Agregar el directorio Pages al path para imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_trainer import MedicalDiagnosisModel

# Configuración de la página
st.set_page_config(
    page_title="Entrenamiento del Modelo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .compact-metric-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1f77b4;
        text-align: center;
    }
    .compact-metric-card h5 {
        margin: 0 0 0.2rem 0;
        font-size: 0.8rem;
        color: #666;
    }
    .compact-metric-card h3 {
        margin: 0;
        font-size: 1.2rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .success-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuración inicial de variables
if 'dataset_size' not in st.session_state:
    st.session_state.dataset_size = 1000
if 'test_size' not in st.session_state:
    st.session_state.test_size = 20
if 'cv_folds' not in st.session_state:
    st.session_state.cv_folds = 5
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 100
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = None
if 'min_samples_split' not in st.session_state:
    st.session_state.min_samples_split = 2
if 'min_samples_leaf' not in st.session_state:
    st.session_state.min_samples_leaf = 1

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Modelo", "🌳 Parámetros del Random Forest", "📊 Configuración del Dataset", "🔄 Validación Cruzada", "📈 Evaluación del Modelo"])

# TAB 1: Modelo
with tab1:
    st.header("⚙️ Configuración del Modelo")
    st.subheader("📊 Configuración Actual del Modelo")

    st.info(f"""
    🎯 **Resumen Completo**: Esta configuración generará un Random Forest con {st.session_state.n_estimators} árboles,
    entrenado con {st.session_state.dataset_size} muestras y evaluado usando {st.session_state.cv_folds}-fold cross-validation.
    """)

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.markdown(f"""
        <div class="compact-metric-card">
            <h5>🌳 Árboles</h5>
            <h3>{st.session_state.n_estimators}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="compact-metric-card">
            <h5>📏 Profundidad</h5>
            <h3>{'Sin límite' if st.session_state.max_depth is None else st.session_state.max_depth}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with config_col2:
        st.markdown(f"""
        <div class="compact-metric-card">
            <h5>📈 Dataset</h5>
            <h3>{st.session_state.dataset_size}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="compact-metric-card">
            <h5>🧪 Test</h5>
            <h3>{st.session_state.test_size}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with config_col3:
        st.markdown(f"""
        <div class="compact-metric-card">
            <h5>🔄 CV Folds</h5>
            <h3>{st.session_state.cv_folds}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="compact-metric-card">
            <h5>🍃 Min Leaf</h5>
            <h3>{st.session_state.min_samples_leaf}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Sección de Entrenamiento
    st.markdown("---")
    st.subheader("🚀 Entrenamiento del Modelo")
    
    # Inicializar variables de estado para el entrenamiento
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'model_instance' not in st.session_state:
        st.session_state.model_instance = None

    train_col1, train_col2 = st.columns([2, 1])
    
    with train_col1:
        st.info(f"""
        🎯 **Listo para entrenar**: El modelo se entrenará con los parámetros configurados arriba.
        Esto puede tomar unos momentos dependiendo del tamaño del dataset y número de árboles.
        """)
    
    with train_col2:
        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            # Mostrar progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Paso 1: Inicializar modelo
                status_text.text("🔧 Inicializando modelo...")
                progress_bar.progress(20)
                
                model = MedicalDiagnosisModel()
                
                # Paso 2: Entrenar modelo con todos los parámetros
                status_text.text("🌳 Entrenando Random Forest...")
                progress_bar.progress(60)
                trained_model, results = model.train_model(
                    n_estimators=st.session_state.n_estimators,
                    max_depth=st.session_state.max_depth,
                    min_samples_split=st.session_state.min_samples_split,
                    min_samples_leaf=st.session_state.min_samples_leaf,
                    test_size=st.session_state.test_size/100,
                    cv_folds=st.session_state.cv_folds,
                    n_samples=st.session_state.dataset_size
                )
                
                # Paso 5: Completado
                status_text.text("✅ Entrenamiento completado!")
                progress_bar.progress(100)
                
                # Guardar resultados en session state
                st.session_state.model_trained = True
                st.session_state.training_results = results
                st.session_state.model_instance = model
                
                # Mostrar éxito
                st.success("🎉 ¡Modelo entrenado exitosamente!")
                
                # Limpiar indicadores de progreso
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                progress_bar.empty()
                status_text.empty()

    # Mostrar resultados si el modelo está entrenado
    if st.session_state.model_trained and st.session_state.training_results:
        st.markdown("---")
        st.subheader("📊 Resultados del Entrenamiento")
        
        results = st.session_state.training_results
        
        # Métricas principales en columnas
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("🎯 Accuracy", f"{results['accuracy']:.3f}")
            st.metric("📊 CV Desviación", f"{results['cv_std']:.3f}")
        with metric_col2:
            st.metric("📈 Precisión", f"{results['accuracy']*100:.1f}%")
            st.metric("🌳 Árboles", f"{results['model_params']['n_estimators']}")
        with metric_col3:
            st.metric("📈 CV Score Promedio", f"{results['cv_mean']:.3f}")


        
        # Importancia de características
        if 'feature_importance' in results:
            st.subheader("🎯 Importancia de Características")
            st.dataframe(results['feature_importance'], use_container_width=True)

# TAB 2: Parámetros Random Forest
with tab2:
    st.subheader("Parámetros del Random Forest")

    st.info("""
    🌳 **Random Forest** es un algoritmo de ensemble que combina múltiples árboles de decisión.
    Estos parámetros controlan la estructura y comportamiento de cada árbol individual.
    """)

    rf_col1, rf_col2 = st.columns(2)

    with rf_col1:
        n_estimators = st.slider(
            "Número de Árboles (n_estimators)", 
            min_value=10, max_value=500, value=st.session_state.n_estimators, step=10
        )
        st.session_state.n_estimators = n_estimators

        min_samples_split = st.slider(
            "Mínimo de Muestras para División",
            min_value=2, max_value=20, value=st.session_state.min_samples_split
        )
        st.session_state.min_samples_split = min_samples_split

    with rf_col2:
        depth_options = [None, 5, 10, 15, 20, 25, 30]
        current_index = depth_options.index(st.session_state.max_depth) if st.session_state.max_depth in depth_options else 0
        max_depth = st.selectbox("Profundidad Máxima (max_depth)", options=depth_options, index=current_index)
        st.session_state.max_depth = max_depth

        min_samples_leaf = st.slider(
            "Mínimo de Muestras por Hoja",
            min_value=1, max_value=10, value=st.session_state.min_samples_leaf
        )
        st.session_state.min_samples_leaf = min_samples_leaf

    # Guía de parámetros en la misma pestaña
    st.markdown("---")
    st.subheader("📚 Guía de Parámetros del Random Forest")
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        st.markdown(f"""
        **🌳 Número de Árboles (n_estimators): {n_estimators}**
        - Más árboles = mejor rendimiento pero más tiempo
        - 100-300 recomendado

        **📏 Profundidad Máxima: {'Sin límite' if max_depth is None else max_depth}**
        - None = sin límite (riesgo de sobreajuste)
        - 10-20 = recomendado para datasets medianos
        """)
    with param_col2:
        st.markdown(f"""
        **🔢 Min Samples Split: {min_samples_split}**
        - 2 = máxima flexibilidad
        - 5-10 recomendado

        **🍃 Min Samples Leaf: {min_samples_leaf}**
        - 1 = máxima flexibilidad
        - 2-5 recomendado
        """)

# TAB 3: Dataset
with tab3:
    st.subheader("Configuración del Dataset")

    st.info(f"""
    📊 **Dataset Sintético**: Se generarán {st.session_state.dataset_size} muestras con características médicas simuladas.
    El {st.session_state.test_size}% se reservará para evaluación final.
    """)

    dataset_col1, dataset_col2 = st.columns(2)
    with dataset_col1:
        dataset_size = st.slider("Tamaño del Dataset", 100, 1000, st.session_state.dataset_size, 50)
        st.session_state.dataset_size = dataset_size
    with dataset_col2:
        test_size = st.slider("Porcentaje para Prueba (%)", 10, 40, st.session_state.test_size)
        st.session_state.test_size = test_size

    # Guía en la misma pestaña
    st.markdown("---")
    st.subheader("📚 Guía del Dataset")
    st.markdown(f"""
    - {dataset_size} muestras totales
    - {int(dataset_size * (1-test_size/100))} entrenamiento
    - {int(dataset_size * test_size/100)} prueba
    - Recomendado: 20-25% de prueba
    """)

# TAB 4: Validación Cruzada
with tab4:
    st.subheader("Configuración de Validación Cruzada")

    st.info(f"""
    🔄 **Validación Cruzada**: Divide los datos en {st.session_state.cv_folds} folds.
    Cada fold se usa como prueba mientras los otros {st.session_state.cv_folds-1} entrenan.
    """)

    cv_folds = st.slider("Folds de Validación Cruzada", 3, 10, st.session_state.cv_folds)
    st.session_state.cv_folds = cv_folds

    # Guía en la misma pestaña
    st.markdown("---")
    st.subheader("📚 Guía de Validación Cruzada")
    st.markdown(f"""
    - {cv_folds} folds = {cv_folds} modelos entrenados
    - 5-10 recomendado
    - Cada fold tendrá ~{int(dataset_size * (1-test_size/100) / cv_folds)} muestras
    """)

# TAB 5: Evaluación
with tab5:
    st.subheader("📈 Evaluación Avanzada del Modelo")
    
    # Verificar si hay un modelo entrenado
    if not st.session_state.get('model_trained', False) or 'training_results' not in st.session_state:
        st.warning("⚠️ **No hay modelo entrenado**")
        st.info("""
        Para ver la evaluación detallada del modelo:
        1. Ve a la pestaña **'Modelo'**
        2. Haz clic en **'🚀 Entrenar Modelo'**
        3. Regresa aquí para ver los resultados
        """)
    else:
        # Mostrar evaluación completa
        results = st.session_state.training_results
        
        # Métricas principales
        st.subheader("🎯 Métricas de Rendimiento")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{results['accuracy']:.3f}",
                help="Porcentaje de predicciones correctas"
            )
        with metric_col2:
            st.metric(
                label="📈 CV Score Promedio",
                value=f"{results['cv_mean']:.3f}",
                help="Promedio de validación cruzada"
            )
        with metric_col3:
            st.metric(
                label="📊 CV Desviación",
                value=f"{results['cv_std']:.3f}",
                help="Desviación estándar de CV"
            )
        with metric_col4:
            st.metric(
                label="🌳 Número de Árboles",
                value=f"{results['model_params']['n_estimators']}",
                help="Árboles en el Random Forest"
            )
        
        # Matriz de Confusión
        st.subheader("🔍 Matriz de Confusión")
        if 'confusion_matrix' in results:
            conf_matrix = results['confusion_matrix']
            
            # Crear gráfica interactiva de la matriz de confusión
            fig_conf = px.imshow(
                conf_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                title="Matriz de Confusión"
            )
            fig_conf.update_layout(
                xaxis_title="Predicción",
                yaxis_title="Valor Real",
                width=500,
                height=400
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Importancia de Características
        st.subheader("🎯 Importancia de Características")
        if 'feature_importance' in results:
            feature_df = results['feature_importance']
            
            # Gráfica de barras de importancia
            fig_importance = px.bar(
                feature_df.head(10),  # Top 10 características
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Características Más Importantes",
                color='importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Tabla detallada
            st.dataframe(feature_df, use_container_width=True)
        
        # Validación Cruzada
        st.subheader("🔄 Validación Cruzada")
        if 'cv_scores' in results:
            cv_scores = results['cv_scores']
            
            # Estadísticas de CV en una fila compacta
            cv_stats_col1, cv_stats_col2, cv_stats_col3 = st.columns(3)
            with cv_stats_col1:
                st.metric("📊 Score Mínimo", f"{cv_scores.min():.3f}")
            with cv_stats_col2:
                st.metric("📈 Score Máximo", f"{cv_scores.max():.3f}")
            with cv_stats_col3:
                st.metric("📏 Rango", f"{cv_scores.max() - cv_scores.min():.3f}")
        
        # Reporte de Clasificación
        st.subheader("📋 Reporte de Clasificación Detallado")
        if 'classification_report' in results:
            class_report = results['classification_report']
            
            # Métricas resumidas principales
            if 'macro avg' in class_report and 'weighted avg' in class_report:
                st.markdown("#### 📊 Métricas Generales")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric(
                        "🎯 Precisión Promedio", 
                        f"{class_report['macro avg']['precision']:.3f}",
                        help="Promedio de precisión entre todas las clases"
                    )
                with summary_col2:
                    st.metric(
                        "🔍 Recall Promedio", 
                        f"{class_report['macro avg']['recall']:.3f}",
                        help="Promedio de recall entre todas las clases"
                    )
                with summary_col3:
                    st.metric(
                        "⚖️ F1-Score Promedio", 
                        f"{class_report['macro avg']['f1-score']:.3f}",
                        help="Promedio de F1-Score entre todas las clases"
                    )
            
            st.markdown("#### 📈 Rendimiento por Clase")
            
            # Crear DataFrame del reporte por clases
            report_data = []
            class_names = []
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_names.append(class_name)
                    report_data.append({
                        '🏷️ Clase': class_name,
                        '🎯 Precisión': f"{metrics.get('precision', 0):.3f}",
                        '🔍 Recall': f"{metrics.get('recall', 0):.3f}",
                        '⚖️ F1-Score': f"{metrics.get('f1-score', 0):.3f}",
                        '📊 Soporte': int(metrics.get('support', 0))
                    })
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                st.dataframe(
                    report_df, 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Gráfica de barras para F1-Score por clase
                if len(class_names) > 1:
                    st.markdown("#### 📊 Comparación de F1-Score por Clase")
                    f1_scores = []
                    for class_name in class_names:
                        if class_name in class_report:
                            f1_scores.append(class_report[class_name]['f1-score'])
                    
                    if f1_scores:
                        fig_f1 = px.bar(
                            x=class_names,
                            y=f1_scores,
                            title="F1-Score por Clase de Diagnóstico",
                            labels={'x': 'Clase de Diagnóstico', 'y': 'F1-Score'},
                            color=f1_scores,
                            color_continuous_scale='viridis'
                        )
                        fig_f1.update_layout(
                            showlegend=False,
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_f1, use_container_width=True)
        
        # Información del Dataset
        st.subheader("📊 Información del Dataset Utilizado")
        dataset_info = results['dataset_info']
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("📈 Total Muestras", dataset_info['total_samples'])
        with info_col2:
            st.metric("🏋️ Entrenamiento", dataset_info['train_samples'])
        with info_col3:
            st.metric("🧪 Prueba", dataset_info['test_samples'])
        with info_col4:
            st.metric("📊 Proporción Test", f"{(dataset_info['test_samples']/dataset_info['total_samples']*100):.1f}%")
        
        # Parámetros del Modelo
        st.subheader("⚙️ Parámetros del Modelo Entrenado")
        params = results['model_params']
        
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            st.markdown(f"""
            **🌳 Número de Árboles**: {params['n_estimators']}
            **📏 Profundidad Máxima**: {'Sin límite' if params['max_depth'] is None else params['max_depth']}
            **🔢 Min Samples Split**: {params['min_samples_split']}
            """)
        with param_col2:
            st.markdown(f"""
            **🍃 Min Samples Leaf**: {params['min_samples_leaf']}
            **🎲 Random State**: {params.get('random_state', 'No especificado')}
            **🔄 Folds CV**: {st.session_state.cv_folds}
            """)
        
        # Interpretación de Resultados
        st.subheader("💡 Interpretación de Resultados")
        accuracy = results['accuracy']
        cv_mean = results['cv_mean']
        cv_std = results['cv_std']
        
        if accuracy >= 0.9:
            st.success(f"🎉 **Excelente rendimiento**: El modelo tiene una accuracy de {accuracy:.1%}, lo que indica un rendimiento muy bueno.")
        elif accuracy >= 0.8:
            st.info(f"✅ **Buen rendimiento**: El modelo tiene una accuracy de {accuracy:.1%}, lo que es satisfactorio para la mayoría de aplicaciones.")
        elif accuracy >= 0.7:
            st.warning(f"⚠️ **Rendimiento moderado**: El modelo tiene una accuracy de {accuracy:.1%}. Considera ajustar los parámetros.")
        else:
            st.error(f"❌ **Rendimiento bajo**: El modelo tiene una accuracy de {accuracy:.1%}. Se recomienda revisar los datos y parámetros.")
        
        if cv_std < 0.05:
            st.success(f"🎯 **Modelo estable**: La desviación estándar de CV ({cv_std:.3f}) es baja, indicando consistencia.")
        elif cv_std < 0.1:
            st.info(f"📊 **Estabilidad moderada**: La desviación estándar de CV ({cv_std:.3f}) es aceptable.")
        else:
            st.warning(f"⚠️ **Modelo inestable**: La desviación estándar de CV ({cv_std:.3f}) es alta. El modelo puede ser inconsistente.")
