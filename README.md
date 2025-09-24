# 🏥 Proyecto Random Forest - Diagnóstico Médico

## 📋 Descripción
Aplicación web desarrollada con **Streamlit** que utiliza algoritmos de **Random Forest** para el diagnóstico médico automatizado. El sistema permite cargar datasets médicos, entrenar modelos de machine learning y realizar predicciones de diagnósticos.

## 🚀 Características Principales

### 📊 Gestión de Datasets
- Carga de archivos CSV con datos médicos
- Visualización interactiva de datos
- Análisis estadístico automático
- Detección de valores faltantes
- Gráficos de distribución y correlación

### 🤖 Entrenamiento de Modelos
- Algoritmo Random Forest optimizado
- Validación cruzada automática
- Métricas de rendimiento completas
- Matriz de confusión interactiva
- Análisis de importancia de características

### 🔮 Predicciones
- Interfaz intuitiva para nuevas predicciones
- Resultados con probabilidades
- Interpretación automática de resultados
- Historial de predicciones

### 📈 Evaluación Avanzada
- Reporte de clasificación detallado
- Métricas de precisión, recall y F1-score
- Visualizaciones interactivas con Plotly
- Análisis de validación cruzada
- Comparación de rendimiento por clase

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit** - Framework web
- **Scikit-learn** - Machine Learning
- **Pandas** - Manipulación de datos
- **NumPy** - Computación numérica
- **Plotly** - Visualizaciones interactivas
- **Seaborn** - Gráficos estadísticos

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/EMARTINEZ1993/Proyecto_RandomsFores.git
cd Proyecto_RandomsFores
```

2. Instala las dependencias:
```bash
pip install streamlit pandas numpy scikit-learn plotly seaborn
```

3. Ejecuta la aplicación:
```bash
streamlit run Inicio.py
```

## 🎯 Uso de la Aplicación

### 1. Página de Inicio
- Información general del proyecto
- Instrucciones de uso
- Navegación a las diferentes secciones

### 2. Dataset (📊)
- Carga tu archivo CSV con datos médicos
- Explora las características del dataset
- Visualiza distribuciones y correlaciones

### 3. Entrenamiento (🤖)
- Configura los parámetros del modelo
- Entrena el algoritmo Random Forest
- Evalúa el rendimiento del modelo

### 4. Predicción (🔮)
- Ingresa nuevos datos de pacientes
- Obtén predicciones de diagnóstico
- Visualiza probabilidades y confianza

### 5. Evaluación (📈)
- Analiza métricas detalladas del modelo
- Revisa la matriz de confusión
- Examina la importancia de características

## 📁 Estructura del Proyecto

```
Proyecto_RandomsFores/
├── Inicio.py                 # Página principal
├── Pages/
│   ├── 1-Dataset.py          # Gestión de datasets
│   ├── 2-Entrenamiento.py    # Entrenamiento de modelos
│   ├── 3-Prediccion.py       # Predicciones
│   └── model_trainer.py      # Lógica de ML
├── enhanced_rf_model.pkl     # Modelo entrenado
└── README.md                 # Este archivo
```

## 🔧 Configuración del Modelo

El modelo Random Forest está optimizado con los siguientes parámetros:
- **n_estimators**: 100 árboles
- **max_depth**: 10 niveles máximos
- **min_samples_split**: 5 muestras mínimas
- **min_samples_leaf**: 2 muestras por hoja
- **random_state**: 42 (reproducibilidad)

## 📊 Métricas de Evaluación

La aplicación proporciona métricas completas:
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precisión y recall
- **Validación Cruzada**: Robustez del modelo

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**EMARTINEZ1993**
- GitHub: [@EMARTINEZ1993](https://github.com/EMARTINEZ1993)

## 🙏 Agradecimientos

- Comunidad de Streamlit por la excelente documentación
- Scikit-learn por las herramientas de machine learning
- Plotly por las visualizaciones interactivas

---

⭐ Si este proyecto te ha sido útil, ¡no olvides darle una estrella!