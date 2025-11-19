import streamlit as st
import pandas as pd
import joblib
import requests
import warnings
import os
import numpy as np
import tempfile
from audio_recorder_streamlit import audio_recorder
import altair as alt

# Configuración
st.set_page_config(page_title="Predictor Musical", page_icon="🎵", layout="wide")
warnings.filterwarnings('ignore')

TAMAÑO_OBJETIVO_MB = 4.9
MODELO_PATH = 'modelo.joblib'
DATASET_PATH = 'dataset.csv'

@st.cache_data
def cargar_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
        return None

@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load(MODELO_PATH)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def estimate_key_mode(y, sr):
    """Algoritmo Krumhansl-Schmuckler para detección de tonalidad"""
    try:
        import librosa
        
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        corr_major = [np.corrcoef(np.roll(maj_profile, i), chroma_mean)[0, 1] for i in range(12)]
        corr_minor = [np.corrcoef(np.roll(min_profile, i), chroma_mean)[0, 1] for i in range(12)]
        
        key_major = np.argmax(corr_major)
        key_minor = np.argmax(corr_minor)
        
        if max(corr_major) >= max(corr_minor):
            return key_major, 1, max(corr_major)
        else:
            return key_minor, 0, max(corr_minor)
            
    except Exception:
        return 0, 1, 0.5

def extraer_key_mode(archivo_bytes, nombre_archivo):
    """Extrae key y mode del audio"""
    try:
        import librosa
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{nombre_archivo.split('.')[-1]}") as temp_file:
            temp_file.write(archivo_bytes)
            temp_path = temp_file.name
        
        try:
            y, sr = librosa.load(temp_path, sr=None, mono=True)
            key, mode, confidence = estimate_key_mode(y, sr)
            
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            return {
                "key": int(key),
                "mode": int(mode),
                "key_name": key_names[key],
                "mode_name": "Mayor" if mode == 1 else "Menor",
                "confidence": float(confidence)
            }
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception:
        return {"key": 0, "mode": 1, "key_name": "C", "mode_name": "Mayor", "confidence": 0.5}

def agregar_escalas(features, key, mode):
    """Agrega scale y non_naturales_notes basado en key y mode"""
    data_escala = [
        ["C Mayor", 0, 1, 0], ["Db Mayor", 1, 1, 5], ["D Mayor", 2, 1, 2],
        ["Eb Mayor", 3, 1, 3], ["E Mayor", 4, 1, 4], ["F Mayor", 5, 1, 1],
        ["F# Mayor", 6, 1, 5], ["G Mayor", 7, 1, 1], ["Ab Mayor", 8, 1, 4],
        ["A Mayor", 9, 1, 3], ["Bb Mayor", 10, 1, 2], ["B Mayor", 11, 1, 5],
        ["A Menor", 9, 0, 0], ["Bb Menor", 10, 0, 5], ["B Menor", 11, 0, 2],
        ["C Menor", 0, 0, 3], ["C# Menor", 1, 0, 4], ["D Menor", 2, 0, 1],
        ["D# Menor", 3, 0, 5], ["E Menor", 4, 0, 1], ["F Menor", 5, 0, 4],
        ["F# Menor", 6, 0, 3], ["G Menor", 7, 0, 2], ["G# Menor", 8, 0, 5]
    ]
    
    df_escala = pd.DataFrame(data_escala, columns=["scale", "key", "mode", "non_naturales_notes"])
    fila = df_escala[(df_escala["key"] == key) & (df_escala["mode"] == mode)]
    
    if not fila.empty:
        features["scale"] = fila.iloc[0]["scale"]
        features["non_naturales_notes"] = fila.iloc[0]["non_naturales_notes"]
    else:
        features["scale"] = "C Mayor"
        features["non_naturales_notes"] = 0
    
    return features

def cortar_audio(archivo_bytes, tamaño_objetivo_mb):
    """Corta el audio al tamaño objetivo"""
    tamaño_objetivo_bytes = tamaño_objetivo_mb * 1024 * 1024
    
    if len(archivo_bytes) <= tamaño_objetivo_bytes:
        return archivo_bytes
    
    proporcion = tamaño_objetivo_bytes / len(archivo_bytes) * 0.95
    nuevo_tamaño = int(len(archivo_bytes) * proporcion)
    return archivo_bytes[:nuevo_tamaño]

def analizar_con_reccobeats(archivo_audio):
    """Análisis con API ReccoBeats"""
    try:
        files = {'audioFile': (archivo_audio.name, archivo_audio.getvalue(), archivo_audio.type)}
        response = requests.post('https://api.reccobeats.com/v1/analysis/audio-features', 
                                files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json(), True
        else:
            return None, False
    except Exception:
        return None, False

def crear_dataframe_prediccion(features, columnas_modelo):
    datos = {}
    for col in columnas_modelo:
        datos[col] = features.get(col, 0)
    return pd.DataFrame([datos])

class ArchivoAudio:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.type = 'audio/mp3'
        self.size = len(data)
    
    def getvalue(self):
        return self.data

def pagina_predictor():
    """Página del predictor de artistas"""
    st.title("Predictor de Artistas")
    
    # Cargar modelo
    modelo = cargar_modelo()
    if modelo is None:
        st.error("No se pudo cargar el modelo.")
        return
    
    # Obtener columnas del modelo
    if hasattr(modelo, 'feature_names_in_'):
        columnas_modelo = list(modelo.feature_names_in_)
    else:
        columnas_modelo = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                          'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                          'key', 'mode', 'scale', 'non_naturales_notes']
    
    # Tabs para subir o grabar
    tab1, tab2 = st.tabs(["Subir Archivo", "Grabar Audio"])
    
    archivo = None
    
    with tab1:
        archivo_subido = st.file_uploader("Seleccionar archivo de audio", type=['wav', 'mp3'])
        if archivo_subido:
            archivo = archivo_subido
    
    with tab2:
        st.write("Presiona el botón para iniciar/detener la grabación:")
        st.info("Puedes grabar hasta 2 minutos de audio. El sistema ajustará automáticamente el tamaño.")
        
        audio_bytes = audio_recorder(
            text="Haz clic para grabar",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="3x",
            sample_rate=44100,
            key="audio_recorder"
        )
        
        if audio_bytes:
            audio_cortado = cortar_audio(audio_bytes, TAMAÑO_OBJETIVO_MB)
            archivo = ArchivoAudio(audio_cortado, "grabacion.mp3")
            st.success(f"Audio grabado: {len(audio_cortado)/1024/1024:.2f} MB")
            st.audio(audio_cortado, format='audio/mp3')
    
    if archivo:
        tamaño_mb = archivo.size / 1024 / 1024
        st.write(f"**Archivo:** {archivo.name} ({tamaño_mb:.2f} MB)")
        
        archivo_final = archivo
        if tamaño_mb > 5.0:
            audio_cortado = cortar_audio(archivo.getvalue(), TAMAÑO_OBJETIVO_MB)
            archivo_final = ArchivoAudio(audio_cortado, archivo.name)
            st.info(f"Archivo ajustado a {len(audio_cortado)/1024/1024:.2f} MB")
        
        if st.button("Analizar", type="primary", use_container_width=True):
            
            with st.spinner("Procesando..."):
                features_reccobeats, exito = analizar_con_reccobeats(archivo_final)
                
                if not exito or not features_reccobeats:
                    st.error("Error en el análisis del audio.")
                    return
                
                key_mode_info = extraer_key_mode(archivo_final.getvalue(), archivo_final.name)
                
                features_completas = features_reccobeats.copy()
                features_completas["key"] = key_mode_info["key"]
                features_completas["mode"] = key_mode_info["mode"]
                features_completas = agregar_escalas(features_completas, 
                                                     key_mode_info["key"], 
                                                     key_mode_info["mode"])
            
            st.success("Análisis completado")
            
            st.subheader("Características Musicales")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tonalidad", f"{key_mode_info['key_name']} {key_mode_info['mode_name']}")
                st.metric("Tempo", f"{features_completas.get('tempo', 0):.1f} BPM")
                st.metric("Energy", f"{features_completas.get('energy', 0):.3f}")
            
            with col2:
                st.metric("Danceability", f"{features_completas.get('danceability', 0):.3f}")
                st.metric("Valence", f"{features_completas.get('valence', 0):.3f}")
                st.metric("Loudness", f"{features_completas.get('loudness', 0):.1f} dB")
            
            with col3:
                st.metric("Acousticness", f"{features_completas.get('acousticness', 0):.3f}")
                st.metric("Instrumentalness", f"{features_completas.get('instrumentalness', 0):.3f}")
                st.metric("Speechiness", f"{features_completas.get('speechiness', 0):.3f}")
            
            st.subheader("Predicción")
            
            try:
                df_prediccion = crear_dataframe_prediccion(features_completas, columnas_modelo)
                prediccion = modelo.predict(df_prediccion)[0]
                probabilidades = modelo.predict_proba(df_prediccion)[0]
                
                st.markdown(f"### Artista Predicho: **{prediccion}**")
                
                # Mostrar probabilidades
                clases = modelo.classes_
                prob_dict = dict(zip(clases, probabilidades))
                prob_sorted = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                
                st.write("**Probabilidades:**")
                col_prob1, col_prob2 = st.columns(2)
                for i, (artista, prob) in enumerate(prob_sorted):
                    if i < len(prob_sorted) // 2:
                        with col_prob1:
                            st.write(f"- {artista}: {prob*100:.2f}%")
                    else:
                        with col_prob2:
                            st.write(f"- {artista}: {prob*100:.2f}%")
                
                # Análisis de la Predicción
                st.subheader("Análisis de la Predicción")
                
                # Cargar dataset para el análisis
                df_dataset = cargar_dataset()
                if df_dataset is not None:
                    try:
                        import shap
                        
                        # Extraer el modelo del Pipeline
                        if hasattr(modelo, 'named_steps'):
                            # Es un Pipeline, extraer el modelo
                            actual_model = modelo.named_steps['model']
                            # Transformar los datos usando el preprocessing del pipeline
                            X_pred_transformed = modelo.named_steps['preprocessing'].transform(df_prediccion)
                        else:
                            actual_model = modelo
                            X_pred_transformed = df_prediccion.values
                        
                        # Crear explicador SHAP
                        explainer_shap = shap.TreeExplainer(actual_model)
                        
                        # Calcular valores SHAP
                        shap_values = explainer_shap.shap_values(X_pred_transformed)
                        
                        # Obtener el índice de la clase predicha
                        pred_idx = list(clases).index(prediccion)
                        
                        # Si shap_values es una lista (multiclase), tomar la clase predicha
                        if isinstance(shap_values, list):
                            shap_vals = shap_values[pred_idx][0]
                        else:
                            shap_vals = shap_values[0]
                        
                        # Asegurar que shap_vals sea un array 1D
                        shap_vals_flat = np.array(shap_vals).flatten()
                        
                        # Obtener nombres de características después del preprocessing
                        if hasattr(modelo.named_steps['preprocessing'], 'get_feature_names_out'):
                            feature_names_transformed = list(modelo.named_steps['preprocessing'].get_feature_names_out())
                        else:
                            feature_names_transformed = [f"feature_{i}" for i in range(len(shap_vals_flat))]
                        
                        # Asegurar que ambos arrays tengan la misma longitud
                        min_len = min(len(feature_names_transformed), len(shap_vals_flat))
                        feature_names_transformed = feature_names_transformed[:min_len]
                        shap_vals_flat = shap_vals_flat[:min_len]
                        
                        # Crear DataFrame para el gráfico
                        df_shap = pd.DataFrame({
                            'Feature': feature_names_transformed,
                            'SHAP Value': shap_vals_flat
                        })
                        
                        # Simplificar nombres de características
                        df_shap['Feature'] = df_shap['Feature'].str.replace('remainder__', '').str.replace('num__', '').str.replace('cat__', '')
                        
                        # Separar escalas de otras características
                        df_scales = df_shap[df_shap['Feature'].str.contains('scale_', case=False, na=False)].copy()
                        df_non_scales = df_shap[~df_shap['Feature'].str.contains('scale_', case=False, na=False)].copy()
                        
                        # Si hay escalas, tomar solo la más importante
                        if len(df_scales) > 0:
                            df_scales['abs_shap'] = df_scales['SHAP Value'].abs()
                            top_scale = df_scales.nlargest(1, 'abs_shap')
                            
                            # Obtener el nombre de la escala y verificar si está presente
                            scale_feature_name = top_scale['Feature'].iloc[0]
                            scale_col_idx = list(feature_names_transformed).index(scale_feature_name)
                            scale_value = X_pred_transformed[0, scale_col_idx] if hasattr(X_pred_transformed, 'shape') else X_pred_transformed[0][scale_col_idx]
                            
                            # Renombrar según si está presente o no
                            scale_name = scale_feature_name.replace('scale_', '')
                            if scale_value == 1:
                                top_scale['Feature'] = f"Escala {scale_name}"
                            else:
                                top_scale['Feature'] = f"NO Escala {scale_name}"
                            
                            # Combinar con las no-escalas
                            df_shap_filtered = pd.concat([df_non_scales, top_scale], ignore_index=True)
                        else:
                            df_shap_filtered = df_non_scales.copy()
                        
                        # Mapear a nombres originales si es posible
                        feature_mapping = {
                            'acousticness': 'Acústica',
                            'danceability': 'Bailabilidad',
                            'energy': 'Energía',
                            'instrumentalness': 'Instrumental',
                            'speechiness': 'Voz',
                            'valence': 'Positividad',
                            'liveness': 'En Vivo',
                            'loudness': 'Volumen',
                            'tempo': 'Tempo',
                            'key': 'Tonalidad',
                            'mode': 'Modo',
                            'non_naturales_notes': 'Notas no naturales'
                        }
                        
                        # Aplicar mapeo
                        for orig, nuevo in feature_mapping.items():
                            df_shap_filtered['Feature'] = df_shap_filtered['Feature'].str.replace(orig, nuevo, regex=False)
                        
                        # Ordenar por valor absoluto
                        df_shap_filtered['abs_shap'] = df_shap_filtered['SHAP Value'].abs()
                        df_shap_filtered = df_shap_filtered.sort_values('abs_shap', ascending=True)
                        
                        # Limitar a top 10 características más importantes
                        df_shap_top = df_shap_filtered.tail(10).copy()
                        
                        # Crear etiqueta de impacto
                        df_shap_top['Impacto'] = df_shap_top['SHAP Value'].apply(
                            lambda x: 'Aumenta probabilidad' if x > 0 else 'Disminuye probabilidad'
                        )
                        
                        # Crear gráfico con Altair más explicativo
                        chart_shap = (
                            alt.Chart(df_shap_top)
                            .mark_bar(size=25)
                            .encode(
                                x=alt.X('SHAP Value:Q', 
                                       title='Impacto en la Predicción',
                                       axis=alt.Axis(grid=True)),
                                y=alt.Y('Feature:N', 
                                       title='Característica Musical',
                                       sort=None,
                                       axis=alt.Axis(labelLimit=200)),
                                color=alt.Color('Impacto:N',
                                               scale=alt.Scale(
                                                   domain=['Aumenta probabilidad', 'Disminuye probabilidad'],
                                                   range=['#2ca02c', '#d62728']
                                               ),
                                               legend=alt.Legend(title='Efecto')),
                                tooltip=[
                                    alt.Tooltip('Feature:N', title='Característica'),
                                    alt.Tooltip('SHAP Value:Q', title='Impacto', format='.4f'),
                                    alt.Tooltip('Impacto:N', title='Efecto')
                                ]
                            )
                            .properties(
                                title={
                                    "text": f"¿Por qué se predijo {prediccion}?",
                                    "subtitle": "Top 10 características que más influyeron en la decisión"
                                },
                                width=700,
                                height=450
                            )
                            .configure_axis(
                                labelFontSize=12,
                                titleFontSize=14
                            )
                            .configure_title(
                                fontSize=16,
                                anchor='start'
                            )
                        )
                        
                        st.altair_chart(chart_shap, use_container_width=True)
                        
                        # Explicación profesional
                        st.markdown("""
                        **Interpretación del análisis:**
                        - **Verde**: Características que aumentan la probabilidad de la predicción
                        - **Rojo**: Características que disminuyen la probabilidad de la predicción
                        - **Magnitud de las barras**: Indica el nivel de impacto en la decisión del modelo
                        """)
                        
                        # Mostrar resumen de las características más importantes
                        top_3 = df_shap_top.tail(3)
                        st.write("**Características más influyentes:**")
                        for idx, row in top_3.iloc[::-1].iterrows():
                            st.write(f"• **{row['Feature']}**: {row['Impacto']} (impacto: {abs(row['SHAP Value']):.4f})")
                        
                    except Exception as e:
                        st.error(f"Error al generar el análisis: {e}")
                
                with st.expander("Ver todas las características"):
                    st.dataframe(df_prediccion, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

def pagina_datos():
    """Página de análisis de datos"""
    st.title("Análisis de Datos")
    
    # Cargar dataset
    df = cargar_dataset()
    if df is None:
        st.error("No se pudo cargar el dataset.")
        return
    
    # Definir colores específicos para cada artista
    artist_colors = {
        'Todos los artistas': '#17becf',  # Turquesa/Cyan vibrante
        'Taylor Swift': '#e377c2',  # Rosa
        'Daft Punk': '#ff7f0e',  # Naranja
        'Guns N\' Roses': '#d62728',  # Rojo
        'Pink Floyd': '#9467bd',  # Púrpura
        'The Beatles': '#1f77b4',  # Azul
        'Metallica': '#2ca02c',  # Verde
        'One Direction': '#bcbd22',  # Amarillo verdoso
    }
    
    # Estadísticas básicas
    st.write(f"**Total de canciones:** {len(df)} | **Total de artistas:** {df['artist_name'].nunique()}")
    st.write("")
    
    # Preparar datos para el gráfico de torta
    artist_distribution = df['artist_name'].value_counts().reset_index()
    artist_distribution.columns = ['artist_name', 'count']
    artist_distribution['percentage'] = (artist_distribution['count'] / len(df) * 100).round(1)
    
    # Crear etiqueta con nombre, conteo y porcentaje
    artist_distribution['label'] = artist_distribution.apply(
        lambda row: f"{row['artist_name']}: {row['count']} ({row['percentage']}%)", 
        axis=1
    )
    
    # Crear gráfico de torta más grande
    pie_chart = (
        alt.Chart(artist_distribution)
        .mark_arc(innerRadius=80, outerRadius=200)
        .encode(
            theta=alt.Theta('count:Q', stack=True),
            color=alt.Color('artist_name:N',
                          title='Distribución por Artista',
                          scale=alt.Scale(
                              domain=[k for k in artist_colors.keys() if k != 'Todos los artistas'],
                              range=[v for k, v in artist_colors.items() if k != 'Todos los artistas']
                          ),
                          legend=alt.Legend(
                              labelExpr="datum.label",
                              labelLimit=300,
                              titleFontSize=14,
                              labelFontSize=12
                          )),
            tooltip=[
                alt.Tooltip('artist_name:N', title='Artista'),
                alt.Tooltip('count:Q', title='Canciones'),
                alt.Tooltip('percentage:Q', title='Porcentaje', format='.1f')
            ]
        )
        .transform_lookup(
            lookup='artist_name',
            from_=alt.LookupData(artist_distribution, 'artist_name', ['label'])
        )
        .properties(
            title={
                "text": "Distribución de Canciones por Artista",
                "fontSize": 18
            },
            width=600,
            height=500
        )
        .configure_legend(
            orient='right',
            titleFontSize=14,
            labelFontSize=12
        )
    )
    
    st.altair_chart(pie_chart, use_container_width=True)
    
    # Gráfico 1: Características musicales por artista
    st.subheader("Características Musicales por Artista")
    
    # Incluir todas las características musicales
    numeric_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                       'speechiness', 'valence', 'liveness', 'loudness', 'tempo']
    
    # Crear copia del dataframe para normalizar
    df_normalized = df.copy()
    
    # Normalizar PRIMERO cada valor individual entre 0 y 1
    for col in numeric_features:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 0.5
    
    # LUEGO calcular la media de los valores normalizados por artista
    means_by_artist = df_normalized.groupby('artist_name')[numeric_features].mean().reset_index()
    
    # Calcular la media global de los valores normalizados
    means_global = df_normalized[numeric_features].mean().to_frame().T
    means_global['artist_name'] = 'Todos los artistas'
    
    # Combinar ambos
    means = pd.concat([means_global, means_by_artist], ignore_index=True)
    
    # Pasar de formato ancho a largo
    means_long = means.melt(id_vars='artist_name', var_name='Feature', value_name='Mean')
    
    # Mapeo de nombres más legibles
    feature_names = {
        'acousticness': 'Acústica',
        'danceability': 'Bailabilidad',
        'energy': 'Energía',
        'instrumentalness': 'Instrumental',
        'speechiness': 'Voz',
        'valence': 'Positividad',
        'liveness': 'En Vivo',
        'loudness': 'Volumen',
        'tempo': 'Tempo'
    }
    means_long['Feature_Label'] = means_long['Feature'].map(feature_names)
    
    # Selector de artista usando Streamlit
    artist_options = ['Todos los artistas'] + sorted([a for a in means['artist_name'].unique() if a != 'Todos los artistas'])
    
    # Inicializar session_state
    if 'selected_artists_set' not in st.session_state:
        st.session_state.selected_artists_set = {'Todos los artistas'}
    
    if 'reset_version_g1' not in st.session_state:
        st.session_state.reset_version_g1 = 0
    
    # Botón de reset y expander
    col_exp, col_reset = st.columns([4, 1])
    
    with col_reset:
        if st.button("🔄 Resetear", key='reset_btn_graph1', use_container_width=True):
            st.session_state.selected_artists_set = {'Todos los artistas'}
            st.session_state.reset_version_g1 += 1
            st.rerun()
    
    # Lista desplegable con checkboxes
    with col_exp:
        with st.expander("📋 Seleccionar artistas para comparar", expanded=False):
            # Crear checkboxes en columnas
            num_cols = 3
            cols = st.columns(num_cols)
            
            # Usar reset_version en las keys para forzar recreación
            for idx, artist in enumerate(artist_options):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    # Verificar si está seleccionado
                    is_checked = artist in st.session_state.selected_artists_set
                    
                    # Crear checkbox con key única que incluye versión de reset
                    checked = st.checkbox(
                        artist,
                        value=is_checked,
                        key=f'cb_{artist}_g1_v{st.session_state.reset_version_g1}'
                    )
                    
                    # Actualizar el set según el estado del checkbox
                    if checked and artist not in st.session_state.selected_artists_set:
                        st.session_state.selected_artists_set.add(artist)
                    elif not checked and artist in st.session_state.selected_artists_set:
                        st.session_state.selected_artists_set.discard(artist)
    
    # Obtener lista de artistas seleccionados
    selected_artists = list(st.session_state.selected_artists_set)
    
    # Si no hay ninguno seleccionado, seleccionar "Todos los artistas"
    if not selected_artists:
        selected_artists = ['Todos los artistas']
        st.session_state.selected_artists_set = {'Todos los artistas'}
    
    # Mostrar artistas seleccionados
    st.caption(f"Artistas seleccionados: {', '.join(sorted(selected_artists))}")
    
    # Filtrar datos según los artistas seleccionados
    means_long_filtered = means_long[means_long['artist_name'].isin(selected_artists)].copy()
    
    # Crear gráfico de líneas con puntos
    chart1 = (
        alt.Chart(means_long_filtered)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X('Feature_Label:N', 
                   title='Características Musicales',
                   sort=list(feature_names.values()),
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Mean:Q', 
                   title='Valor Normalizado (0-1)', 
                   scale=alt.Scale(domain=[0, 1]),
                   axis=alt.Axis(grid=True)),
            color=alt.Color('artist_name:N', 
                          title='Artista',
                          scale=alt.Scale(
                              domain=list(artist_colors.keys()),
                              range=list(artist_colors.values())
                          )),
            tooltip=[
                alt.Tooltip('artist_name:N', title='Artista'),
                alt.Tooltip('Feature_Label:N', title='Característica'),
                alt.Tooltip('Mean:Q', title='Valor', format='.3f')
            ]
        )
        .properties(
            title={
                "text": "Comparación de Perfiles Musicales",
                "subtitle": "Valores normalizados de características musicales por artista"
            },
            width=800,
            height=450
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        .configure_title(
            fontSize=18,
            anchor='start'
        )
        .configure_point(
            size=100
        )
    )
    
    st.altair_chart(chart1, use_container_width=True)
    
    # Gráfico 2: Distribución de canciones por escala
    st.subheader("Distribución de Canciones por Escala Musical")
    
    # Agregar columna de modo legible
    df_with_mode = df.copy()
    df_with_mode['mode_name'] = df_with_mode['mode'].apply(lambda x: 'Mayor' if x == 1 else 'Menor')
    
    # Contar canciones por artista, escala y modo
    songs_by_scale_artist = df_with_mode.groupby(['artist_name', 'scale', 'mode_name']).size().reset_index(name='count')
    
    # Contar canciones por escala y modo (todos los artistas)
    songs_by_scale_global = df_with_mode.groupby(['scale', 'mode_name']).size().reset_index(name='count')
    songs_by_scale_global['artist_name'] = 'Todos los artistas'
    
    # Combinar ambos
    songs_by_scale = pd.concat([songs_by_scale_global, songs_by_scale_artist], ignore_index=True)
    
    # Ordenar por cantidad (de mayor a menor) para cada artista
    songs_by_scale = songs_by_scale.sort_values(['artist_name', 'count'], ascending=[True, False])
    
    # Opciones de artistas
    artist_options2 = ['Todos los artistas'] + sorted([a for a in songs_by_scale_artist['artist_name'].unique()])
    
    # Inicializar session_state para el artista si no existe
    if 'selected_artist_graph2' not in st.session_state:
        st.session_state.selected_artist_graph2 = 'Todos los artistas'
    
    # Crear selectores usando Streamlit
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        artist_filter_2 = st.selectbox(
            "Seleccionar artista:",
            artist_options2,
            index=artist_options2.index(st.session_state.selected_artist_graph2),
            key='artist_filter_graph2'
        )
        st.session_state.selected_artist_graph2 = artist_filter_2
    
    with col_filter2:
        mode_filter = st.selectbox(
            "Seleccionar modo:",
            ['Todos los modos', 'Mayor', 'Menor'],
            key='mode_filter'
        )
    
    # Filtrar datos según las selecciones
    songs_by_scale_filtered = songs_by_scale[songs_by_scale['artist_name'] == artist_filter_2].copy()
    
    if mode_filter != 'Todos los modos':
        songs_by_scale_filtered = songs_by_scale_filtered[songs_by_scale_filtered['mode_name'] == mode_filter].copy()
    
    # Crear gráfico de barras ordenado con colores específicos
    chart2 = (
        alt.Chart(songs_by_scale_filtered)
        .mark_bar()
        .encode(
            x=alt.X('scale:N', 
                   title='Escala Musical',
                   sort=alt.EncodingSortField(field='count', order='descending'),
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count:Q', 
                   title='Cantidad de Canciones',
                   axis=alt.Axis(grid=True)),
            color=alt.Color('artist_name:N', 
                          title='Artista',
                          scale=alt.Scale(
                              domain=list(artist_colors.keys()),
                              range=list(artist_colors.values())
                          ),
                          legend=None),
            tooltip=[
                alt.Tooltip('artist_name:N', title='Artista'),
                alt.Tooltip('scale:N', title='Escala'),
                alt.Tooltip('mode_name:N', title='Modo'),
                alt.Tooltip('count:Q', title='Canciones')
            ]
        )
        .properties(
            title={
                "text": "Distribución de Canciones por Escala Musical"
            },
            width=800,
            height=450
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        .configure_title(
            fontSize=18,
            anchor='start'
        )
    )
    
    st.altair_chart(chart2, use_container_width=True)
    
    # Gráfico 3: Histograma de características musicales
    st.subheader("Distribución de Características Musicales")
    
    # Características musicales disponibles
    numeric_features_hist = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                             'speechiness', 'valence', 'liveness', 'loudness', 'tempo']
    
    # Mapeo de nombres más legibles
    feature_names_hist = {
        'acousticness': 'Acústica',
        'danceability': 'Bailabilidad',
        'energy': 'Energía',
        'instrumentalness': 'Instrumental',
        'speechiness': 'Voz',
        'valence': 'Positividad',
        'liveness': 'En Vivo',
        'loudness': 'Volumen',
        'tempo': 'Tempo'
    }
    
    # Opciones de artistas
    artist_options3 = ['Todos los artistas'] + sorted(df['artist_name'].unique())
    
    # Inicializar session_state para el artista si no existe
    if 'selected_artist_graph3' not in st.session_state:
        st.session_state.selected_artist_graph3 = 'Todos los artistas'
    
    # Inicializar session_state para la característica si no existe
    if 'selected_feature_graph3' not in st.session_state:
        st.session_state.selected_feature_graph3 = 'energy'
    
    # Crear selectores usando Streamlit
    col_filter3_1, col_filter3_2 = st.columns(2)
    
    with col_filter3_1:
        artist_filter_3 = st.selectbox(
            "Seleccionar artista:",
            artist_options3,
            index=artist_options3.index(st.session_state.selected_artist_graph3),
            key='artist_filter_graph3'
        )
        st.session_state.selected_artist_graph3 = artist_filter_3
    
    with col_filter3_2:
        feature_filter = st.selectbox(
            "Seleccionar característica musical:",
            numeric_features_hist,
            format_func=lambda x: feature_names_hist[x],
            index=numeric_features_hist.index(st.session_state.selected_feature_graph3),
            key='feature_filter_graph3'
        )
        st.session_state.selected_feature_graph3 = feature_filter
    
    # Filtrar datos según el artista seleccionado
    if artist_filter_3 == 'Todos los artistas':
        df_hist = df.copy()
    else:
        df_hist = df[df['artist_name'] == artist_filter_3].copy()
    
    # Eliminar valores nulos o 0 de la característica seleccionada
    df_hist = df_hist[df_hist[feature_filter].notna()].copy()
    df_hist = df_hist[df_hist[feature_filter] > 0].copy()
    
    # Agregar columna con el nombre del artista para el color
    if artist_filter_3 == 'Todos los artistas':
        df_hist['artist_display'] = 'Todos los artistas'
    else:
        df_hist['artist_display'] = artist_filter_3
    
    # Crear histograma con colores específicos
    chart3 = (
        alt.Chart(df_hist)
        .mark_bar()
        .encode(
            x=alt.X(f'{feature_filter}:Q',
                   title=feature_names_hist[feature_filter],
                   bin=alt.Bin(maxbins=30)),
            y=alt.Y('count():Q',
                   title='Frecuencia',
                   axis=alt.Axis(grid=True)),
            color=alt.Color('artist_display:N',
                          title='Artista',
                          scale=alt.Scale(
                              domain=list(artist_colors.keys()),
                              range=list(artist_colors.values())
                          ),
                          legend=None),
            tooltip=[
                alt.Tooltip('artist_display:N', title='Artista'),
                alt.Tooltip(f'{feature_filter}:Q', title=feature_names_hist[feature_filter], bin=True),
                alt.Tooltip('count():Q', title='Frecuencia')
            ]
        )
        .properties(
            title={
                "text": f"Distribución de {feature_names_hist[feature_filter]}"
            },
            width=800,
            height=450
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        .configure_title(
            fontSize=18,
            anchor='start'
        )
    )
    
    st.altair_chart(chart3, use_container_width=True)
    
    # Mostrar estadísticas básicas
    if len(df_hist) > 0:
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric("Total de canciones", len(df_hist))
        
        with col_stats2:
            st.metric("Media", f"{df_hist[feature_filter].mean():.3f}")
        
        with col_stats3:
            st.metric("Mediana", f"{df_hist[feature_filter].median():.3f}")
        
        with col_stats4:
            st.metric("Desv. Estándar", f"{df_hist[feature_filter].std():.3f}")
    else:
        st.warning("No hay datos disponibles para esta combinación de filtros.")

def pagina_explicacion():
    """Documentación técnica del modelo de clasificación"""
    st.title("Documentación Técnica del Modelo")
    
    st.markdown("""
    Este documento describe el funcionamiento, características y rendimiento del sistema de clasificación 
    de artistas musicales basado en análisis de señales de audio.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "El Modelo", 
        "Características",
        "Rendimiento",
        "Limitaciones"
    ])
    
    with tab1:
        st.header("Sistema de Predicción de Artistas Musicales")
        
        st.markdown("""
        ### Funcionalidades de la Aplicación
        
        Esta aplicación ofrece dos módulos principales para el análisis y predicción de música:
        """)
        
        st.markdown("""
        **1. Predictor de Artistas**
        
        Permite subir o grabar audio para predecir el artista entre 7 opciones:
        Taylor Swift, The Beatles, Metallica, Pink Floyd, Guns N' Roses, Daft Punk y One Direction.
        """)
        
        st.markdown("""
        El sistema analiza las características musicales del audio y proporciona:
        - Predicción del artista más probable
        - Probabilidades para cada artista
        - Análisis de qué características influyeron en la decisión
        """)
        
        st.markdown("""
        **2. Análisis de Datos**
        
        Visualización interactiva del dataset de entrenamiento con gráficos de distribución, 
        comparación de perfiles musicales entre artistas y análisis de características.
        """)
        
        st.markdown("---")
        
        st.subheader("Modelo")
        
        st.markdown("""
        ### Random Forest Classifier
        
        El modelo utiliza **Random Forest**, un algoritmo de machine learning que combina múltiples 
        árboles de decisión para realizar predicciones robustas.
        
        **Proceso de predicción:**
        1. Extracción de 13 características musicales del audio
        2. Normalización y preprocesamiento de datos
        3. Análisis mediante Random Forest
        4. Cálculo de probabilidades para cada artista
        5. Explicación de la decisión mediante análisis de importancia
        """)
    
    with tab2:
        st.header("Características del Modelo")
        
        st.markdown("""
        El modelo analiza 13 características extraídas mediante análisis de señales de audio. 
        Estas características capturan propiedades acústicas, rítmicas y tonales de las canciones.
        """)
        
        st.subheader("Descripción de Características")
        
        # Crear tabla de características con descripciones más detalladas
        features_data = {
            'Característica': [
                'Acústica', 'Bailabilidad', 'Energía', 'Instrumental', 
                'Voz Hablada', 'Valencia', 'En Vivo', 'Volumen',
                'Tempo', 'Tonalidad', 'Modo', 'Escala', 'Alteraciones'
            ],
            'Descripción': [
                'Mide la probabilidad de que la canción sea acústica, sin instrumentos electrónicos. Valores altos indican uso de instrumentos tradicionales',
                'Evalúa qué tan adecuada es una canción para bailar, considerando tempo, estabilidad del ritmo y fuerza del beat',
                'Representa la intensidad y actividad percibida. Canciones energéticas se sienten rápidas, fuertes y ruidosas',
                'Predice si una canción no contiene voces. Valores superiores a 0.5 indican pistas principalmente instrumentales',
                'Detecta la presencia de palabras habladas. Valores altos indican contenido hablado como rap, podcast o audiolibros',
                'Describe la positividad musical transmitida. Valores altos suenan alegres y eufóricos, valores bajos suenan tristes',
                'Detecta la presencia de audiencia en la grabación. Valores superiores a 0.8 indican alta probabilidad de grabación en vivo',
                'Volumen general de la canción medido en decibeles. Valores cercanos a 0 indican mayor volumen',
                'Velocidad o ritmo de la canción medido en beats por minuto (BPM). Indica qué tan rápida o lenta es la canción',
                'La tonalidad en la que está la canción usando notación Pitch Class (0=C, 1=C#, 2=D, etc.)',
                'Indica la modalidad de la escala musical. 1 representa modo Mayor (generalmente alegre), 0 representa Menor (melancólico)',
                'Combinación de tonalidad y modo que define el conjunto de notas utilizadas (ej: C Mayor, A Menor, F# Mayor)',
                'Cantidad de sostenidos o bemoles en la armadura de clave. 0 indica sin alteraciones (C Mayor, A Menor)'
            ],
            'Rango': [
                '0.0 - 1.0', '0.0 - 1.0', '0.0 - 1.0', '0.0 - 1.0',
                '0.0 - 1.0', '0.0 - 1.0', '0.0 - 1.0', '-60 a 0 dB',
                'Variable BPM', '0 - 11', '0 o 1', 'Categórica', '0 - 5'
            ]
        }
        
        df_features = pd.DataFrame(features_data)
        
        # Mostrar tabla sin scroll
        st.markdown(df_features.to_html(index=False, escape=False), unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Importancia de Características")
        
        st.markdown("""
        Las siguientes son las 10 características más relevantes para la clasificación, 
        ordenadas por su importancia en el modelo:
        """)
        
        # Crear DataFrame con las importancias
        importances_data = {
            'Característica': [
                'Volumen', 'Instrumental', 'Energía', 'Acústica', 
                'Bailabilidad', 'Valencia', 'Voz Hablada', 'En Vivo',
                'Alteraciones', 'Tempo'
            ],
            'Importancia': [
                0.1573, 0.1293, 0.1161, 0.1135, 0.1036, 0.0893, 
                0.0611, 0.0464, 0.0442, 0.0362
            ]
        }
        
        df_imp = pd.DataFrame(importances_data)
        
        # Gráfico de importancias
        chart_imp = (
            alt.Chart(df_imp)
            .mark_bar()
            .encode(
                x=alt.X('Importancia:Q', title='Importancia Relativa'),
                y=alt.Y('Característica:N', sort='-x', title=''),
                color=alt.value('#1f77b4'),
                tooltip=['Característica', alt.Tooltip('Importancia:Q', format='.4f')]
            )
            .properties(
                title='Top 10 Características Más Importantes',
                height=400
            )
        )
        
        st.altair_chart(chart_imp, use_container_width=True)
    
    with tab3:
        st.header("Métricas de Rendimiento")
        
        st.markdown("""
        ### Evaluación en Conjunto de Test
        
        El modelo fue evaluado en un conjunto de test independiente (257 muestras, 20% del dataset total) 
        que no fue utilizado durante el entrenamiento.
        """)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Accuracy", "82.88%")
        
        with col_m2:
            st.metric("Precision (macro)", "80.44%")
        
        with col_m3:
            st.metric("Recall (macro)", "80.48%")
        
        with col_m4:
            st.metric("F1-Score (macro)", "80.05%")
        
        st.markdown("---")
        
        st.subheader("Definición de Métricas")
        
        col_def1, col_def2 = st.columns(2)
        
        with col_def1:
            st.markdown("""
            **Accuracy (Exactitud)**
            
            Porcentaje de predicciones correctas sobre el total de predicciones realizadas.
            """)
        
        with col_def2:
            st.markdown("""
            **Recall (Sensibilidad)**
            
            De todos los casos reales de una clase, cuántos fueron identificados correctamente. 
            Mide la capacidad del modelo para encontrar todos los casos positivos.
            """)
        
        st.markdown("---")
        
        col_def3, col_def4 = st.columns(2)
        
        with col_def3:
            st.markdown("""
            **Precision (Precisión)**
            
            De todas las predicciones positivas para una clase, cuántas fueron correctas. 
            Mide la calidad de las predicciones positivas.
            """)
        
        with col_def4:
            st.markdown("""
            **F1-Score**
            
            Media armónica entre Precision y Recall. Proporciona un balance entre ambas métricas.
            """)
        
        st.markdown("---")
        
        st.subheader("Rendimiento por Clase")
        
        performance_data = {
            'Artista': ['Metallica', 'Pink Floyd', 'Taylor Swift', 'One Direction', 
                       'Guns N\' Roses', 'The Beatles', 'Daft Punk'],
            'Precision': [0.898, 0.744, 0.855, 0.750, 0.889, 0.864, 0.632],
            'Recall': [0.957, 0.906, 0.903, 0.818, 0.727, 0.691, 0.632],
            'F1-Score': [0.926, 0.817, 0.878, 0.783, 0.800, 0.768, 0.632]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        st.dataframe(
            df_perf.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}'
            }).background_gradient(subset=['F1-Score'], cmap='RdYlGn', vmin=0.6, vmax=1.0),
            use_container_width=True
        )
        
        st.caption("🟢 Verde: Mejor rendimiento | 🟡 Amarillo: Rendimiento medio | 🔴 Rojo: Menor rendimiento")
    
    with tab4:
        st.header("Limitaciones del Sistema")
        
        st.markdown("""
        ### Restricciones de la API de Análisis
        
        El sistema utiliza la API de **ReccoBeats** para extraer las características musicales del audio. 
        Esta API tiene una limitación importante que afecta la precisión del análisis:
        """)
        
        st.warning("⚠️ La API de ReccoBeats solo acepta archivos de hasta **5 MB**")
        
        st.markdown("""
        ### Impacto en la Predicción
        
        Cuando subes o grabas una canción que excede este límite, el sistema automáticamente:
        
        1. **Recorta el audio** a aproximadamente 4.9 MB para cumplir con la restricción
        2. **Analiza solo una porción** de la canción completa
        3. **Extrae características** basándose en este fragmento reducido
        
        #### Consecuencias:
        
        - **Análisis incompleto**: Las características musicales se calculan sobre una parte de la canción, 
          no sobre su totalidad
        - **Pérdida de contexto**: Secciones importantes como el coro, puente o final pueden quedar fuera del análisis
        - **Variabilidad en resultados**: Canciones con cambios dinámicos significativos pueden no ser 
          representadas adecuadamente
        - **Menor precisión**: La predicción del artista puede ser menos exacta al no considerar 
          la canción completa
        
        El modelo fue entrenado con características extraídas de canciones completas. Por lo tanto, 
        el análisis de fragmentos puede generar predicciones con menor confianza o precisión comparado 
        con el rendimiento reportado en las métricas de evaluación.
        """)


def main():
    # Sidebar para navegación
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='margin: 0;'>🎵 Predictor Musical</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # CSS para forzar botones al 100% del ancho
        st.markdown("""
        <style>
        .stButton button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Inicializar estado de navegación
        if 'pagina_actual' not in st.session_state:
            st.session_state.pagina_actual = 'Explicación del Modelo'
        
        # Botones de navegación (orden cambiado)
        if st.button("Explicación del Modelo", use_container_width=True, type="primary" if st.session_state.pagina_actual == 'Explicación del Modelo' else "secondary"):
            st.session_state.pagina_actual = 'Explicación del Modelo'
            st.rerun()
        
        if st.button("Análisis de Datos", use_container_width=True, type="primary" if st.session_state.pagina_actual == 'Análisis de Datos' else "secondary"):
            st.session_state.pagina_actual = 'Análisis de Datos'
            st.rerun()
        
        if st.button("Predictor", use_container_width=True, type="primary" if st.session_state.pagina_actual == 'Predictor' else "secondary"):
            st.session_state.pagina_actual = 'Predictor'
            st.rerun()
    
    # Mostrar página según selección
    if st.session_state.pagina_actual == "Predictor":
        pagina_predictor()
    elif st.session_state.pagina_actual == "Análisis de Datos":
        pagina_datos()
    else:
        pagina_explicacion()

if __name__ == "__main__":
    main()
