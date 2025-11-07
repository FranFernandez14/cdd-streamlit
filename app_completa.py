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
                
                st.markdown(f"### Artista Predicho: **{prediccion}**")
                
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
    
    st.write(f"**Total de canciones:** {len(df)}")
    st.write(f"**Total de artistas:** {df['artist_name'].nunique()}")
    
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
    
    # Inicializar session_state para el artista si no existe
    if 'selected_artist_graph1' not in st.session_state:
        st.session_state.selected_artist_graph1 = 'Todos los artistas'
    
    artist_filter_1 = st.selectbox(
        "Seleccionar artista:",
        artist_options,
        index=artist_options.index(st.session_state.selected_artist_graph1),
        key='artist_filter_graph1'
    )
    st.session_state.selected_artist_graph1 = artist_filter_1
    
    # Filtrar datos según el artista seleccionado
    means_long_filtered = means_long[means_long['artist_name'] == artist_filter_1].copy()
    
    # Crear gráfico con colores específicos
    chart1 = (
        alt.Chart(means_long_filtered)
        .mark_bar(size=40)
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
                          ),
                          legend=None),
            tooltip=[
                alt.Tooltip('artist_name:N', title='Artista'),
                alt.Tooltip('Feature_Label:N', title='Característica'),
                alt.Tooltip('Mean:Q', title='Valor', format='.3f')
            ]
        )
        .properties(
            title={
                "text": "Perfil Musical por Artista",
                "subtitle": "Valores normalizados de características musicales"
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
                "text": "Distribución de Canciones por Escala Musical",
                "subtitle": f"Artista: {artist_filter_2} - Modo: {mode_filter}"
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
                "text": f"Distribución de {feature_names_hist[feature_filter]}",
                "subtitle": f"Artista: {artist_filter_3} (valores nulos y ceros excluidos)"
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

def main():
    # Sidebar para navegación
    with st.sidebar:
        st.title("Navegación")
        opcion = st.radio(
            "Selecciona una opción:",
            ["Predictor", "Análisis de Datos"],
            label_visibility="collapsed"
        )
    
    # Mostrar página según selección
    if opcion == "Predictor":
        pagina_predictor()
    else:
        pagina_datos()

if __name__ == "__main__":
    main()
