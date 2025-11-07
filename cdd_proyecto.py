import streamlit as st
import pandas as pd
import joblib
import requests
import warnings
import os
import numpy as np
import tempfile
from audio_recorder_streamlit import audio_recorder

# Configuración
st.set_page_config(page_title="Predictor Musical", page_icon="🎵", layout="wide")
warnings.filterwarnings('ignore')

TAMAÑO_OBJETIVO_MB = 4.9
MODELO_PATH = 'modelo.joblib'

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

def main():
    st.title("Predictor Musical")
    
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
            # Cortar audio grabado
            audio_cortado = cortar_audio(audio_bytes, TAMAÑO_OBJETIVO_MB)
            archivo = ArchivoAudio(audio_cortado, "grabacion.mp3")
            st.success(f"Audio grabado: {len(audio_cortado)/1024/1024:.2f} MB")
            
            # Mostrar reproductor
            st.audio(audio_cortado, format='audio/mp3')
    
    if archivo:
        # Información del archivo
        tamaño_mb = archivo.size / 1024 / 1024
        st.write(f"**Archivo:** {archivo.name} ({tamaño_mb:.2f} MB)")
        
        # Procesar si es necesario
        archivo_final = archivo
        if tamaño_mb > 5.0:
            audio_cortado = cortar_audio(archivo.getvalue(), TAMAÑO_OBJETIVO_MB)
            archivo_final = ArchivoAudio(audio_cortado, archivo.name)
            st.info(f"Archivo ajustado a {len(audio_cortado)/1024/1024:.2f} MB")
        
        # Botón de análisis
        if st.button("Analizar", type="primary", use_container_width=True):
            
            with st.spinner("Procesando..."):
                # Análisis con ReccoBeats
                features_reccobeats, exito = analizar_con_reccobeats(archivo_final)
                
                if not exito or not features_reccobeats:
                    st.error("Error en el análisis del audio.")
                    return
                
                # Extraer key y mode
                key_mode_info = extraer_key_mode(archivo_final.getvalue(), archivo_final.name)
                
                # Combinar features
                features_completas = features_reccobeats.copy()
                features_completas["key"] = key_mode_info["key"]
                features_completas["mode"] = key_mode_info["mode"]
                features_completas = agregar_escalas(features_completas, 
                                                     key_mode_info["key"], 
                                                     key_mode_info["mode"])
            
            # Mostrar resultados
            st.success("Análisis completado")
            
            # Características musicales
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
            
            # Predicción
            st.subheader("Predicción")
            
            try:
                df_prediccion = crear_dataframe_prediccion(features_completas, columnas_modelo)
                prediccion = modelo.predict(df_prediccion)[0]
                
                st.markdown(f"### Artista Predicho: **{prediccion}**")
                
                # Tabla de features
                with st.expander("Ver todas las características"):
                    st.dataframe(df_prediccion, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

if __name__ == "__main__":
    main()