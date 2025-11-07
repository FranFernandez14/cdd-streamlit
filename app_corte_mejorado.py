import streamlit as st
import pandas as pd
import joblib
import requests
import warnings
import os
import io
import struct
import numpy as np

# Configuración
st.set_page_config(page_title="🎵 Predictor - Corte Mejorado", page_icon="🎵", layout="wide")
warnings.filterwarnings('ignore')

def obtener_tiempo_modificacion(archivo):
    try:
        return os.path.getmtime(archivo)
    except:
        return 0

@st.cache_resource
def cargar_modelo_con_cache(archivo_modelo, tiempo_modificacion):
    try:
        modelo = joblib.load(archivo_modelo)
        return modelo, None
    except Exception as e:
        return None, str(e)

def cargar_modelo_dinamico(archivo_modelo='modelo.joblib'):
    tiempo_mod = obtener_tiempo_modificacion(archivo_modelo)
    return cargar_modelo_con_cache(archivo_modelo, tiempo_mod)

def obtener_columnas_modelo(modelo):
    try:
        if hasattr(modelo, 'feature_names_in_'):
            return list(modelo.feature_names_in_)
        if hasattr(modelo, 'steps'):
            for name, step in modelo.steps:
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
        return ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    except Exception:
        return ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

def estimate_key_mode(y, sr):
    """
    Estima key y mode usando el algoritmo Krumhansl-Schmuckler
    """
    try:
        import librosa
        
        # Calcula el croma promedio
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Plantillas de tonalidades mayores y menores (Krumhansl-Schmuckler)
        maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Correlacionar el perfil de cromas con cada posible rotación de la escala
        corr_major = [np.corrcoef(np.roll(maj_profile, i), chroma_mean)[0, 1] for i in range(12)]
        corr_minor = [np.corrcoef(np.roll(min_profile, i), chroma_mean)[0, 1] for i in range(12)]
        
        # Tomar la máxima correlación entre mayor y menor
        key_major = np.argmax(corr_major)
        key_minor = np.argmax(corr_minor)
        
        if max(corr_major) >= max(corr_minor):
            return key_major, 1, max(corr_major)  # 1 = mayor
        else:
            return key_minor, 0, max(corr_minor)  # 0 = menor
            
    except Exception as e:
        # Fallback en caso de error
        return 0, 1, 0.5  # C mayor por defecto

def extraer_key_mode_librosa(archivo_bytes, nombre_archivo):
    """
    Extrae SOLO key y mode usando librosa con algoritmo Krumhansl-Schmuckler
    """
    try:
        import librosa
        import tempfile
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{nombre_archivo.split('.')[-1]}") as temp_file:
            temp_file.write(archivo_bytes)
            temp_path = temp_file.name
        
        try:
            # Cargar audio
            y, sr = librosa.load(temp_path, sr=None, mono=True)
            
            if len(y) == 0:
                return None, "Audio vacío", {}
            
            # Key y Mode con algoritmo Krumhansl-Schmuckler
            key, mode, confidence = estimate_key_mode(y, sr)
            
            # Información sobre key/mode
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_info = {
                "key": int(key),
                "mode": int(mode),
                "key_name": key_names[key],
                "mode_name": "Mayor" if mode == 1 else "Menor",
                "confidence": float(confidence)
            }
            
            return key_info, f"Key/Mode detectado: {key_info['key_name']} {key_info['mode_name']} (confianza: {confidence:.3f})", key_info
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except ImportError:
        return None, "Librosa no está instalado", {}
    except Exception as e:
        return None, f"Error en detección de key/mode: {str(e)}", {}

def agregar_features_escalas(features_reccobeats, key_mode_info):
    """
    Combina features de ReccoBeats con key/mode de librosa y agrega scale y non_naturales_notes
    """
    # Tabla de escalas
    data_escala = [
        ["C Mayor", 0, 1, 0],
        ["Db Mayor", 1, 1, 5],
        ["D Mayor", 2, 1, 2],
        ["Eb Mayor", 3, 1, 3],
        ["E Mayor", 4, 1, 4],
        ["F Mayor", 5, 1, 1],
        ["F# Mayor", 6, 1, 5],
        ["G Mayor", 7, 1, 1],
        ["Ab Mayor", 8, 1, 4],
        ["A Mayor", 9, 1, 3],
        ["Bb Mayor", 10, 1, 2],
        ["B Mayor", 11, 1, 5],
        ["A Menor", 9, 0, 0],
        ["Bb Menor", 10, 0, 5],
        ["B Menor", 11, 0, 2],
        ["C Menor", 0, 0, 3],
        ["C# Menor", 1, 0, 4],
        ["D Menor", 2, 0, 1],
        ["D# Menor", 3, 0, 5],
        ["E Menor", 4, 0, 1],
        ["F Menor", 5, 0, 4],
        ["F# Menor", 6, 0, 3],
        ["G Menor", 7, 0, 2],
        ["G# Menor", 8, 0, 5]
    ]
    
    # Crear DataFrame auxiliar
    df_escala = pd.DataFrame(data_escala, columns=["scale", "key", "mode", "non_naturales_notes"])
    
    # Combinar features
    features_combinadas = features_reccobeats.copy()
    
    # Agregar key y mode de librosa
    if key_mode_info:
        features_combinadas["key"] = key_mode_info["key"]
        features_combinadas["mode"] = key_mode_info["mode"]
        
        # Buscar en la tabla de escalas
        key_val = key_mode_info["key"]
        mode_val = key_mode_info["mode"]
        
        # Encontrar la fila correspondiente
        fila_escala = df_escala[(df_escala["key"] == key_val) & (df_escala["mode"] == mode_val)]
        
        if not fila_escala.empty:
            features_combinadas["scale"] = fila_escala.iloc[0]["scale"]
            features_combinadas["non_naturales_notes"] = fila_escala.iloc[0]["non_naturales_notes"]
        else:
            # Valores por defecto si no se encuentra
            features_combinadas["scale"] = "C Mayor"
            features_combinadas["non_naturales_notes"] = 0
    else:
        # Valores por defecto si no hay key/mode
        features_combinadas["key"] = 0
        features_combinadas["mode"] = 1
        features_combinadas["scale"] = "C Mayor"
        features_combinadas["non_naturales_notes"] = 0
    
    return features_combinadas

def cortar_por_bytes_inteligente(archivo_bytes, nombre_archivo, tamaño_objetivo_mb):
    """Corte inteligente manteniendo calidad"""
    tamaño_objetivo_bytes = tamaño_objetivo_mb * 1024 * 1024
    tamaño_original = len(archivo_bytes)
    
    if tamaño_original <= tamaño_objetivo_bytes:
        return archivo_bytes, ["No necesita corte"], tamaño_original
    
    proporcion_objetivo = tamaño_objetivo_bytes / tamaño_original * 0.95
    nuevo_tamaño = int(tamaño_original * proporcion_objetivo)
    archivo_cortado = archivo_bytes[:nuevo_tamaño]
    
    cambios = [f"Cortado a {proporcion_objetivo*100:.1f}% del tamaño original"]
    return archivo_cortado, cambios, len(archivo_cortado)

def analizar_con_reccobeats(archivo_audio):
    """Análisis con API ReccoBeats"""
    try:
        files = {'audioFile': (archivo_audio.name, archivo_audio.getvalue(), archivo_audio.type)}
        response = requests.post('https://api.reccobeats.com/v1/analysis/audio-features', files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json(), "API ReccoBeats exitosa"
        else:
            return None, f"Error API: {response.status_code}"
    except Exception as e:
        return None, f"Error: {e}"

def crear_dataframe_prediccion(caracteristicas_audio, columnas_modelo):
    datos = {}
    for col in columnas_modelo:
        datos[col] = caracteristicas_audio.get(col, 0)
    return pd.DataFrame([datos])

def mostrar_comparacion_features(features_reccobeats, features_combinadas, key_mode_info):
    """Muestra comparación entre ReccoBeats y features combinadas"""
    
    st.subheader("📊 Features por Fuente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🌐 ReccoBeats API:**")
        for key, value in features_reccobeats.items():
            if isinstance(value, float):
                st.write(f"• {key}: {value:.3f}")
            else:
                st.write(f"• {key}: {value}")
    
    with col2:
        st.write("**🔬 Agregadas por Librosa:**")
        if key_mode_info:
            st.write(f"• key: {key_mode_info['key']} ({key_mode_info['key_name']})")
            st.write(f"• mode: {key_mode_info['mode']} ({key_mode_info['mode_name']})")
            st.write(f"• scale: {features_combinadas.get('scale', 'N/A')}")
            st.write(f"• non_naturales_notes: {features_combinadas.get('non_naturales_notes', 'N/A')}")
            st.write(f"• confidence: {key_mode_info['confidence']:.3f}")
        else:
            st.write("• No se pudieron detectar key/mode")
    
    return features_combinadas

class ArchivoCortado:
    def __init__(self, data, name_original):
        self.data = data
        base_name = name_original.split('.')[0]
        extension = name_original.split('.')[-1]
        self.name = f"{base_name}_cortado.{extension}"
        self.type = f'audio/{extension}'
        self.size = len(data)
    
    def getvalue(self):
        return self.data

def main():
    st.title("🎵 Predictor Musical - Análisis Mejorado")
    st.markdown("**ReccoBeats API + Librosa con Algoritmo Krumhansl-Schmuckler**")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Configuración")
        
        modelos = [f for f in os.listdir('.') if f.endswith('.joblib')]
        if modelos:
            archivo_modelo = st.selectbox("📁 Modelo:", modelos)
        else:
            st.error("❌ No hay archivos .joblib")
            return
        
        if st.button("🔄 Recargar"):
            st.cache_resource.clear()
            st.rerun()
        
        # Configuración
        st.header("✂️ Configuración")
        tamaño_objetivo = st.slider("Tamaño objetivo (MB)", 2.0, 4.9, 4.8, 0.1)
        
        # Estado del sistema
        st.header("🔧 Estado")
        try:
            import librosa
            st.success("✅ Librosa disponible")
        except ImportError:
            st.error("❌ Librosa no instalado")
            st.code("pip install librosa")
    
    # Cargar modelo
    modelo, error = cargar_modelo_dinamico(archivo_modelo)
    
    if modelo is None:
        st.error(f"❌ Error: {error}")
        return
    
    st.success(f"✅ Modelo: {archivo_modelo}")
    
    # Detectar columnas
    columnas_modelo = obtener_columnas_modelo(modelo)
    
    with st.sidebar:
        st.header("📊 Info del Modelo")
        st.write(f"Columnas: {len(columnas_modelo)}")
        with st.expander("Ver columnas"):
            for col in columnas_modelo:
                st.write(f"• {col}")
    
    # Información sobre métodos
    st.info("""
    🎯 **Método Híbrido:**
    • **ReccoBeats API**: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence
    • **Librosa + Krumhansl-Schmuckler**: key, mode (algoritmo científico)
    • **Tabla de escalas**: scale, non_naturales_notes (basado en key/mode)
    """)
    
    # Upload
    st.header("📁 Subir Audio")
    archivo = st.file_uploader("Archivo de audio", type=['wav', 'mp3'])
    
    if archivo:
        tamaño_original = archivo.size
        tamaño_limite = 5 * 1024 * 1024
        
        # Info del archivo
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("📄 Archivo", archivo.name)
        with col2: st.metric("📏 Tamaño", f"{tamaño_original/1024/1024:.1f} MB")
        with col3: st.metric("🎵 Formato", archivo.name.split('.')[-1].upper())
        
        archivo_final = archivo
        
        # Cortar si es necesario
        if tamaño_original > tamaño_limite:
            st.warning(f"⚠️ Archivo muy grande. Cortando...")
            
            with st.spinner("✂️ Cortando audio..."):
                audio_cortado, cambios, tamaño_final = cortar_por_bytes_inteligente(
                    archivo.getvalue(), archivo.name, tamaño_objetivo
                )
                
                if audio_cortado:
                    archivo_final = ArchivoCortado(audio_cortado, archivo.name)
                    reduccion = ((tamaño_original - tamaño_final) / tamaño_original) * 100
                    st.success(f"✅ Cortado: {reduccion:.1f}% reducción")
                else:
                    st.error("❌ No se pudo cortar")
                    return
        
        # Análisis híbrido
        st.header("🔍 Análisis Híbrido")
        
        if st.button("🚀 Analizar con Método Híbrido", type="primary", use_container_width=True):
            
            # Crear columnas para mostrar progreso
            col1, col2 = st.columns(2)
            
            # Análisis con ReccoBeats API
            with col1:
                st.subheader("🌐 ReccoBeats API")
                with st.spinner("Extrayendo features principales..."):
                    features_reccobeats, mensaje_api = analizar_con_reccobeats(archivo_final)
                
                if features_reccobeats:
                    st.success("✅ Features principales extraídas")
                else:
                    st.error(f"❌ {mensaje_api}")
            
            # Análisis de key/mode con Librosa
            with col2:
                st.subheader("🎼 Key/Mode Detection")
                with st.spinner("Detectando key y mode..."):
                    key_mode_data, mensaje_librosa, key_info = extraer_key_mode_librosa(
                        archivo_final.getvalue(), archivo_final.name
                    )
                
                if key_mode_data:
                    st.success("✅ Key/Mode detectados")
                    st.info(f"🎼 {key_info['key_name']} {key_info['mode_name']} (conf: {key_info['confidence']:.3f})")
                else:
                    st.error(f"❌ {mensaje_librosa}")
            
            # Combinar resultados si ambos funcionaron
            if features_reccobeats:
                
                # Combinar features
                features_combinadas = agregar_features_escalas(features_reccobeats, key_mode_data)
                
                # Mostrar comparación
                st.header("📊 Features Combinadas")
                mostrar_comparacion_features(features_reccobeats, features_combinadas, key_mode_data)
                
                # Mostrar todas las features en una tabla
                st.subheader("📋 Todas las Features")
                df_features = pd.DataFrame([features_combinadas])
                st.dataframe(df_features, use_container_width=True)
                
                # Predicción con features combinadas
                st.header("🎤 Predicción Final")
                
                try:
                    df_prediccion = crear_dataframe_prediccion(features_combinadas, columnas_modelo)
                    prediccion = modelo.predict(df_prediccion)[0]
                    
                    st.success(f"## 🎵 **{prediccion}**")
                    st.balloons()
                    
                    # Mostrar información detallada
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎼 Información Musical:**")
                        if key_mode_data:
                            st.write(f"• **Tonalidad:** {key_info['key_name']} {key_info['mode_name']}")
                            st.write(f"• **Escala:** {features_combinadas.get('scale', 'N/A')}")
                            st.write(f"• **Notas no naturales:** {features_combinadas.get('non_naturales_notes', 'N/A')}")
                            st.write(f"• **Confianza:** {key_info['confidence']:.3f}")
                    
                    with col2:
                        st.write("**📊 Features Principales:**")
                        principales = ['tempo', 'energy', 'danceability', 'valence']
                        for feature in principales:
                            if feature in features_combinadas:
                                val = features_combinadas[feature]
                                st.write(f"• **{feature.title()}:** {val:.3f}" if isinstance(val, float) else f"• **{feature.title()}:** {val}")
                    
                    # Mostrar datos técnicos completos
                    with st.expander("📋 Datos técnicos completos"):
                        st.dataframe(df_prediccion, use_container_width=True)
                    
                    # Mostrar respuesta de ReccoBeats
                    with st.expander("🌐 Respuesta de ReccoBeats"):
                        st.json(features_reccobeats)
                        
                except Exception as e:
                    st.error(f"❌ Error en predicción: {e}")
            
            else:
                st.error("❌ No se pudieron extraer las features principales con ReccoBeats")

if __name__ == "__main__":
    main()