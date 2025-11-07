import streamlit as st
import pandas as pd
import joblib
import requests
import warnings
import os
import io
import struct

# Importar nuestros módulos
try:
    from extractor_essentia import extraer_features_essentia, verificar_essentia, instalar_essentia
    ESSENTIA_REAL_DISPONIBLE = True
except ImportError:
    ESSENTIA_REAL_DISPONIBLE = False

from extractor_essentia_simulado import extraer_features_essentia_simulado

# Configuración
st.set_page_config(page_title="🎵 Predictor Dual", page_icon="🎵", layout="wide")
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

def comparar_features(features_api, features_essentia, columnas_disponibles):
    """Compara features de ambos métodos"""
    comparacion = []
    
    for col in columnas_disponibles:
        api_val = features_api.get(col, 0) if features_api else 0
        essentia_val = features_essentia.get(col, 0) if features_essentia else 0
        
        diferencia = abs(api_val - essentia_val) if isinstance(api_val, (int, float)) and isinstance(essentia_val, (int, float)) else 0
        
        comparacion.append({
            'Feature': col,
            'ReccoBeats API': api_val,
            'Essentia Local': essentia_val,
            'Diferencia': diferencia
        })
    
    return pd.DataFrame(comparacion)

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
    st.title("🎵 Predictor Musical Dual")
    st.markdown("**Comparación: API ReccoBeats vs Essentia Local**")
    
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
        
        # Configuración de corte
        st.header("✂️ Configuración")
        tamaño_objetivo = st.slider("Tamaño objetivo (MB)", 2.0, 4.9, 4.8, 0.1)
        
        # Estado de Essentia
        st.header("🔧 Estado del Sistema")
        if ESSENTIA_REAL_DISPONIBLE:
            essentia_disponible = verificar_essentia()
            if essentia_disponible:
                st.success("✅ Essentia Real disponible")
            else:
                st.warning("⚠️ Essentia Real no funciona")
        else:
            essentia_disponible = False
            st.info("🔄 Usando Essentia Simulado")
        
        st.success("✅ Extractor simulado siempre disponible")
    
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
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        🌐 **ReccoBeats API**
        • Análisis en la nube
        • Rápido y confiable
        • Requiere internet
        """)
    with col2:
        if essentia_disponible:
            st.success("""
            🏠 **Essentia Local**
            • Análisis local
            • Sin límites de uso
            • Funciona offline
            """)
        else:
            st.warning("""
            🏠 **Essentia Local**
            • No disponible
            • Requiere instalación
            • pip install essentia-tensorflow
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
        
        # Análisis dual
        st.header("🔍 Análisis Dual")
        
        if st.button("🚀 Analizar con Ambos Métodos", type="primary", use_container_width=True):
            
            # Crear columnas para mostrar progreso
            col1, col2 = st.columns(2)
            
            # Análisis con ReccoBeats API
            with col1:
                st.subheader("🌐 ReccoBeats API")
                with st.spinner("Analizando con API..."):
                    features_api, mensaje_api = analizar_con_reccobeats(archivo_final)
                
                if features_api:
                    st.success("✅ API exitosa")
                else:
                    st.error(f"❌ {mensaje_api}")
            
            # Análisis con Essentia
            with col2:
                st.subheader("🏠 Essentia Simulado")
                with st.spinner("Analizando con Essentia simulado..."):
                    features_essentia, mensaje_essentia = extraer_features_essentia_simulado(
                        archivo_final.getvalue(), archivo_final.name
                    )
                
                if features_essentia:
                    st.success("✅ Essentia simulado exitoso")
                else:
                    st.error(f"❌ {mensaje_essentia}")
            
            # Mostrar resultados si al menos uno funcionó
            if features_api or features_essentia:
                
                # Comparación de features
                st.header("📊 Comparación de Features")
                
                if features_api and features_essentia:
                    # Ambos métodos funcionaron
                    df_comparacion = comparar_features(features_api, features_essentia, columnas_modelo)
                    st.dataframe(df_comparacion, use_container_width=True)
                    
                    # Mostrar diferencias significativas
                    diferencias_grandes = df_comparacion[df_comparacion['Diferencia'] > 0.2]
                    if not diferencias_grandes.empty:
                        st.warning("⚠️ Diferencias significativas encontradas:")
                        st.dataframe(diferencias_grandes)
                
                # Predicciones
                st.header("🎤 Predicciones")
                
                col1, col2 = st.columns(2)
                
                # Predicción con API
                if features_api:
                    with col1:
                        st.subheader("🌐 Con ReccoBeats API")
                        try:
                            df_api = crear_dataframe_prediccion(features_api, columnas_modelo)
                            pred_api = modelo.predict(df_api)[0]
                            st.success(f"🎵 **{pred_api}**")
                            
                            with st.expander("Ver datos API"):
                                st.dataframe(df_api)
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                # Predicción con Essentia
                if features_essentia:
                    with col2:
                        st.subheader("🏠 Con Essentia Local")
                        try:
                            df_essentia = crear_dataframe_prediccion(features_essentia, columnas_modelo)
                            pred_essentia = modelo.predict(df_essentia)[0]
                            st.success(f"🎵 **{pred_essentia}**")
                            
                            with st.expander("Ver datos Essentia"):
                                st.dataframe(df_essentia)
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                # Comparar predicciones
                if features_api and features_essentia:
                    try:
                        pred_api = modelo.predict(crear_dataframe_prediccion(features_api, columnas_modelo))[0]
                        pred_essentia = modelo.predict(crear_dataframe_prediccion(features_essentia, columnas_modelo))[0]
                        
                        if pred_api == pred_essentia:
                            st.success("🎉 ¡Ambos métodos predicen el mismo artista!")
                            st.balloons()
                        else:
                            st.warning(f"⚠️ Predicciones diferentes: API='{pred_api}' vs Essentia='{pred_essentia}'")
                    except:
                        pass

if __name__ == "__main__":
    main()