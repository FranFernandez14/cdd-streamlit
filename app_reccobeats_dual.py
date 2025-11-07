import streamlit as st
import pandas as pd
import joblib
import requests
import warnings
import os
import struct

# Configuración
st.set_page_config(page_title="🎵 Predictor ReccoBeats Dual", page_icon="🎵", layout="wide")
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
    st.title("🎵 Predictor Musical - Solo ReccoBeats")
    st.markdown("**Análisis con API ReccoBeats + Predicción con Ambos Modelos**")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Configuración de Modelos")
        
        # Buscar todos los modelos
        modelos = [f for f in os.listdir('.') if f.endswith('.joblib')]
        if not modelos:
            st.error("❌ No hay archivos .joblib")
            return
        
        # Selector de modelos
        modelos_seleccionados = st.multiselect(
            "📁 Seleccionar modelos para comparar:",
            modelos,
            default=modelos[:2] if len(modelos) >= 2 else modelos,
            help="Puedes seleccionar múltiples modelos para comparar predicciones"
        )
        
        if not modelos_seleccionados:
            st.warning("⚠️ Selecciona al menos un modelo")
            return
        
        if st.button("🔄 Recargar Modelos"):
            st.cache_resource.clear()
            st.rerun()
        
        # Configuración de corte
        st.header("✂️ Configuración")
        tamaño_objetivo = st.slider("Tamaño objetivo (MB)", 2.0, 4.9, 4.8, 0.1)
        
        # Estado de la API
        st.header("🌐 Estado API")
        if st.button("🧪 Probar ReccoBeats"):
            try:
                response = requests.get("https://api.reccobeats.com", timeout=5)
                if response.status_code < 500:
                    st.success("✅ API accesible")
                else:
                    st.warning("⚠️ API con problemas")
            except:
                st.error("❌ API no accesible")
    
    # Cargar modelos seleccionados
    modelos_cargados = {}
    columnas_por_modelo = {}
    
    for archivo_modelo in modelos_seleccionados:
        modelo, error = cargar_modelo_dinamico(archivo_modelo)
        if modelo is not None:
            modelos_cargados[archivo_modelo] = modelo
            columnas_por_modelo[archivo_modelo] = obtener_columnas_modelo(modelo)
            st.success(f"✅ {archivo_modelo} cargado")
        else:
            st.error(f"❌ Error en {archivo_modelo}: {error}")
    
    if not modelos_cargados:
        st.error("❌ No se pudo cargar ningún modelo")
        return
    
    # Mostrar información de los modelos
    with st.sidebar:
        st.header("📊 Info de Modelos")
        for archivo, columnas in columnas_por_modelo.items():
            with st.expander(f"📋 {archivo}"):
                st.write(f"Columnas: {len(columnas)}")
                for col in columnas:
                    st.write(f"• {col}")
    
    # Información sobre ReccoBeats
    st.info("""
    🌐 **ReccoBeats API**
    • Análisis profesional de audio en la nube
    • Extrae 9 características principales
    • Compatible con WAV y MP3
    • Rápido y confiable
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
        
        # Análisis
        st.header("🔍 Análisis con ReccoBeats")
        
        if st.button("🚀 Analizar y Predecir con Todos los Modelos", type="primary", use_container_width=True):
            
            # Análisis con ReccoBeats
            with st.spinner("🌐 Analizando con ReccoBeats API..."):
                features_api, mensaje_api = analizar_con_reccobeats(archivo_final)
            
            if features_api:
                st.success("✅ Análisis completado")
                
                # Mostrar características
                st.subheader("📊 Características Extraídas")
                
                # Mostrar en grid de 3 columnas
                caracteristicas_disponibles = list(features_api.keys())
                for i in range(0, len(caracteristicas_disponibles), 3):
                    cols = st.columns(3)
                    for j, feature_name in enumerate(caracteristicas_disponibles[i:i+3]):
                        with cols[j]:
                            val = features_api[feature_name]
                            if isinstance(val, float):
                                st.metric(
                                    feature_name.replace('_', ' ').title(),
                                    f"{val:.3f}"
                                )
                            else:
                                st.metric(
                                    feature_name.replace('_', ' ').title(),
                                    str(val)
                                )
                
                # Predicciones con todos los modelos
                st.header("🎤 Predicciones por Modelo")
                
                predicciones = {}
                
                for archivo_modelo, modelo in modelos_cargados.items():
                    st.subheader(f"🤖 {archivo_modelo}")
                    
                    try:
                        columnas_modelo = columnas_por_modelo[archivo_modelo]
                        df_prediccion = crear_dataframe_prediccion(features_api, columnas_modelo)
                        prediccion = modelo.predict(df_prediccion)[0]
                        
                        predicciones[archivo_modelo] = prediccion
                        
                        # Mostrar resultado
                        st.success(f"🎵 **{prediccion}**")
                        
                        # Mostrar qué características se usaron
                        cols_usadas = [col for col in columnas_modelo if col in features_api]
                        cols_faltantes = [col for col in columnas_modelo if col not in features_api]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if cols_usadas:
                                st.write("✅ **Características usadas:**")
                                st.write(", ".join(cols_usadas))
                        with col2:
                            if cols_faltantes:
                                st.write("⚠️ **Características faltantes (valor 0):**")
                                st.write(", ".join(cols_faltantes))
                        
                        # Mostrar datos técnicos
                        with st.expander(f"📋 Datos técnicos - {archivo_modelo}"):
                            st.dataframe(df_prediccion, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error con {archivo_modelo}: {e}")
                
                # Comparar predicciones si hay múltiples modelos
                if len(predicciones) > 1:
                    st.header("🔍 Comparación de Predicciones")
                    
                    # Crear tabla de comparación
                    df_comparacion = pd.DataFrame([
                        {"Modelo": modelo, "Predicción": pred} 
                        for modelo, pred in predicciones.items()
                    ])
                    
                    st.dataframe(df_comparacion, use_container_width=True)
                    
                    # Verificar si todas las predicciones son iguales
                    predicciones_unicas = set(predicciones.values())
                    
                    if len(predicciones_unicas) == 1:
                        st.success("🎉 ¡Todos los modelos predicen el mismo artista!")
                        st.balloons()
                    else:
                        st.warning("⚠️ Los modelos predicen artistas diferentes")
                        
                        # Mostrar estadísticas
                        from collections import Counter
                        contador = Counter(predicciones.values())
                        
                        st.write("📊 **Frecuencia de predicciones:**")
                        for artista, count in contador.most_common():
                            st.write(f"• **{artista}**: {count} modelo(s)")
                
                # Mostrar respuesta completa de la API
                with st.expander("🔍 Respuesta completa de ReccoBeats"):
                    st.json(features_api)
            
            else:
                st.error(f"❌ {mensaje_api}")
                st.info("💡 Verifica tu conexión a internet y el formato del archivo")

if __name__ == "__main__":
    main()