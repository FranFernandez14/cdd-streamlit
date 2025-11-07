import streamlit as st
import pandas as pd
import joblib
import requests
import warnings
import os
import io
import struct
import wave

# Configuración
st.set_page_config(page_title="🎵 Predictor - Corte Nativo", page_icon="🎵", layout="wide")
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

def obtener_info_wav(archivo_bytes):
    """Obtiene información de un archivo WAV"""
    try:
        # Leer header WAV
        if len(archivo_bytes) < 44:
            return None
        
        # Verificar que sea WAV
        if archivo_bytes[:4] != b'RIFF' or archivo_bytes[8:12] != b'WAVE':
            return None
        
        # Extraer información del header
        channels = struct.unpack('<H', archivo_bytes[22:24])[0]
        sample_rate = struct.unpack('<I', archivo_bytes[24:28])[0]
        byte_rate = struct.unpack('<I', archivo_bytes[28:32])[0]
        bits_per_sample = struct.unpack('<H', archivo_bytes[34:36])[0]
        
        # Encontrar el chunk de datos
        pos = 36
        while pos < len(archivo_bytes) - 8:
            chunk_id = archivo_bytes[pos:pos+4]
            chunk_size = struct.unpack('<I', archivo_bytes[pos+4:pos+8])[0]
            
            if chunk_id == b'data':
                data_start = pos + 8
                data_size = chunk_size
                break
            
            pos += 8 + chunk_size
        else:
            return None
        
        # Calcular duración
        bytes_per_sample = bits_per_sample // 8
        total_samples = data_size // (channels * bytes_per_sample)
        duracion = total_samples / sample_rate
        
        return {
            'channels': channels,
            'sample_rate': sample_rate,
            'bits_per_sample': bits_per_sample,
            'duracion': duracion,
            'data_start': data_start,
            'data_size': data_size
        }
    except:
        return None

def cortar_wav_nativo(archivo_bytes, proporcion_objetivo):
    """Corta un archivo WAV manteniendo la estructura correcta"""
    try:
        info = obtener_info_wav(archivo_bytes)
        if not info:
            return None, ["No es un archivo WAV válido"]
        
        # Calcular nuevo tamaño de datos
        nuevo_data_size = int(info['data_size'] * proporcion_objetivo)
        
        # Asegurar que sea múltiplo del frame size
        frame_size = info['channels'] * (info['bits_per_sample'] // 8)
        nuevo_data_size = (nuevo_data_size // frame_size) * frame_size
        
        # Crear nuevo archivo WAV
        nuevo_archivo = bytearray(archivo_bytes[:info['data_start']])
        
        # Agregar datos cortados
        datos_cortados = archivo_bytes[info['data_start']:info['data_start'] + nuevo_data_size]
        nuevo_archivo.extend(datos_cortados)
        
        # Actualizar headers
        # Tamaño total del archivo
        nuevo_tamaño_total = len(nuevo_archivo) - 8
        nuevo_archivo[4:8] = struct.pack('<I', nuevo_tamaño_total)
        
        # Tamaño del chunk de datos
        pos_data_size = info['data_start'] - 4
        nuevo_archivo[pos_data_size:pos_data_size+4] = struct.pack('<I', nuevo_data_size)
        
        nueva_duracion = (nuevo_data_size // frame_size) / info['sample_rate']
        
        cambios = [
            f"Duración: {info['duracion']:.1f}s → {nueva_duracion:.1f}s",
            f"Calidad preservada: {info['channels']} canales, {info['sample_rate']}Hz, {info['bits_per_sample']} bits",
            "Formato: WAV nativo"
        ]
        
        return bytes(nuevo_archivo), cambios
        
    except Exception as e:
        return None, [f"Error procesando WAV: {e}"]

def cortar_por_bytes_inteligente(archivo_bytes, nombre_archivo, tamaño_objetivo_mb):
    """Corte inteligente por bytes manteniendo estructura"""
    
    tamaño_objetivo_bytes = tamaño_objetivo_mb * 1024 * 1024
    tamaño_original = len(archivo_bytes)
    
    if tamaño_original <= tamaño_objetivo_bytes:
        return archivo_bytes, ["No necesita corte"], tamaño_original
    
    proporcion_objetivo = tamaño_objetivo_bytes / tamaño_original * 0.95  # 95% para margen
    
    # Si es WAV, usar método nativo
    if nombre_archivo.lower().endswith('.wav'):
        resultado = cortar_wav_nativo(archivo_bytes, proporcion_objetivo)
        if resultado[0] is not None:
            return resultado[0], resultado[1], len(resultado[0])
    
    # Para MP3 o si WAV falla, corte simple pero inteligente
    try:
        # Para MP3, intentar cortar en boundaries de frames
        if nombre_archivo.lower().endswith('.mp3'):
            # Buscar headers de frames MP3 (0xFF 0xFB o similares)
            nuevo_tamaño = int(tamaño_original * proporcion_objetivo)
            
            # Buscar hacia atrás desde el punto de corte para encontrar un frame boundary
            for i in range(nuevo_tamaño, max(0, nuevo_tamaño - 1000), -1):
                if i < len(archivo_bytes) - 1:
                    if archivo_bytes[i] == 0xFF and (archivo_bytes[i+1] & 0xE0) == 0xE0:
                        archivo_cortado = archivo_bytes[:i]
                        cambios = [
                            f"Cortado en frame boundary MP3",
                            f"Tamaño: {tamaño_original/1024/1024:.1f}MB → {len(archivo_cortado)/1024/1024:.1f}MB"
                        ]
                        return archivo_cortado, cambios, len(archivo_cortado)
        
        # Corte simple si no se encuentra boundary
        nuevo_tamaño = int(tamaño_original * proporcion_objetivo)
        archivo_cortado = archivo_bytes[:nuevo_tamaño]
        
        cambios = [
            f"Corte proporcional: {proporcion_objetivo*100:.1f}% del archivo",
            f"Tamaño: {tamaño_original/1024/1024:.1f}MB → {len(archivo_cortado)/1024/1024:.1f}MB"
        ]
        
        return archivo_cortado, cambios, len(archivo_cortado)
        
    except Exception as e:
        return None, [f"Error en corte: {e}"], 0

def analizar_audio(archivo_audio):
    try:
        st.info("🔄 Analizando con ReccoBeats API...")
        files = {'audioFile': (archivo_audio.name, archivo_audio.getvalue(), archivo_audio.type)}
        response = requests.post('https://api.reccobeats.com/v1/analysis/audio-features', files=files, timeout=30)
        
        if response.status_code == 200:
            st.success("✅ Análisis completado")
            return response.json()
        else:
            st.error(f"❌ Error en API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def crear_dataframe_prediccion(caracteristicas_audio, columnas_modelo):
    datos = {}
    for col in columnas_modelo:
        datos[col] = caracteristicas_audio.get(col, 0)
    return pd.DataFrame([datos])

class ArchivoCortadoNativo:
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
    st.title("🎵 Predictor Musical - Corte Nativo")
    st.markdown("**Corte inteligente sin dependencias externas** - Preserva calidad")
    
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
        tamaño_objetivo = st.slider(
            "Tamaño objetivo (MB)", 
            min_value=2.0, 
            max_value=4.9, 
            value=4.8, 
            step=0.1
        )
    
    # Cargar modelo
    modelo, error = cargar_modelo_dinamico(archivo_modelo)
    
    if modelo is None:
        st.error(f"❌ Error: {error}")
        return
    
    st.success(f"✅ Modelo: {archivo_modelo}")
    
    # Detectar columnas
    columnas_modelo = obtener_columnas_modelo(modelo)
    
    with st.sidebar:
        st.header("📊 Info")
        st.write(f"Columnas: {len(columnas_modelo)}")
        st.success("✅ Método nativo")
        st.info("No requiere ffmpeg")
    
    # Información del método
    st.info("""
    🎯 **Método de corte nativo:**
    • WAV: Corte estructural preservando headers
    • MP3: Corte en frame boundaries cuando es posible
    • Sin dependencias externas (no ffmpeg)
    • Mantiene formato y calidad original
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
        with col3: 
            formato = archivo.name.split('.')[-1].upper()
            st.metric("🎵 Formato", formato)
        
        archivo_final = archivo
        
        # Mostrar info adicional para WAV
        if archivo.name.lower().endswith('.wav'):
            info_wav = obtener_info_wav(archivo.getvalue())
            if info_wav:
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("⏱️ Duración", f"{info_wav['duracion']:.1f}s")
                with col2: st.metric("🔊 Canales", info_wav['channels'])
                with col3: st.metric("📊 Sample Rate", f"{info_wav['sample_rate']}Hz")
        
        # Cortar si es necesario
        if tamaño_original > tamaño_limite:
            st.warning(f"⚠️ Archivo muy grande. Aplicando corte nativo...")
            
            with st.spinner("✂️ Cortando audio (método nativo)..."):
                audio_cortado, cambios, tamaño_final = cortar_por_bytes_inteligente(
                    archivo.getvalue(), archivo.name, tamaño_objetivo
                )
                
                if audio_cortado:
                    archivo_final = ArchivoCortadoNativo(audio_cortado, archivo.name)
                    reduccion = ((tamaño_original - tamaño_final) / tamaño_original) * 100
                    
                    st.success("✅ Corte completado - Estructura preservada")
                    
                    # Estadísticas
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("📉 Reducción", f"{reduccion:.1f}%")
                    with col2: st.metric("📏 Nuevo tamaño", f"{tamaño_final/1024/1024:.1f} MB")
                    with col3: st.metric("🔧 Método", "Nativo")
                    
                    # Cambios aplicados
                    if cambios:
                        with st.expander("✂️ Detalles del corte"):
                            for cambio in cambios:
                                st.write(f"• {cambio}")
                else:
                    st.error("❌ No se pudo cortar el archivo")
                    return
        
        # Análisis
        st.header("🎯 Análisis y Predicción")
        
        if st.button("🔍 Analizar Audio", type="primary", use_container_width=True):
            caracteristicas = analizar_audio(archivo_final)
            
            if caracteristicas:
                st.subheader("📊 Características Musicales")
                
                cols_disp = [c for c in columnas_modelo if c in caracteristicas]
                
                if cols_disp:
                    for i in range(0, len(cols_disp), 3):
                        cols = st.columns(3)
                        for j, col_name in enumerate(cols_disp[i:i+3]):
                            with cols[j]:
                                val = caracteristicas[col_name]
                                st.metric(
                                    col_name.replace('_', ' ').title(),
                                    f"{val:.3f}" if isinstance(val, float) else str(val)
                                )
                
                # Predicción
                st.subheader("🎤 Predicción del Artista")
                try:
                    df = crear_dataframe_prediccion(caracteristicas, columnas_modelo)
                    pred = modelo.predict(df)[0]
                    
                    st.success(f"## 🎵 **{pred}**")
                    st.balloons()
                    
                    with st.expander("📋 Datos técnicos"):
                        st.dataframe(df)
                        
                except Exception as e:
                    st.error(f"❌ Error en predicción: {e}")

if __name__ == "__main__":
    main()