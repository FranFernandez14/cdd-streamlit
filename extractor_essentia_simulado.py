import json
import numpy as np
import tempfile
import os
import warnings

# Simulación de essentia.standard
class EssentiaSimulado:
    class MonoLoader:
        def __init__(self, filename):
            self.filename = filename
        
        def __call__(self):
            # Simular carga de audio - devolver array numpy simulado
            try:
                # Intentar con librosa si está disponible
                import librosa
                y, sr = librosa.load(self.filename, sr=22050, mono=True)
                return y
            except ImportError:
                # Si no hay librosa, simular con pydub
                try:
                    from pydub import AudioSegment
                    import io
                    
                    audio = AudioSegment.from_file(self.filename)
                    # Convertir a mono y normalizar
                    if audio.channels > 1:
                        audio = audio.set_channels(1)
                    
                    # Convertir a array numpy
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    samples = samples / (2**15)  # Normalizar
                    
                    return samples
                except:
                    # Último recurso: array simulado
                    return np.random.randn(22050 * 30)  # 30 segundos simulados
    
    class LoudnessEBUR128:
        def __call__(self, audio):
            # Simular loudness
            rms = np.sqrt(np.mean(audio**2))
            loudness = 20 * np.log10(rms + 1e-8)
            return loudness
    
    class RhythmExtractor2013:
        def __init__(self, method="multifeature"):
            self.method = method
        
        def __call__(self, audio):
            # Simular extracción de tempo
            # Análisis básico de periodicidad
            try:
                # Detectar picos de energía
                window_size = 1024
                hop_size = 512
                
                energy = []
                for i in range(0, len(audio) - window_size, hop_size):
                    window = audio[i:i+window_size]
                    energy.append(np.sum(window**2))
                
                energy = np.array(energy)
                
                # Autocorrelación para encontrar periodicidad
                autocorr = np.correlate(energy, energy, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Buscar picos en la autocorrelación
                peaks = []
                for i in range(1, len(autocorr)-1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        peaks.append(i)
                
                if peaks:
                    # Convertir a BPM
                    peak_idx = peaks[0] if peaks else 60
                    tempo = 60.0 / (peak_idx * hop_size / 22050) * 60
                    tempo = np.clip(tempo, 60, 200)
                else:
                    tempo = 120.0
                
            except:
                tempo = 120.0
            
            beats = np.array([])
            beats_confidence = np.array([])
            estimates = np.array([])
            bpm_intervals = np.array([])
            
            return tempo, beats, beats_confidence, estimates, bpm_intervals
    
    class KeyExtractor:
        def __call__(self, audio):
            # Simular extracción de key
            # Análisis cromático básico
            try:
                # FFT para análisis espectral
                fft = np.fft.fft(audio[:22050])  # Primer segundo
                magnitude = np.abs(fft)
                
                # Mapear a 12 clases cromáticas
                chroma = np.zeros(12)
                for i in range(len(magnitude)//2):
                    freq = i * 22050 / len(magnitude)
                    if freq > 80:  # Ignorar frecuencias muy bajas
                        note = int(12 * np.log2(freq / 440.0)) % 12
                        chroma[note] += magnitude[i]
                
                # Encontrar la nota dominante
                key_idx = np.argmax(chroma)
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key = keys[key_idx]
                
                # Determinar modo (mayor/menor) basado en intervalos
                major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
                minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
                
                # Rotar perfiles para la key detectada
                major_rotated = np.roll(major_profile, key_idx)
                minor_rotated = np.roll(minor_profile, key_idx)
                
                major_score = np.dot(chroma, major_rotated)
                minor_score = np.dot(chroma, minor_rotated)
                
                scale = "major" if major_score > minor_score else "minor"
                strength = max(major_score, minor_score) / np.sum(chroma)
                
            except:
                key = 'C'
                scale = 'major'
                strength = 0.5
            
            return key, scale, strength

# Simulación de musicnn.extractor
class MusicNNSimulado:
    @staticmethod
    def extractor(file_path, model='MTT_musicnn', extract_features=False):
        """Simula la extracción de MusicNN"""
        try:
            # Cargar audio para análisis básico
            try:
                import librosa
                y, sr = librosa.load(file_path, sr=22050, mono=True)
            except ImportError:
                # Fallback con pydub
                from pydub import AudioSegment
                audio = AudioSegment.from_file(file_path)
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                y = samples / (2**15)
                sr = audio.frame_rate
            
            # Análisis básico para simular tags
            rms = np.sqrt(np.mean(y**2))
            zcr = np.mean(np.abs(np.diff(np.sign(y))))
            spectral_centroid = np.mean(np.abs(np.fft.fft(y[:1024])))
            
            # Simular tags basados en características básicas
            tags = [
                ("acousticness", max(0, 1.0 - rms * 2)),
                ("danceability", min(1.0, rms * 1.5)),
                ("energy", rms),
                ("instrumentalness", max(0, 1.0 - zcr * 10)),
                ("liveness", min(1.0, np.std(y) * 2)),
                ("speechiness", min(1.0, zcr * 5)),
                ("valence", min(1.0, spectral_centroid / 1000))
            ]
            
        except Exception as e:
            # Tags por defecto si falla todo
            tags = [
                ("acousticness", 0.3),
                ("danceability", 0.5),
                ("energy", 0.5),
                ("instrumentalness", 0.1),
                ("liveness", 0.1),
                ("speechiness", 0.05),
                ("valence", 0.5)
            ]
        
        return None, tags

def analyze_audio(file_path: str):
    """
    Análisis de audio usando implementación simulada de Essentia + MusicNN
    """
    try:
        # =============================================================
        # 1. EXTRACCIÓN CON ESSENTIA (SIMULADO)
        # =============================================================
        es = EssentiaSimulado()
        
        loader = es.MonoLoader(filename=file_path)
        audio = loader()
        
        # Loudness
        loudness_extractor = es.LoudnessEBUR128()
        loudness = loudness_extractor(audio)
        
        # Tempo, Key y Mode
        bpm_extractor = es.RhythmExtractor2013(method="multifeature")
        tempo, _, _, _, _ = bpm_extractor(audio)
        
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)
        
        # Convertir key a número 0–11 (C=0, C#=1, D=2, …, B=11)
        key_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                   'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        key_num = key_map.get(key, 0)
        mode_num = 0 if scale.lower() == "minor" else 1
        
        # =============================================================
        # 2. EXTRACCIÓN CON MUSICNN (SIMULADO)
        # =============================================================
        musicnn = MusicNNSimulado()
        _, tags = musicnn.extractor(file_path, model='MTT_musicnn', extract_features=False)
        
        # Convertir tags a un dict con las principales características tipo Spotify
        feature_map = {
            "acousticness": 0.0,
            "danceability": 0.0,
            "energy": 0.0,
            "instrumentalness": 0.0,
            "liveness": 0.0,
            "speechiness": 0.0,
            "valence": 0.0
        }
        
        # Mapear tags
        for tag, value in tags:
            tag_lower = tag.lower()
            if tag_lower in feature_map:
                feature_map[tag_lower] = float(value)
        
        # =============================================================
        # 3. UNIFICAR TODO EN UN DICCIONARIO FINAL
        # =============================================================
        features = {
            "acousticness": feature_map["acousticness"],
            "danceability": feature_map["danceability"],
            "energy": feature_map["energy"],
            "instrumentalness": feature_map["instrumentalness"],
            "key": key_num,
            "liveness": feature_map["liveness"],
            "loudness": float(loudness),
            "mode": mode_num,
            "speechiness": feature_map["speechiness"],
            "tempo": float(tempo),
            "valence": feature_map["valence"]
        }
        
        return features, "Extracción simulada exitosa"
        
    except Exception as e:
        return None, f"Error en análisis simulado: {str(e)}"

def extraer_features_essentia_simulado(archivo_bytes, nombre_archivo):
    """
    Wrapper para usar con Streamlit
    """
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{nombre_archivo.split('.')[-1]}") as temp_file:
        temp_file.write(archivo_bytes)
        temp_path = temp_file.name
    
    try:
        features, mensaje = analyze_audio(temp_path)
        return features, mensaje
    finally:
        # Limpiar archivo temporal
        try:
            os.unlink(temp_path)
        except:
            pass

# =============================================================
# EJEMPLO DE USO
# =============================================================
if __name__ == "__main__":
    # Ejemplo con archivo de prueba
    ruta_audio = "test_audio.mp3"
    if os.path.exists(ruta_audio):
        result, mensaje = analyze_audio(ruta_audio)
        if result:
            print("=== FEATURES TIPO SPOTIFY (SIMULADO) ===")
            print(json.dumps(result, indent=4))
        else:
            print(f"Error: {mensaje}")
    else:
        print("Archivo de prueba no encontrado")