import numpy as np
import tempfile
import os
import warnings

def extraer_features_essentia(archivo_bytes, nombre_archivo):
    """
    Extrae features de audio usando Essentia (tipo Spotify)
    """
    try:
        # Verificar si essentia está disponible
        import essentia.standard as es
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{nombre_archivo.split('.')[-1]}") as temp_file:
            temp_file.write(archivo_bytes)
            temp_path = temp_file.name
        
        try:
            # 1. Cargar el audio
            loader = es.MonoLoader(filename=temp_path)
            audio = loader()
            
            if len(audio) == 0:
                return None, "Audio vacío"
            
            # 2. Extraer métricas directas
            
            # Loudness
            try:
                loudness_extractor = es.LoudnessEBUR128()
                loudness_result = loudness_extractor(audio)
                loudness = float(np.mean(loudness_result[0]))
            except:
                loudness = -10.0  # Valor por defecto
            
            # Tempo
            try:
                rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
                tempo, beats, beats_confidence, _, _ = rhythm_extractor(audio)
                tempo = float(tempo)
            except:
                tempo = 120.0  # Valor por defecto
            
            # Key y modo
            try:
                key_extractor = es.KeyExtractor()
                key_str, scale_str, strength = key_extractor(audio)
                
                # Mapeo de notas a número
                key_map = {
                    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
                }
                key = key_map.get(key_str, 0)
                
                # Mode: 0 = major, 1 = minor
                mode = 0 if scale_str.lower() == "major" else 1
            except:
                key = 0
                mode = 0
            
            # 3. MusicNN para etiquetas perceptuales
            try:
                musicnn = es.MusicNN()
                tags = musicnn(audio)
                tags = {k.lower(): v for k, v in tags.items()}
            except:
                # Si MusicNN falla, usar valores por defecto
                tags = {}
            
            # 4. Mapear etiquetas a métricas tipo Spotify
            acousticness = float(tags.get("acoustic", 0.3))
            danceability = float(tags.get("dance", 0.5))
            valence = float(tags.get("happy", 0.5))
            instrumentalness = float(tags.get("instrumental", 0.1))
            liveness = float(tags.get("live", 0.1))
            speechiness = float(tags.get("speech", 0.05))
            
            # Heurística para energy
            energy_tags = ["electronic", "rock", "metal", "punk", "techno"]
            energy_values = [tags.get(t, 0.0) for t in energy_tags]
            energy = float(np.mean(energy_values)) if energy_values else 0.5
            
            # 5. Resultado final
            features = {
                "acousticness": acousticness,
                "danceability": danceability,
                "energy": energy,
                "instrumentalness": instrumentalness,
                "key": key,
                "mode": mode,
                "liveness": liveness,
                "loudness": loudness,
                "speechiness": speechiness,
                "tempo": tempo,
                "valence": valence
            }
            
            return features, "Extracción exitosa"
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except ImportError:
        return None, "Essentia no está instalado"
    except Exception as e:
        return None, f"Error en extracción: {str(e)}"

def verificar_essentia():
    """Verifica si Essentia está disponible"""
    try:
        import essentia.standard as es
        return True
    except ImportError:
        return False

def instalar_essentia():
    """Instrucciones para instalar Essentia"""
    return """
    Para instalar Essentia:
    
    **Windows:**
    ```
    pip install essentia-tensorflow
    ```
    
    **Linux/Mac:**
    ```
    pip install essentia-tensorflow
    ```
    
    **Alternativa con conda:**
    ```
    conda install -c mtg essentia-tensorflow
    ```
    """