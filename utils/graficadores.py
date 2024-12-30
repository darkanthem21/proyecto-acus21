import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import filtros as flt

def graficar_muestra(muestra):
    for emocion, ruta in zip(muestra.Emociones, muestra.Ruta):
        y, sr = librosa.load(ruta, duration=3)
        grafica(y, sr, emocion)
        
def total_emociones(d_path):  
    # Configurar estilo y paleta de colores
    sns.set_style("whitegrid")
    colors = sns.color_palette('pastel', n_colors=len(d_path['Emociones'].unique()))
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Crear gráfico con colores personalizados
    sns.countplot(
        data=d_path, 
        x='Emociones', 
        palette=colors,
        order=d_path['Emociones'].value_counts().index,
        edgecolor='black'
    )
    
    # Personalizar título y etiquetas
    plt.title(
        'Distribución de Emociones', 
        fontsize=24, 
        pad=20, 
        fontweight='bold',
        color='#333333'
    )
    plt.xlabel('Emociones', fontsize=18, labelpad=15)
    plt.ylabel('Cantidad', fontsize=18, labelpad=15)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)
    
    # Añadir etiquetas de valores
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', 
            va='bottom', 
            fontsize=14,
            color='#555555',
            xytext=(0, 10),
            textcoords='offset points'
        )
    
    # Estilizar ejes y fondo
    sns.despine(left=True, bottom=True)
    ax.grid(False)
    ax.set_facecolor('#f7f7f7')
    fig.patch.set_facecolor('#f7f7f7')
    
    plt.tight_layout()
    plt.show()

def grafica(data, sr, e):
    plt.figure(figsize=(15, 10))
    
    # Waveform subplot
    plt.subplot(2, 1, 1)
    plt.title('Grafico de onda normalizado para {} '.format(e), size=16)
    normalized_data = librosa.util.normalize(data)
    librosa.display.waveshow(normalized_data, sr=sr)
    
    plt.subplot(2, 1, 2)
    plt.title('Espectrograma para {} '.format(e), size=16)
    D = librosa.amplitude_to_db(abs(librosa.stft(data, 
                                                n_fft=1024,  
                                                hop_length=128,  
                                                win_length=1024)),  
                              ref=np.max)
    
    librosa.display.specshow(D, 
                            sr=sr, 
                            x_axis='time', 
                            y_axis='hz',
                            hop_length=128,
                            fmax=8000,  # Limit the maximum frequency to 8000 Hz
                            cmap='magma') 
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()

def mostrar_coef(audio_path):
    """Visualiza los coeficientes extraídos del audio"""
    # Cargar audio
    y, sr = librosa.load(audio_path, duration=3)
    
    plt.figure(figsize=(15, 12))
    
    # 1. Zero Crossing Rate
    plt.subplot(3, 2, 1)
    zcr = librosa.feature.zero_crossing_rate(y)
    plt.plot(zcr[0])
    plt.title('Zero Crossing Rate')
    plt.grid()
    
    # 2. Chroma STFT
    plt.subplot(3, 2, 2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', cmap = "Greens")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Chroma STFT')
    
    # 3. MFCC
    plt.subplot(3, 2, 3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title('MFCC')
    
    # 4. RMS Energy
    plt.subplot(3, 2, 4)
    rms = librosa.feature.rms(y=y)
    plt.plot(rms[0])
    plt.title('RMS Energy')
    plt.grid()
    
    # 5. Mel Spectrogram
    plt.subplot(3, 2, 5)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                           y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    # 6. Waveform normalizado
    plt.subplot(3, 2, 6)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform Normalizado')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    # Reproducir audio
    return Audio(y, rate=sr)

def compare_augmentations(audio_path):
    """Compara y visualiza el audio original con sus versiones aumentadas"""
    # Cargar audio original
    data, sr = librosa.load(audio_path)
    
    # Aplicar filtros
    noisy_data = flt.noise(data)
    stretched_shifted_data = flt.stretch(flt.shift(data))
    pitched_data = flt.pitch(data, sr)
    
    # Crear visualización
    fig, axs = plt.subplots(4, 2, figsize=(15, 16))
    # Original
    axs[0,0].set_title('Forma de onda original')
    librosa.display.waveshow(data, sr=sr, ax=axs[0,0])
    
    axs[0,1].set_title('Espectrograma original')
    D = librosa.amplitude_to_db(abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[0,1])
    
    # Ruido
    axs[1,0].set_title('Forma de onda con ruido')
    librosa.display.waveshow(noisy_data, sr=sr, ax=axs[1,0])
    
    axs[1,1].set_title('Espectrograma con ruido')
    D = librosa.amplitude_to_db(abs(librosa.stft(noisy_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[1,1])
    
    # Estirado y Desplazado
    axs[2,0].set_title('Forma de onda estirada y desplazada')
    librosa.display.waveshow(stretched_shifted_data, sr=sr, ax=axs[2,0])
    
    axs[2,1].set_title('Espectrograma estirado y desplazado')
    D = librosa.amplitude_to_db(abs(librosa.stft(stretched_shifted_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[2,1])
    
    # Tono modificado
    axs[3,0].set_title('Forma de onda con pitch modificado')
    librosa.display.waveshow(pitched_data, sr=sr, ax=axs[3,0])
    
    axs[3,1].set_title('Espectrograma con pitch modificado')
    D = librosa.amplitude_to_db(abs(librosa.stft(pitched_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[3,1])
    
    plt.tight_layout()
    plt.show()
    
    # Reproducir audios
    print("\nAudio Original:")
    display(Audio(data, rate=sr))
    
    print("\nAudio con Ruido:")
    display(Audio(noisy_data, rate=sr))
    
    print("\nAudio Estirado y Desplazado:")
    display(Audio(stretched_shifted_data, rate=sr))
    
    print("\nAudio con Pitch Modificado:")
    display(Audio(pitched_data, rate=sr))
