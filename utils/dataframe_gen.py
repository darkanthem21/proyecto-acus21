import os
import pandas as pd

def ravdess_df(ravdess_path):
    rv_lista = os.listdir(ravdess_path)
    f_emocion = []
    f_path = []
    
    for dir in rv_lista:
        actor = os.listdir(ravdess_path + dir)
        for archivo in actor:
            part = archivo.split('.')[0]
            part = part.split('-')
            f_emocion.append(int(part[2]))
            f_path.append(ravdess_path + dir + '/' + archivo)
    
    df_emociones = pd.DataFrame(f_emocion, columns=['Emociones'])
    df_path = pd.DataFrame(f_path, columns=['Ruta'])
    ravdess_df = pd.concat([df_emociones, df_path], axis=1)
    
    ravdess_df.Emociones.replace({ # no hay emociones mas generales?
        1:'neutral', 2:'calma', 3:'felicidad', 4:'tristeza', 
        5:'enojo', 6:'miedo', 7:'desagrado', 8:'sorpresa'
    }, inplace=True)

    # Imprime el dataframe
    # ravdess_df.head()
    
    return ravdess_df

def crema_df(crema_path):
    cr_lista = os.listdir(crema_path)
    f_emocion = []
    f_path = []
    
    for archivo in cr_lista:
        f_path.append(crema_path + archivo)
        part = archivo.split('_')
        if part[2] == 'SAD': # Rustico pero funca
            f_emocion.append('tristeza')
        elif part[2] == 'ANG':
            f_emocion.append('enojo')
        elif part[2] == 'DIS':
            f_emocion.append('desagrado')
        elif part[2] == 'FEA':
            f_emocion.append('miedo')
        elif part[2] == 'HAP':
            f_emocion.append('felicidad')
        elif part[2] == 'NEU':
            f_emocion.append('neutral')
        else:
            f_emocion.append('queseso?')
    
    df_emociones = pd.DataFrame(f_emocion, columns=['Emociones'])
    df_path = pd.DataFrame(f_path, columns=['Ruta'])
    crema_df = pd.concat([df_emociones, df_path], axis=1)
    # crema_df.head()
    return crema_df

def tess_df(tess_path):
    ts_lista = os.listdir(tess_path)
    f_emocion = []
    f_path = []
    
    emotion_map = {
        'ps': 'sorpresa',
        'angry': 'enojo',
        'disgust': 'desagrado',
        'fear': 'miedo',
        'happy': 'felicidad',
        'neutral': 'neutral',
        'sad': 'tristeza'
    }
    
    for archivo in ts_lista:
        directorios = os.listdir(tess_path + archivo)
        for dir in directorios:
            part = dir.split('.')[0]
            emotion = part.split('_')[2]
            
            if emotion in emotion_map:
                f_emocion.append(emotion_map[emotion])
                f_path.append(tess_path + archivo + '/' + dir)

    df_emociones = pd.DataFrame(f_emocion, columns=['Emociones'])
    df_path = pd.DataFrame(f_path, columns=['Ruta'])
    tess_df = pd.concat([df_emociones, df_path], axis=1)
    
    # tess_df.head()
    return tess_df

# Crea una funcion que eliga un audio por emocion del dataframe completo
def df_muestra(df):
    return df.groupby('Emociones').sample(1)
