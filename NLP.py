import pandas as pd
import nltk


class NLP:
    """
    Clase que contiene algunas de las funciones mas usadas dentro de los procesos de NLP
    """

    def __init__(self):
        pass

    def dfColumnToken(self, df, columnaTexto, separador=' '):
        """
        Funcion que permite convertir una columna de texto de un Pandas DataFrame a tokens
        :param df: Pandas DataFrame con la columna de texto a tokenizar
        :param columnaTexto: Nombre de la columna de texto a tokenizar
        :param separador: Separados de las palabras dentro del texto, default ' '
        :return: Retorna un arreglo con las palabras que hay en el texto
        """
        # Columna de texto en minuscula
        text = df[columnaTexto].to_string(index=False).lower()

        # Se extrae cada una de las palabras del texto delimitadas por el separador
        tokens = [t for t in text.split(separador)]

        return tokens

    def remplzarStopWords(self, df, columnaTexto, stopWords):
        """
        Funcion que permite remplazar las stopwords que hay en un texto
        :param df: Pandas DataFrame con la columna de texto de donde se quiere eliminar las stopwords
        :param columnaTexto: Nombre de la columna de texto
        :param stopWords: Arreglo con las palabras que se buscaran para eliminar
        :return Retorna la columna del DataFrame con las palabras ya eliminadas
        """
        for texto in df[columnaTexto]:
            for palabra in stopWords:
                if palabra in texto:
                    texto = texto.replace(palabra, '')

        return df[columnaTexto]

    def countTokens(self, tokens):
        """
        Funcion que permite retornar la frecuencia de cada uno de los tokens
        :param tokens: Arreglo con las palabras
        :return: Frecuencia de cada una de las palabras
        """
        # Contamos frecuencia de palabras
        freq = nltk.FreqDist(tokens)

        # Se convierte en DF el arreglo de palabras
        df_fdist = pd.DataFrame.from_dict(freq, orient='index').reset_index()

        return df_fdist
