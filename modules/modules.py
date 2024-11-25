import numpy as np


def entrena_perceptron(X, y, z, eta, t, funcion_activacion):
    """ Entrena un perceptron simple
    Parámetros
    ----------
        X -- valores de x_i para cada uno de los datos de entrenamiento
        y -- valor de salida deseada para cada uno de los datos de entrenamiento
        z -- valor del umbral para la función de activación
        eta -- coeficiente de aprendizaje
        t -- numero de epochs o iteraciones que se quieren realizar con los datos de entrenamiento
        funcion_activacion -- función de activación para el perceptrón.
    
    Devolución
    --------
        w -- valores de los pesos del perceptron
        J -- error cuadrático obtenido de comparar la salida deseada con la que se obtiene 
            con los pesos de cada iteración
    """  
   
    w = np.zeros(len(X[0]))  # Pesos a ceros
    n = 0  # Num de iteraciones a 0                     
    
    yhat_vec = np.zeros(len(y))     # Array para las predicciones 
    errors = np.zeros(len(y))       # Array para los errores
    J = []                          # Error total     
   
    while n < t:  
        for i in range(len(X)):  # Iterar sobre cada ejemplo 
            # Calcular el sumatorio de las entradas por los pesos 
            a = np.dot(X[i], w)
            
            # Pasar el valor resultante por la función de activación
            yhat_vec[i] = funcion_activacion(a, z)
            
            # Calcular el error
            error = y[i] - yhat_vec[i]
            
            # Actualizar los pesos 
            w += eta * error * X[i]
        
        # Incremento iteracion
        n += 1
        
        # Calcular el error cuadrático
        for i in range(len(y)):     
            errors[i] = (y[i] - yhat_vec[i])**2
        J.append(0.5 * np.sum(errors))  # Guardamos el error cuadrático
    
    # Devolver los pesos y el error cuadrático
    return w, J


def predice(w, x, z, funcion_activacion):
    """ Función para la predicción 
    Parámetros
    ----------
        w -- array con los pesos obtenidos en el entrenamiento del perceptrón
        x -- valores de x_i para cada uno de los datos de test
        z -- valor del umbral para la función de activación
        funcion_activacion -- función de activación para el perceptrón.
    
    Devolución
    --------
        y -- array con los valores predichos para los datos de test
    """ 
    # Inicializar un array para las predicciones
    y = np.zeros(len(x))
    
    # Calcular la prediccion
    for i in range(len(x)):
        # Calcular el sumatorio (producto punto)
        a = np.dot(x[i], w)
        
        # Pasar resultado por la funcion activacion
        y[i] = funcion_activacion(a, z)
    
    return y


def evalua(y_test, y_pred):
    """ Función de evaluación del perceptrón para calcular el porcentaje de aciertos
    Parámetros
    ----------
        y_test -- array con los valores salida conocidos para los datos de test
        y_pred -- array con los valores salida estimados por el perceptrón para los datos de test
    
    Devolución
    --------
        acierto -- float con el valor del porcentaje de valores acertados con respecto al total de elementos
    """ 
    #cuantos valores predichos coinciden valores reales
    aciertos = np.sum(y_test == y_pred)
    
    # Porcentaje aciertos
    acierto = (aciertos / len(y_test)) * 100
    
    return acierto


def crea_diccionario(archivo_claves):
    dicc={}
    with open(archivo_claves,'r')as archivo:
        for lin in archivo:
            clave,valor=lin.split()
            dicc[int(clave)]=chr(int(valor))
    return dicc

def getdataset(images,labels, caracteres, num_pix):
    from skimage.transform import resize
    import warnings
    """ Obtiene los arrays de numpy con las imágenes y las etiquetas
    Parámetros
    ----------
        imagenes -- estructura de datos que contiene la información de cada una de las imágenes
        etiquetas -- estructura de datos que contiene la información de la clase a la que
        pertenece cada una de las imágenes
        caracteres -- diccionario que contiene la "traducción" a ASCII de cada una de las etiquetas
        num_pix -- valor de la resolución de la imagen (se debe obtener una imagen num_pix x num_pix)
    
    Devolución
    --------
        X -- array 2D (numero_imagenes x numero_pixeles) con los datos de cada una de las imágenes
        y -- array 1D (numero_imagenes) con el caracter que representa cada una de las imágenes
    """ 
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    
    
    X=[]#lista img procesadas
    y=[]#lista etiquetas traducidas
    
    for i in range(len(images)):
        if labels[i] not in caracteres:
            print(f"etiqueta{labels[i]} no tiene un caracter asociado")
            continue
        #redimensionar
        img_resized=resize(images[i], (16,16))
        img_flatten = img_resized.flatten()  #a vector 
        char=caracteres[labels[i]]#obtengo caracter correspondiente
                  
                  
        #añadir los datos
        X.append(img_flatten)
        y.append(char)
                  
                  
    X=np.array(X)
    y=np.array(y)
                  
        

    #Completar el código necesario:
    #  Recorre todos las imágenes y etiquetas
    #  Asigna un caracter al valor de etiqueta
    #  Convierte cada imagen a un vector de tamaño (num_pix * num_pix)
    #  Los datos de salida deben ser un array de numpy.
    #  Si alguna imagen no es correcta, muestra un error y no almacenes ni la imagen ni la etiqueta
    
    return X, y


