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
