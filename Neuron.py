
import numpy as np
import sympy as sp # pour la dérivation

def gradient_descent(func, gradient, initial_point=np.array([1,1],dtype=float), learning_rate=0.01, num_iterations=2000, tolerance=1e-6, derivationStep =1e-7):
    """
    Fonction pour trouver le minimum d'une fonction avec la méthode du gradient à pas constant.

    :param func: La fonction à minimiser.
    :param gradient: La fonction du gradient de 'func'.
    :param initial_point: Le point de départ de l'algorithme.
    :param learning_rate: Taux d'apprentissage (par défaut, 0.01).
    :param num_iterations: Nombre maximal d'itérations (par défaut, 1000).
    :param tolerance: Tolérance pour la convergence (par défaut, 1e-6).
    :param derivationStep le pas de dérivation de la fonction gradient
    :return: Le point de minimum de 'func'.
    """
    current_point = np.array(initial_point)
    
    for iteration in range(num_iterations):
        gradient_value = gradient(func,current_point,derivationStep)
        current_point = current_point-learning_rate * gradient_value
        
        # Vérifier la convergence en calculant la norme du gradient
        gradient_norm = np.linalg.norm(gradient_value)
        if gradient_norm < tolerance:
            break
    
    return current_point

def func(X:np.array):
    return (X[0] + X[1]-1)**2

# Le gradient de f(X)
def gradient(f,X:np.array,h=1e-6)->np.array:
    result = []
    for i,val in enumerate(X): 

        nextX = X.copy() 
        nextX[i]+=h

        partial_derivative = (f(nextX)-f(X))/h

        result.append(partial_derivative)

    return np.array(result)

class Neuron:
    x=0

