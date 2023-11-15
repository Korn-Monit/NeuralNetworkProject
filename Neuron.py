
import numpy as np

class Neuron:
    def __init__(self):
        self.parameters = None # liste d'une seule variable

    parameters = np.array([]) # la liste des variables qu'il faut modifier pour optimiser la prédiction

    def gradient_descent(self,func, gradient, initial_point=np.array([1,1],dtype=float), learning_rate=0.01, num_iterations=2000, tolerance=1e-6, derivationStep =1e-7):
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
            gradient_value = self.gradient(func,current_point,derivationStep)
            # le resultat de la fonction gradient 
            current_point = current_point-learning_rate * gradient_value
            
            # Vérifier la convergence en calculant la norme du gradient
            # pourquoi calculer la norme du gradient ?
            # comment on determine le nombre de torlerance ?
            gradient_norm = np.linalg.norm(gradient_value)
            if gradient_norm < tolerance:
                break
        
        return current_point
    #calculer le dérivé partielle
    # X:np.array, it specify that the X must be a numpy array
    def gradient(self,f,X:np.array,h=1e-6)->np.array:
        result = np.zeros(len(X),dtype=float)
        # i represent the index of the variable
        # val represent the value of the variable
        for i,val in enumerate(X): 

            nextX = X.copy() 
            
            # pourquoi on ajoute h à la variable i ?
            nextX[i]+=h
            partial_derivative = (f(nextX)-f(X))/h

            result[i]=partial_derivative

        return np.array(result)
    def output(point:np.array,self): # la fonction de prédiction après l'entraienement
        return 0
    #même si on appelle ce fonction, il n'execute pas encore la fonction innerfunction, il s'execute seulement quand on le donne les paramètres
    def costFunction(self, points,results): # results est la liste des images de tout point de points, cette fonction retourne la fonction des coût en fonction des paramètres uniquement
        def innerfunction(parameters):
            result = 0
            for i,val in enumerate(points):
                result += (self.output(parameters,val) - results[i])**2
            return result / len(points)
        return innerfunction
    #result = original value
    def train(self,points,results):
        cost = self.costFunction( points,results)
    #here is where we pass the parameter to of gradient descent to var parameter
        self.parameters = self.gradient_descent(cost,self.gradient,np.zeros(len(self.parameters),dtype=float))    

# sub class of Neuron
class NinearNeuron(Neuron):
    def __init__(self):
        super().__init__()
        #this is just the initial value of the training parameter
        self.parameters = np.array([1,1],dtype=float) # matrix = [a,b]

    def output(self,parameters,point:np.array): # point = [x]
        return parameters[0]*point[0]+parameters[1] # y=ax+b

#Testing Neuron

import RandomPoints

Regressor = NinearNeuron()
randpoints = RandomPoints.UniformRandomPoints(1000,lambda x:x,0,10,0.2)

x=[]
y=[]

for i,val in enumerate(randpoints):
    
    x.append(np.array([val[0]]))
    y.append(val[1])

#c'est où on obtiens les paramètres optimisés
Regressor.train(x,y)

