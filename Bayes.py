import numpy as np
from decimal import Decimal
import math
class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        #n_features = X.shape[1]
        n_classes = len(np.unique(y))
        # initialization of parameters
        #self.mu = np.zeros((n_classes, n_features))
        #self.sigma = np.zeros((n_classes, n_features, n_features))
        # learning
        self.mu = np.array([X[np.where(y==i)].mean(axis=0) for i in range(n_classes)])
        self.sigma = np.array([np.cov(X[np.where(y==i)].T) for i in range(n_classes)])


    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        np.seterr(all='raise')
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]
        coeff = - n_features / 2 * np.log(2 * np.pi)
        # initalize the output vector
        y = np.empty(n_obs)
        for i in range(n_obs) :
            scores = np.empty(n_classes)
            for j in range(n_classes) :
                x_sub_mu = X[i] - self.mu[j]
                try :
                    scores[j] = coeff - (1/2) * (np.log(np.linalg.det(self.sigma[j])) + x_sub_mu.T @ np.linalg.inv(self.sigma[j]) @ x_sub_mu)
                    if self.priors is not None:
                        scores[j] += np.log(self.priors[j])
                except :
                    scores[j] = 0
            y[i] = np.argmax(scores)
        return y


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)


    # Fonction qui teste les probabilités à priori possibles et retient celles qui donnent le meilleur score
    def adjust_priors(self, X: np.ndarray, y: np.ndarray) -> None:
        # Probabilités à priori possibles
        # On aurait pu aussi les générer aléatoirement mais après multiples tests, ça ne change rien donc je laisse comme ça
        priors_candidates = np.array([[0.25, 0.55, 0.2], [0.1,0.2,0.7], [0.4,0.1,0.5], [0.6,0.2,0.2], [0.2,0.7,0.1]])

        n_candidates = priors_candidates.shape[0]
        # Si les probas sont déjà définies à l'instanciation, on les prend en compte
        if self.priors is not None :
            max_score = self.score(X,y)
        else :
            max_score = 0
        # test du score pour chaque set de probabilités
        for i in range(n_candidates) :
            previous_priors = self.priors.copy()
            self.priors = priors_candidates[i]
            c_score = self.score(X,y)
            if c_score > max_score :
                max_score = c_score
            else :
                self.priors = previous_priors
