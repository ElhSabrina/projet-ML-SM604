"""
utils.py - Fonctions communes Parties 1 et 2
Projet SM604 - Mathematiques pour le Machine Learning - EFREI 2025-2026
Sabrina El Hassani, Aude Labat, Thomas Duriaud, Paul Fontaine, Evan Ladeira

Contient :
    - softmax, cross_entropy, relu, relu_derivative, one_hot, error_rate
    - Modele lineaire      : init_lin, forward_lin, grad_lin, train_lin
    - Modele H=1 (1 couche): init_h1, forward_h1, backward_h1, train_h1
    - Modele H=2 (2 couches): init_h2, forward_h2, backward_h2, train_h2

Note sur les choix d'implementation :
    softmax et cross_entropy utilisent NumPy car appeles sur 50 000 images
    a chaque epoch. En Python pur, le temps de calcul serait prohibitif.
    Toutes les autres boucles (mini-batches, epochs) restent en Python pur.
"""

import numpy as np
import math


# ----------------------------------------------------------------------------
# Fonctions de base
# ----------------------------------------------------------------------------

def softmax(o):
    """
    Transforme les scores bruts en probabilites.

    On soustrait le max de chaque ligne avant d'exponentier : cela evite
    que exp() explose sur de grands scores (stabilite numerique).

    o : array (n, C) scores bruts
    Retourne P : array (n, C), chaque ligne somme a 1
    """
    o      = np.array(o, dtype=float)
    stable = o - o.max(axis=1, keepdims=True)   # soustraction du max ligne par ligne
    exp_o  = np.exp(stable)
    return exp_o / exp_o.sum(axis=1, keepdims=True)


def cross_entropy(P, Y):
    """
    Entropie croisee moyenne : L = -1/n * sum_i log(P_vraie_classe_i).

    np.clip evite log(0). Y est one-hot donc seul le terme de la vraie
    classe contribue a la somme.

    P : (n, C)   Y : (n, C) one-hot
    Retourne un scalaire
    """
    n      = len(P)
    log_P  = np.log(np.clip(P, 1e-12, 1.0))    # clip evite ln(0)
    return -np.sum(Y * log_P) / n


def relu(x):
    """ReLU : max(0, x) element par element."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivee de ReLU : 1 si x > 0, 0 sinon."""
    return (x > 0).astype(float)


def one_hot(y, C=10):
    """
    Convertit les etiquettes entieres en matrice one-hot.
    y=3 -> [0,0,0,1,0,0,0,0,0,0]

    y : array (n,) entiers entre 0 et C-1
    Retourne Y : array (n, C)
    """
    n = len(y)
    Y = np.zeros((n, C))
    Y[np.arange(n), y] = 1.0
    return Y


def error_rate(P, y_true):
    """
    Proportion d'images mal classees.

    P      : (n, C) probabilites predites
    y_true : (n,) etiquettes entieres
    Retourne un scalaire entre 0 et 1
    """
    n       = len(y_true)
    erreurs = 0
    for i in range(n):
        if int(np.argmax(P[i])) != int(y_true[i]):
            erreurs += 1
    return erreurs / n


# ----------------------------------------------------------------------------
# Modele lineaire : o = A.x + b
# ----------------------------------------------------------------------------

def init_lin(n_input, n_classes=10):
    """
    A (n_classes, n_input) aleatoire * 0.01  : petit pour eviter saturation softmax
    b (n_classes,) = 0
    """
    A = np.random.randn(n_classes, n_input) * 0.01
    b = np.zeros(n_classes)
    return A, b


def forward_lin(X, A, b):
    """
    o = X . A^T + b   produit matriciel NumPy  shape (n, n_classes)
    P = softmax(o)                              shape (n, n_classes)
    Retourne o, P
    """
    o = X @ A.T + b
    P = softmax(o)
    return o, P


def grad_lin(X, P, Y):
    """
    delta = P - Y   erreur par classe  (n, n_classes)
    dA    = delta^T . X / n            (n_classes, n_input)
    db    = moyenne de delta           (n_classes,)
    """
    n     = len(X)
    delta = P - Y
    dA    = (delta.T @ X) / n
    db    = delta.mean(axis=0)
    return dA, db


def train_lin(X_tr, Y_tr, y_tr, X_te, y_te, n_input,
              lr=0.1, batch_size=256, seuil=1e-4, max_epochs=50):
    """
    Descente de gradient mini-batches pour le modele lineaire.

    A chaque epoch :
      1. Melange aleatoire du dataset
      2. Decoupage en mini-batches de taille batch_size
      3. Forward -> gradient -> mise a jour de A et b
      4. Evaluation sur tout le dataset
      5. Criteres d'arret : convergence de la loss ou early stopping
    """
    np.random.seed(42)
    A, b = init_lin(n_input)
    n    = len(X_tr)
    hist_loss, hist_train, hist_test = [], [], []
    epoch, converge = 0, False

    while not converge:

        # Melange aleatoire a chaque epoch
        idx  = np.random.permutation(n)
        X_s, Y_s = X_tr[idx], Y_tr[idx]

        # Mini-batches
        for start in range(0, n, batch_size):
            Xb = X_s[start : start + batch_size]
            Yb = Y_s[start : start + batch_size]
            _, Pb  = forward_lin(Xb, A, b)
            dA, db = grad_lin(Xb, Pb, Yb)
            A -= lr * dA    # A <- A - eta * dL/dA
            b -= lr * db    # b <- b - eta * dL/db

        # Evaluation fin d'epoch
        _, P_tr = forward_lin(X_tr, A, b)
        _, P_te = forward_lin(X_te, A, b)
        loss = cross_entropy(P_tr, Y_tr)
        e_tr = error_rate(P_tr, y_tr)
        e_te = error_rate(P_te, y_te)
        hist_loss.append(loss)
        hist_train.append(e_tr)
        hist_test.append(e_te)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss={loss:.4f} | Train={e_tr*100:.2f}% | Test={e_te*100:.2f}%")

        if epoch > 0:
            if abs(hist_loss[-2] - hist_loss[-1]) < seuil:
                print(f"  Convergence epoch {epoch}")
                converge = True
            elif epoch > 5 and all(hist_test[-i] > hist_test[-i-1] for i in range(1, 5)):
                print(f"  Early stopping epoch {epoch}")
                converge = True
        if epoch >= max_epochs:
            print(f"  Arret : {max_epochs} epochs atteintes")
            converge = True
        epoch += 1

    return A, b, hist_loss, hist_train, hist_test


# ----------------------------------------------------------------------------
# Modele H=1 : une couche cachee
# ----------------------------------------------------------------------------

def init_h1(n_input, n_h1=128, n_classes=10):
    """
    A1 (n_h1, n_input)      poids entree -> couche cachee
    b1 (n_h1,)              biais couche cachee
    A2 (n_classes, n_h1)    poids couche cachee -> sortie
    b2 (n_classes,)         biais sortie
    """
    A1 = np.random.randn(n_h1,      n_input)  * 0.01
    b1 = np.zeros(n_h1)
    A2 = np.random.randn(n_classes, n_h1)     * 0.01
    b2 = np.zeros(n_classes)
    return A1, b1, A2, b2


def forward_h1(X, A1, b1, A2, b2):
    """
    Couche cachee : o1 = X.A1^T + b1   z1 = ReLU(o1)    shape (n, n_h1)
    Sortie        : o2 = z1.A2^T + b2  P  = softmax(o2)  shape (n, n_classes)
    On retourne o1 et z1 pour la retropropagation.
    """
    o1 = X  @ A1.T + b1
    z1 = relu(o1)
    o2 = z1 @ A2.T + b2
    P  = softmax(o2)
    return o1, z1, o2, P


def backward_h1(X, o1, z1, P, Y, A2):
    """
    Regle de la chaine depuis la sortie vers la couche cachee.

    delta2 = P - Y                       erreur sortie       (n, n_classes)
    delta1 = (delta2.A2) * relu'(o1)     erreur couche 1     (n, n_h1)
    relu_derivative bloque le gradient la ou ReLU etait eteint (o1 <= 0).
    """
    n      = len(X)
    delta2 = P - Y
    dA2    = (delta2.T @ z1) / n
    db2    = delta2.mean(axis=0)
    delta1 = (delta2 @ A2) * relu_derivative(o1)
    dA1    = (delta1.T @ X) / n
    db1    = delta1.mean(axis=0)
    return dA1, db1, dA2, db2


def train_h1(X_tr, Y_tr, y_tr, X_te, y_te, n_input,
             lr=0.1, batch_size=256, seuil=1e-4, max_epochs=50):
    """Descente de gradient mini-batches pour le modele H=1."""
    np.random.seed(42)
    A1, b1, A2, b2 = init_h1(n_input)
    n = len(X_tr)
    hist_loss, hist_train, hist_test = [], [], []
    epoch, converge = 0, False

    while not converge:
        idx  = np.random.permutation(n)
        X_s, Y_s = X_tr[idx], Y_tr[idx]

        for start in range(0, n, batch_size):
            Xb = X_s[start : start + batch_size]
            Yb = Y_s[start : start + batch_size]
            o1, z1, o2, Pb     = forward_h1(Xb, A1, b1, A2, b2)
            dA1, db1, dA2, db2 = backward_h1(Xb, o1, z1, Pb, Yb, A2)
            A1 -= lr * dA1
            b1 -= lr * db1
            A2 -= lr * dA2
            b2 -= lr * db2

        _, _, _, P_tr = forward_h1(X_tr, A1, b1, A2, b2)
        _, _, _, P_te = forward_h1(X_te, A1, b1, A2, b2)
        loss = cross_entropy(P_tr, Y_tr)
        e_tr = error_rate(P_tr, y_tr)
        e_te = error_rate(P_te, y_te)
        hist_loss.append(loss)
        hist_train.append(e_tr)
        hist_test.append(e_te)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss={loss:.4f} | Train={e_tr*100:.2f}% | Test={e_te*100:.2f}%")

        if epoch > 0:
            if abs(hist_loss[-2] - hist_loss[-1]) < seuil:
                print(f"  Convergence epoch {epoch}")
                converge = True
            elif epoch > 5 and all(hist_test[-i] > hist_test[-i-1] for i in range(1, 5)):
                print(f"  Early stopping epoch {epoch}")
                converge = True
        if epoch >= max_epochs:
            print(f"  Arret : {max_epochs} epochs atteintes")
            converge = True
        epoch += 1

    return A1, b1, A2, b2, hist_loss, hist_train, hist_test


# ----------------------------------------------------------------------------
# Modele H=2 : deux couches cachees
# ----------------------------------------------------------------------------

def init_h2(n_input, n_h1=128, n_h2=64, n_classes=10):
    """
    A1 (n_h1, n_input)     A2 (n_h2, n_h1)     A3 (n_classes, n_h2)
    et leurs biais respectifs b1, b2, b3.
    """
    A1 = np.random.randn(n_h1,      n_input) * 0.01
    b1 = np.zeros(n_h1)
    A2 = np.random.randn(n_h2,      n_h1)    * 0.01
    b2 = np.zeros(n_h2)
    A3 = np.random.randn(n_classes, n_h2)    * 0.01
    b3 = np.zeros(n_classes)
    return A1, b1, A2, b2, A3, b3


def forward_h2(X, A1, b1, A2, b2, A3, b3):
    """
    Couche 1 : o1, z1 = ReLU(X.A1^T + b1)      (n, n_h1)
    Couche 2 : o2, z2 = ReLU(z1.A2^T + b2)     (n, n_h2)
    Sortie   : o3, P  = softmax(z2.A3^T + b3)   (n, n_classes)
    """
    o1 = X  @ A1.T + b1
    z1 = relu(o1)
    o2 = z1 @ A2.T + b2
    z2 = relu(o2)
    o3 = z2 @ A3.T + b3
    P  = softmax(o3)
    return o1, z1, o2, z2, o3, P


def backward_h2(X, o1, z1, o2, z2, P, Y, A2, A3):
    """
    Retropropagation H=2 : on remonte de delta3 (sortie) vers delta1
    en appliquant la regle de la chaine a chaque couche.
    """
    n      = len(X)
    delta3 = P - Y
    dA3    = (delta3.T @ z2) / n
    db3    = delta3.mean(axis=0)
    delta2 = (delta3 @ A3) * relu_derivative(o2)
    dA2    = (delta2.T @ z1) / n
    db2    = delta2.mean(axis=0)
    delta1 = (delta2 @ A2) * relu_derivative(o1)
    dA1    = (delta1.T @ X) / n
    db1    = delta1.mean(axis=0)
    return dA1, db1, dA2, db2, dA3, db3


def train_h2(X_tr, Y_tr, y_tr, X_te, y_te, n_input,
             lr=0.1, batch_size=256, seuil=1e-4, max_epochs=50):
    """Descente de gradient mini-batches pour le modele H=2."""
    np.random.seed(42)
    A1, b1, A2, b2, A3, b3 = init_h2(n_input)
    n = len(X_tr)
    hist_loss, hist_train, hist_test = [], [], []
    epoch, converge = 0, False

    while not converge:
        idx  = np.random.permutation(n)
        X_s, Y_s = X_tr[idx], Y_tr[idx]

        for start in range(0, n, batch_size):
            Xb = X_s[start : start + batch_size]
            Yb = Y_s[start : start + batch_size]
            o1, z1, o2, z2, o3, Pb           = forward_h2(Xb, A1, b1, A2, b2, A3, b3)
            dA1, db1, dA2, db2, dA3, db3     = backward_h2(Xb, o1, z1, o2, z2, Pb, Yb, A2, A3)
            A1 -= lr * dA1
            b1 -= lr * db1
            A2 -= lr * dA2
            b2 -= lr * db2
            A3 -= lr * dA3
            b3 -= lr * db3

        _, _, _, _, _, P_tr = forward_h2(X_tr, A1, b1, A2, b2, A3, b3)
        _, _, _, _, _, P_te = forward_h2(X_te, A1, b1, A2, b2, A3, b3)
        loss = cross_entropy(P_tr, Y_tr)
        e_tr = error_rate(P_tr, y_tr)
        e_te = error_rate(P_te, y_te)
        hist_loss.append(loss)
        hist_train.append(e_tr)
        hist_test.append(e_te)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | Loss={loss:.4f} | Train={e_tr*100:.2f}% | Test={e_te*100:.2f}%")

        if epoch > 0:
            if abs(hist_loss[-2] - hist_loss[-1]) < seuil:
                print(f"  Convergence epoch {epoch}")
                converge = True
            elif epoch > 5 and all(hist_test[-i] > hist_test[-i-1] for i in range(1, 5)):
                print(f"  Early stopping epoch {epoch}")
                converge = True
        if epoch >= max_epochs:
            print(f"  Arret : {max_epochs} epochs atteintes")
            converge = True
        epoch += 1

    return A1, b1, A2, b2, A3, b3, hist_loss, hist_train, hist_test
