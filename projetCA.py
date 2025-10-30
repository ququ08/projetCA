"""
Logiciel Complet de Résolution d'Équations Non Linéaires
Méthodes: Dichotomie, Point Fixe, Newton-Raphson, Sécante

Fonctionnalités:
- Résolution individuelle et comparative
- Analyse des erreurs d'arrondi
- Visualisations avancées
- Export de rapports

Auteurs: [Votre Groupe]
Date: Octobre 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import sympy as sp
from abc import ABC, abstractmethod
import time
import sys
from io import BytesIO
import base64

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Résolution d'Équations Non Linéaires",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CLASSES DES SOLVEURS ====================

class Solver(ABC):
    """Classe abstraite pour tous les solveurs"""
    
    def __init__(self, tolerance=1e-6, max_iter=100, precision='double'):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.precision = precision
        self.historique = []
        self.temps_execution = 0
        self.converged = False
        self.epsilon_machine = self._get_epsilon_machine()
    
    def _get_epsilon_machine(self):
        """Calcule l'epsilon machine selon la précision"""
        if self.precision == 'simple':
            return np.finfo(np.float32).eps
        else:
            return np.finfo(np.float64).eps
    
    @abstractmethod
    def solve(self, *args, **kwargs):
        """Méthode abstraite à implémenter par chaque solveur"""
        pass
    
    def get_historique_df(self):
        """Retourne l'historique sous forme de DataFrame"""
        return pd.DataFrame(self.historique)
    
    def get_nom_methode(self):
        """Retourne le nom de la méthode"""
        return self.__class__.__name__.replace('Solver', '')


class DichotomieSolver(Solver):
    """Solveur par méthode de dichotomie"""
    
    def solve(self, f, a, b):
        """Résout f(x) = 0 sur [a, b] par dichotomie"""
        start_time = time.time()
        self.historique = []
        
        if self.precision == 'simple':
            a, b = np.float32(a), np.float32(b)
        
        fa = f(a)
        fb = f(b)
        
        if fa * fb >= 0:
            raise ValueError("f(a) et f(b) doivent avoir des signes opposés")
        
        for n in range(self.max_iter):
            c = (a + b) / 2
            if self.precision == 'simple':
                c = np.float32(c)
            
            fc = f(c)
            erreur = (b - a) / 2
            
            self.historique.append({
                'Itération': n,
                'a': float(a),
                'b': float(b),
                'c': float(c),
                'f(c)': float(fc),
                'Erreur': float(erreur),
                'Erreur_log': float(np.log10(abs(erreur) + 1e-16))
            })
            
            if abs(fc) < self.tolerance or erreur < self.tolerance:
                self.temps_execution = time.time() - start_time
                self.converged = True
                return c, n + 1, True
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        self.temps_execution = time.time() - start_time
        return c, self.max_iter, False


class PointFixeSolver(Solver):
    """Solveur par méthode du point fixe"""
    
    def solve(self, g, x0):
        """Résout x = g(x) par itération"""
        start_time = time.time()
        self.historique = []
        
        x = np.float32(x0) if self.precision == 'simple' else x0
        
        for n in range(self.max_iter):
            try:
                x_new = g(x)
                if self.precision == 'simple':
                    x_new = np.float32(x_new)
                
                erreur = abs(x_new - x)
                
                self.historique.append({
                    'Itération': n,
                    'xₙ': float(x),
                    'g(xₙ)': float(x_new),
                    'Erreur': float(erreur),
                    'Erreur_log': float(np.log10(abs(erreur) + 1e-16))
                })
                
                if erreur < self.tolerance:
                    self.temps_execution = time.time() - start_time
                    self.converged = True
                    return x_new, n + 1, True
                
                if abs(x_new) > 1e10:
                    self.temps_execution = time.time() - start_time
                    raise RuntimeError("Divergence détectée")
                
                x = x_new
            except Exception as e:
                self.temps_execution = time.time() - start_time
                raise e
        
        self.temps_execution = time.time() - start_time
        return x, self.max_iter, False


class NewtonRaphsonSolver(Solver):
    """Solveur par méthode de Newton-Raphson"""
    
    def solve(self, f, df, x0):
        """Résout f(x) = 0 par Newton-Raphson"""
        start_time = time.time()
        self.historique = []
        
        x = np.float32(x0) if self.precision == 'simple' else x0
        
        for n in range(self.max_iter):
            try:
                fx = f(x)
                dfx = df(x)
                
                if abs(dfx) < 1e-12:
                    self.temps_execution = time.time() - start_time
                    raise RuntimeError(f"Dérivée trop petite: f'({x:.6f}) = {dfx:.2e}")
                
                x_new = x - fx / dfx
                if self.precision == 'simple':
                    x_new = np.float32(x_new)
                
                erreur = abs(x_new - x)
                
                self.historique.append({
                    'Itération': n,
                    'xₙ': float(x),
                    'f(xₙ)': float(fx),
                    "f'(xₙ)": float(dfx),
                    'xₙ₊₁': float(x_new),
                    'Erreur': float(erreur),
                    'Erreur_log': float(np.log10(abs(erreur) + 1e-16))
                })
                
                if erreur < self.tolerance or abs(fx) < self.tolerance:
                    self.temps_execution = time.time() - start_time
                    self.converged = True
                    return x_new, n + 1, True
                
                x = x_new
            except Exception as e:
                self.temps_execution = time.time() - start_time
                raise e
        
        self.temps_execution = time.time() - start_time
        return x, self.max_iter, False


class SecanteSolver(Solver):
    """Solveur par méthode de la sécante"""
    
    def solve(self, f, x0, x1):
        """Résout f(x) = 0 par la méthode de la sécante"""
        start_time = time.time()
        self.historique = []
        
        if self.precision == 'simple':
            x0, x1 = np.float32(x0), np.float32(x1)
        
        for n in range(self.max_iter):
            try:
                fx0 = f(x0)
                fx1 = f(x1)
                
                if abs(fx1 - fx0) < 1e-12:
                    self.temps_execution = time.time() - start_time
                    raise RuntimeError(f"Dénominateur trop petit à l'itération {n}")
                
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                if self.precision == 'simple':
                    x2 = np.float32(x2)
                
                erreur = abs(x2 - x1)
                
                self.historique.append({
                    'Itération': n,
                    'xₙ₋₁': float(x0),
                    'xₙ': float(x1),
                    'f(xₙ₋₁)': float(fx0),
                    'f(xₙ)': float(fx1),
                    'xₙ₊₁': float(x2),
                    'Erreur': float(erreur),
                    'Erreur_log': float(np.log10(abs(erreur) + 1e-16))
                })
                
                if erreur < self.tolerance or abs(fx1) < self.tolerance:
                    self.temps_execution = time.time() - start_time
                    self.converged = True
                    return x2, n + 1, True
                
                x0, x1 = x1, x2
            except Exception as e:
                self.temps_execution = time.time() - start_time
                raise e
        
        self.temps_execution = time.time() - start_time
        return x2, self.max_iter, False


# ==================== FONCTIONS UTILITAIRES ====================

def parse_function(expr_str, var='x'):
    """Parse une expression mathématique"""
    try:
        x = sp.Symbol(var)
        expr = sp.sympify(expr_str)
        f = sp.lambdify(x, expr, 'numpy')
        return f, expr
    except Exception as e:
        raise ValueError(f"Erreur de parsing: {str(e)}")


def calculate_derivative(expr, var='x'):
    """Calcule la dérivée symbolique"""
    x = sp.Symbol(var)
    df_expr = sp.diff(expr, x)
    df = sp.lambdify(x, df_expr, 'numpy')
    return df, df_expr


def plot_function_detailed(f, x_min, x_max, racine=None, historique=None, methode="", show_iterations=True):
    """Visualisation détaillée avec itérations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphe 1: Fonction avec itérations
    x = np.linspace(x_min, x_max, 1000)
    try:
        y = f(x)
    except:
        y = [f(xi) for xi in x]
    
    ax1.plot(x, y, 'b-', linewidth=2.5, label='f(x)', zorder=1)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_xlabel('x', fontsize=13, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Fonction f(x) - {methode}', fontsize=14, fontweight='bold', pad=15)
    
    if racine is not None:
        ax1.axvline(x=racine, color='red', linestyle='--', linewidth=2, 
                   label=f'Racine ≈ {racine:.8f}', alpha=0.7, zorder=2)
        ax1.plot(racine, 0, 'ro', markersize=12, label='Racine trouvée', 
                zorder=5, markeredgecolor='darkred', markeredgewidth=2)
    
    # Marquer les itérations avec trajectoire
    if historique is not None and len(historique) > 0 and show_iterations:
        df_hist = pd.DataFrame(historique)
        
        if 'c' in df_hist.columns:  # Dichotomie
            points_x = df_hist['c'].values
            points_y = [f(xi) for xi in points_x]
            ax1.plot(points_x, points_y, 'go', markersize=6, alpha=0.6, 
                    label='Itérations', zorder=3)
            for i in range(len(points_x)-1):
                ax1.plot([points_x[i], points_x[i+1]], [points_y[i], points_y[i+1]], 
                        'g--', alpha=0.3, linewidth=1)
        
        elif 'xₙ₊₁' in df_hist.columns and "f'(xₙ)" in df_hist.columns:  # Newton
            for idx, row in df_hist.iterrows():
                xn = row['xₙ']
                fn = row['f(xₙ)']
                dfn = row["f'(xₙ)"]
                
                ax1.plot(xn, fn, 'go', markersize=6, alpha=0.6, zorder=3)
                
                # Tracer la tangente
                if abs(dfn) > 1e-10:
                    x_tangent = np.array([xn - 0.5, xn + 0.5])
                    y_tangent = fn + dfn * (x_tangent - xn)
                    ax1.plot(x_tangent, y_tangent, 'orange', alpha=0.4, 
                            linewidth=1.5, linestyle='--')
        
        elif 'xₙ₊₁' in df_hist.columns:  # Sécante
            for idx, row in df_hist.iterrows():
                if idx == 0:
                    ax1.plot([row['xₙ₋₁'], row['xₙ']], 
                            [row['f(xₙ₋₁)'], row['f(xₙ)']], 
                            'g-', alpha=0.4, linewidth=1.5)
                ax1.plot(row['xₙ'], row['f(xₙ)'], 'go', markersize=6, alpha=0.6)
        
        elif 'xₙ' in df_hist.columns:  # Point fixe
            points_x = df_hist['xₙ'].values
            points_y = [f(xi) for xi in points_x]
            ax1.plot(points_x, points_y, 'go', markersize=6, alpha=0.6, 
                    label='Itérations', zorder=3)
    
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Graphe 2: Convergence logarithmique
    if historique is not None and len(historique) > 0:
        df_hist = pd.DataFrame(historique)
        iterations = df_hist['Itération'].values
        erreurs = df_hist['Erreur'].values
        
        ax2.semilogy(iterations, erreurs, 'ro-', linewidth=2.5, markersize=7, 
                    markerfacecolor='red', markeredgecolor='darkred', 
                    markeredgewidth=1.5, label=methode)
        ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=2, 
                   label='Tolérance (10⁻⁶)', alpha=0.7)
        ax2.grid(True, alpha=0.4, which='both', linestyle=':', linewidth=0.8)
        ax2.set_xlabel('Itération', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Erreur (échelle log)', fontsize=13, fontweight='bold')
        ax2.set_title('Courbe de convergence', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Annoter la dernière itération
        if len(iterations) > 0:
            ax2.annotate(f'{erreurs[-1]:.2e}', 
                        xy=(iterations[-1], erreurs[-1]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    return fig


def plot_comparison(resultats_dict, f, x_min, x_max, racine_reference):
    """Graphique comparatif de toutes les méthodes"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Graphe 1: Fonction avec toutes les racines
    x = np.linspace(x_min, x_max, 1000)
    try:
        y = f(x)
    except:
        y = [f(xi) for xi in x]
    
    ax1.plot(x, y, 'b-', linewidth=2.5, label='f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    couleurs = {'Dichotomie': 'red', 'PointFixe': 'green', 
                'NewtonRaphson': 'orange', 'Secante': 'purple'}
    
    for methode, data in resultats_dict.items():
        if data['racine'] is not None:
            ax1.axvline(x=data['racine'], color=couleurs.get(methode, 'gray'), 
                       linestyle='--', alpha=0.6, label=f"{methode}: {data['racine']:.6f}")
    
    ax1.set_xlabel('x', fontsize=12, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=12, fontweight='bold')
    ax1.set_title('Racines trouvées par chaque méthode', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    
    # Graphe 2: Comparaison des convergences
    for methode, data in resultats_dict.items():
        if data['historique'] is not None and len(data['historique']) > 0:
            df_hist = pd.DataFrame(data['historique'])
            ax2.semilogy(df_hist['Itération'], df_hist['Erreur'], 
                        'o-', linewidth=2, markersize=5, 
                        label=methode, color=couleurs.get(methode, 'gray'))
    
    ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=1.5, 
               label='Tolérance', alpha=0.7)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlabel('Itération', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Erreur (échelle log)', fontsize=12, fontweight='bold')
    ax2.set_title('Comparaison des convergences', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    
    # Graphe 3: Nombre d'itérations
    methodes = list(resultats_dict.keys())
    iterations = [resultats_dict[m]['nb_iterations'] for m in methodes]
    couleurs_bar = [couleurs.get(m, 'gray') for m in methodes]
    
    bars = ax3.bar(methodes, iterations, color=couleurs_bar, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Nombre d\'itérations', fontsize=12, fontweight='bold')
    ax3.set_title('Nombre d\'itérations par méthode', fontsize=13, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Annoter les barres
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Graphe 4: Temps d'exécution
    temps = [resultats_dict[m]['temps_execution']*1000 for m in methodes]
    bars = ax4.bar(methodes, temps, color=couleurs_bar, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Temps (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Temps d\'exécution par méthode', fontsize=13, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Annoter les barres
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_comparison_table(resultats_dict):
    """Crée un tableau comparatif des résultats"""
    data = []
    for methode, res in resultats_dict.items():
        data.append({
            'Méthode': methode,
            'Racine': f"{res['racine']:.10f}" if res['racine'] is not None else "Échec",
            'Itérations': res['nb_iterations'],
            'Temps (ms)': f"{res['temps_execution']*1000:.4f}",
            'Convergé': '✅' if res['converged'] else '❌',
            'Résidu |f(x)|': f"{res['residu']:.2e}" if res['residu'] is not None else "N/A"
        })
    
    return pd.DataFrame(data)


def analyze_rounding_errors(f, methode_class, x0_or_interval, tolerance=1e-6):
    """Analyse l'impact des erreurs d'arrondi"""
    resultats = {}
    
    # Test en double précision
    solver_double = methode_class(tolerance=tolerance, precision='double')
    
    # Test en simple précision
    solver_simple = methode_class(tolerance=tolerance, precision='simple')
    
    try:
        if methode_class == DichotomieSolver:
            a, b = x0_or_interval
            racine_double, _, _ = solver_double.solve(f, a, b)
            racine_simple, _, _ = solver_simple.solve(f, a, b)
        elif methode_class == NewtonRaphsonSolver:
            df, _ = calculate_derivative(sp.sympify("exp(x) - 3*x"))
            racine_double, _, _ = solver_double.solve(f, df, x0_or_interval)
            racine_simple, _, _ = solver_simple.solve(f, df, x0_or_interval)
        else:
            racine_double, _, _ = solver_double.solve(f, x0_or_interval)
            racine_simple, _, _ = solver_simple.solve(f, x0_or_interval)
        
        resultats['double'] = {
            'racine': racine_double,
            'historique': solver_double.historique,
            'epsilon': solver_double.epsilon_machine
        }
        
        resultats['simple'] = {
            'racine': racine_simple,
            'historique': solver_simple.historique,
            'epsilon': solver_simple.epsilon_machine
        }
        
        resultats['difference'] = abs(racine_double - racine_simple)
        
    except Exception as e:
        st.error(f"Erreur dans l'analyse: {str(e)}")
        return None
    
    return resultats


# ==================== INTERFACE STREAMLIT ====================

def main():
    # En-tête
    st.markdown('<h1 class="main-header">📊 Résolution d\'Équations Non Linéaires</h1>', 
                unsafe_allow_html=True)
    st.markdown("### *Analyse complète et comparative des méthodes numériques*")
    st.markdown("---")
    
    # Sidebar: Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Mode d'utilisation
    mode = st.sidebar.radio(
        "Mode d'utilisation",
        ["🎯 Résolution Simple", "⚖️ Comparaison Multiple", "🔬 Analyse Erreurs d'Arrondi"],
        help="Choisissez le mode d'analyse"
    )
    
    st.sidebar.markdown("---")
    
    # Choix de la méthode (si mode simple)
    if mode == "🎯 Résolution Simple":
        methode = st.sidebar.selectbox(
            "Méthode de résolution",
            ["Dichotomie", "Point Fixe", "Newton-Raphson", "Sécante"]
        )
    
    # Paramètres communs
    st.sidebar.subheader("Paramètres généraux")
    tolerance = st.sidebar.number_input("Tolérance", value=1e-6, format="%.1e", 
                                       min_value=1e-12, max_value=1e-1)
    max_iter = st.sidebar.number_input("Itérations max", value=100, min_value=10, max_value=1000)
    
    # Exemples prédéfinis
    st.sidebar.subheader("📝 Exemples prédéfinis")
    exemple = st.sidebar.selectbox(
        "Charger un exemple",
        ["Personnalisé", "exp(x) = 3x", "x³ - 4x + 1 = 0", "sin(x) = x/2", "x² - 2 = 0", "cos(x) - x = 0"]
    )
    
    # Configuration selon l'exemple
    examples_config = {
        "exp(x) = 3x": {
            'f': "exp(x) - 3*x",
            'a': 0.0, 'b': 1.0,
            'x0': 0.5,
            'g': "(log(3) + log(x))"
        },
        "x³ - 4x + 1 = 0": {
            'f': "x**3 - 4*x + 1",
            'a': 0.0, 'b': 1.0,
            'x0': 0.5,
            'g': "(x**3 + 1)/4"
        },
        "sin(x) = x/2": {
            'f': "sin(x) - x/2",
            'a': 1.0, 'b': 2.5,
            'x0': 2.0,
            'g': "2*sin(x)"
        },
        "x² - 2 = 0": {
            'f': "x**2 - 2",
            'a': 1.0, 'b': 2.0,
            'x0': 1.5,
            'g': "2/x"
        },
        "cos(x) - x = 0": {
            'f': "cos(x) - x",
            'a': 0.0, 'b': 1.0,
            'x0': 0.5,
            'g': "cos(x)"
        },
        "Personnalisé": {
            'f': "x**2 - 4",
            'a': 0.0, 'b': 3.0,
            'x0': 1.0,
            'g': "sqrt(4)"
        }
    }
    
    config = examples_config.get(exemple, examples_config["Personnalisé"])
    
    st.sidebar.markdown("---")
    
    # ==================== MODE RÉSOLUTION SIMPLE ====================
    if mode == "🎯 Résolution Simple":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📐 Définition du problème")
            
            if methode == "Point Fixe":
                st.info("**Format:** Entrez g(x) tel que x = g(x)")
                fonction_input = st.text_input("Fonction g(x)", value=config['g'])
            else:
                st.info("**Format:** Entrez f(x) pour résoudre f(x) = 0")
                fonction_input = st.text_input("Fonction f(x)", value=config['f'])
            
            st.caption("Syntaxe: exp(x), sin(x), cos(x), log(x), sqrt(x), x**2, abs(x), etc.")
        
        with col2:
            st.subheader("🎯 Paramètres initiaux")
            
            if methode == "Dichotomie":
                a = st.number_input("Borne a", value=config['a'], format="%.4f")
                b = st.number_input("Borne b", value=config['b'], format="%.4f")
            elif methode == "Point Fixe" or methode == "Newton-Raphson":
                x0 = st.number_input("Valeur initiale x₀", value=config['x0'], format="%.4f")
            elif methode == "Sécante":
                x0 = st.number_input("Première valeur x₀", value=config['x0'], format="%.4f")
                x1 = st.number_input("Deuxième valeur x₁", value=config['x0'] + 0.5, format="%.4f")
        
        # Bouton de résolution
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            solve_button = st.button("🚀 Résoudre", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Réinitialiser", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        # Résolution
        if solve_button:
            try:
                # Parsing de la fonction
                if methode == "Point Fixe":
                    g, g_expr = parse_function(fonction_input)
                    st.success(f"✅ Fonction g(x) parsée: {g_expr}")
                else:
                    f, f_expr = parse_function(fonction_input)
                    st.success(f"✅ Fonction f(x) parsée: {f_expr}")
                
                # Création du solveur
                solver = None
                racine = None
                nb_iter = 0
                converged = False
                
                if methode == "Dichotomie":
                    st.info(f"🔍 Recherche sur l'intervalle [{a}, {b}]")
                    solver = DichotomieSolver(tolerance=tolerance, max_iter=max_iter)
                    racine, nb_iter, converged = solver.solve(f, a, b)
                    
                elif methode == "Point Fixe":
                    st.info(f"🔍 Itération depuis x₀ = {x0}")
                    solver = PointFixeSolver(tolerance=tolerance, max_iter=max_iter)
                    racine, nb_iter, converged = solver.solve(g, x0)
                    # Recréer f pour la visualisation
                    f, f_expr = parse_function(fonction_input + " - x")
                    
                elif methode == "Newton-Raphson":
                    st.info(f"🔍 Itération depuis x₀ = {x0}")
                    df, df_expr = calculate_derivative(f_expr)
                    st.caption(f"Dérivée calculée: f'(x) = {df_expr}")
                    solver = NewtonRaphsonSolver(tolerance=tolerance, max_iter=max_iter)
                    racine, nb_iter, converged = solver.solve(f, df, x0)
                    
                elif methode == "Sécante":
                    st.info(f"🔍 Itération depuis x₀ = {x0} et x₁ = {x1}")
                    solver = SecanteSolver(tolerance=tolerance, max_iter=max_iter)
                    racine, nb_iter, converged = solver.solve(f, x0, x1)
                
                # Affichage des résultats
                st.markdown("---")
                st.subheader("📊 Résultats")
                
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("Racine trouvée", f"{racine:.10f}")
                with col_r2:
                    st.metric("Itérations", nb_iter)
                with col_r3:
                    st.metric("Temps (ms)", f"{solver.temps_execution*1000:.4f}")
                with col_r4:
                    if converged:
                        st.success("✅ Convergé")
                    else:
                        st.warning("⚠️ Max itérations")
                
                # Vérification
                if methode == "Point Fixe":
                    verif = abs(racine - g(racine))
                    st.info(f"**Vérification:** |x - g(x)| = {verif:.2e}")
                else:
                    verif = abs(f(racine))
                    st.info(f"**Vérification:** |f(x)| = {verif:.2e}")
                
                # Tableau d'itérations
                st.markdown("---")
                st.subheader("📋 Tableau de convergence")
                
                df_hist = solver.get_historique_df()
                st.dataframe(df_hist, use_container_width=True, height=400)
                
                # Bouton de téléchargement
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name=f"convergence_{methode.lower().replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv"
                )
                
                # Graphiques
                st.markdown("---")
                st.subheader("📈 Visualisation détaillée")
                
                # Déterminer les bornes pour le graphe
                if methode == "Dichotomie":
                    x_min_plot, x_max_plot = a - 0.5, b + 0.5
                else:
                    x_min_plot = min(racine - 2, x0 - 1)
                    x_max_plot = max(racine + 2, x0 + 1)
                
                fig = plot_function_detailed(f, x_min_plot, x_max_plot, racine, 
                                            solver.historique, methode, show_iterations=True)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                st.exception(e)
    
    # ==================== MODE COMPARAISON MULTIPLE ====================
    elif mode == "⚖️ Comparaison Multiple":
        st.subheader("📐 Définition du problème")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("**Format:** Entrez f(x) pour résoudre f(x) = 0")
            fonction_input = st.text_input("Fonction f(x)", value=config['f'])
            st.caption("Syntaxe: exp(x), sin(x), cos(x), log(x), sqrt(x), x**2, abs(x), etc.")
        
        with col2:
            st.subheader("🎯 Paramètres")
            a = st.number_input("Borne a", value=config['a'], format="%.4f")
            b = st.number_input("Borne b", value=config['b'], format="%.4f")
            x0 = st.number_input("x₀ initial", value=config['x0'], format="%.4f")
        
        st.markdown("---")
        
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            compare_button = st.button("🔬 Comparer toutes les méthodes", type="primary", use_container_width=True)
        
        if compare_button:
            try:
                f, f_expr = parse_function(fonction_input)
                st.success(f"✅ Fonction f(x) parsée: {f_expr}")
                
                # Préparer g(x) pour point fixe
                g, g_expr = parse_function(config['g'])
                
                # Calculer dérivée pour Newton
                df, df_expr = calculate_derivative(f_expr)
                
                resultats = {}
                
                with st.spinner("🔄 Résolution en cours..."):
                    # Dichotomie
                    try:
                        solver_dicho = DichotomieSolver(tolerance=tolerance, max_iter=max_iter)
                        racine_dicho, nb_iter_dicho, conv_dicho = solver_dicho.solve(f, a, b)
                        resultats['Dichotomie'] = {
                            'racine': racine_dicho,
                            'nb_iterations': nb_iter_dicho,
                            'temps_execution': solver_dicho.temps_execution,
                            'converged': conv_dicho,
                            'historique': solver_dicho.historique,
                            'residu': abs(f(racine_dicho))
                        }
                    except Exception as e:
                        resultats['Dichotomie'] = {
                            'racine': None, 'nb_iterations': 0, 'temps_execution': 0,
                            'converged': False, 'historique': None, 'residu': None
                        }
                        st.warning(f"⚠️ Dichotomie: {str(e)}")
                    
                    # Point Fixe
                    try:
                        solver_pf = PointFixeSolver(tolerance=tolerance, max_iter=max_iter)
                        racine_pf, nb_iter_pf, conv_pf = solver_pf.solve(g, x0)
                        resultats['PointFixe'] = {
                            'racine': racine_pf,
                            'nb_iterations': nb_iter_pf,
                            'temps_execution': solver_pf.temps_execution,
                            'converged': conv_pf,
                            'historique': solver_pf.historique,
                            'residu': abs(f(racine_pf))
                        }
                    except Exception as e:
                        resultats['PointFixe'] = {
                            'racine': None, 'nb_iterations': 0, 'temps_execution': 0,
                            'converged': False, 'historique': None, 'residu': None
                        }
                        st.warning(f"⚠️ Point Fixe: {str(e)}")
                    
                    # Newton-Raphson
                    try:
                        solver_newton = NewtonRaphsonSolver(tolerance=tolerance, max_iter=max_iter)
                        racine_newton, nb_iter_newton, conv_newton = solver_newton.solve(f, df, x0)
                        resultats['NewtonRaphson'] = {
                            'racine': racine_newton,
                            'nb_iterations': nb_iter_newton,
                            'temps_execution': solver_newton.temps_execution,
                            'converged': conv_newton,
                            'historique': solver_newton.historique,
                            'residu': abs(f(racine_newton))
                        }
                    except Exception as e:
                        resultats['NewtonRaphson'] = {
                            'racine': None, 'nb_iterations': 0, 'temps_execution': 0,
                            'converged': False, 'historique': None, 'residu': None
                        }
                        st.warning(f"⚠️ Newton-Raphson: {str(e)}")
                    
                    # Sécante
                    try:
                        solver_sec = SecanteSolver(tolerance=tolerance, max_iter=max_iter)
                        racine_sec, nb_iter_sec, conv_sec = solver_sec.solve(f, x0, x0 + 0.1)
                        resultats['Secante'] = {
                            'racine': racine_sec,
                            'nb_iterations': nb_iter_sec,
                            'temps_execution': solver_sec.temps_execution,
                            'converged': conv_sec,
                            'historique': solver_sec.historique,
                            'residu': abs(f(racine_sec))
                        }
                    except Exception as e:
                        resultats['Secante'] = {
                            'racine': None, 'nb_iterations': 0, 'temps_execution': 0,
                            'converged': False, 'historique': None, 'residu': None
                        }
                        st.warning(f"⚠️ Sécante: {str(e)}")
                
                st.success("✅ Comparaison terminée!")
                
                # Tableau comparatif
                st.markdown("---")
                st.subheader("📊 Tableau comparatif des résultats")
                
                df_comp = create_comparison_table(resultats)
                st.dataframe(df_comp, use_container_width=True)
                
                # Téléchargement
                csv_comp = df_comp.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger le tableau comparatif",
                    data=csv_comp,
                    file_name="comparaison_methodes.csv",
                    mime="text/csv"
                )
                
                # Analyse comparative
                st.markdown("---")
                st.subheader("🎯 Analyse comparative")
                
                col_a1, col_a2, col_a3 = st.columns(3)
                
                with col_a1:
                    st.markdown("**🏆 Méthode la plus rapide**")
                    methode_rapide = min(resultats.items(), 
                                        key=lambda x: x[1]['nb_iterations'] if x[1]['converged'] else float('inf'))
                    st.info(f"{methode_rapide[0]}\n\n{methode_rapide[1]['nb_iterations']} itérations")
                
                with col_a2:
                    st.markdown("**⚡ Temps d'exécution minimal**")
                    methode_temps = min(resultats.items(), 
                                       key=lambda x: x[1]['temps_execution'] if x[1]['converged'] else float('inf'))
                    st.info(f"{methode_temps[0]}\n\n{methode_temps[1]['temps_execution']*1000:.4f} ms")
                
                with col_a3:
                    st.markdown("**🎯 Meilleure précision**")
                    methode_precise = min(resultats.items(), 
                                         key=lambda x: x[1]['residu'] if x[1]['residu'] is not None else float('inf'))
                    st.info(f"{methode_precise[0]}\n\n|f(x)| = {methode_precise[1]['residu']:.2e}")
                
                # Graphiques comparatifs
                st.markdown("---")
                st.subheader("📈 Visualisations comparatives")
                
                racine_ref = next((r['racine'] for r in resultats.values() if r['racine'] is not None), x0)
                fig_comp = plot_comparison(resultats, f, a - 0.5, b + 0.5, racine_ref)
                st.pyplot(fig_comp)
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                st.exception(e)
    
    # ==================== MODE ANALYSE ERREURS D'ARRONDI ====================
    elif mode == "🔬 Analyse Erreurs d'Arrondi":
        st.subheader("🔬 Analyse de l'impact des erreurs d'arrondi")
        
        st.info("""
        **Cette analyse compare les résultats en simple précision (float32) et double précision (float64)**
        
        - **Simple précision**: ~7 chiffres décimaux, epsilon ≈ 1.2×10⁻⁷
        - **Double précision**: ~15 chiffres décimaux, epsilon ≈ 2.2×10⁻¹⁶
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fonction_input = st.text_input("Fonction f(x)", value=config['f'])
            st.caption("Syntaxe: exp(x), sin(x), cos(x), log(x), sqrt(x), x**2, abs(x), etc.")
        
        with col2:
            methode_analyse = st.selectbox(
                "Méthode à analyser",
                ["Dichotomie", "Newton-Raphson", "Point Fixe", "Sécante"]
            )
            
            if methode_analyse == "Dichotomie":
                a = st.number_input("Borne a", value=config['a'], format="%.4f")
                b = st.number_input("Borne b", value=config['b'], format="%.4f")
                param = (a, b)
            else:
                x0 = st.number_input("x₀ initial", value=config['x0'], format="%.4f")
                param = x0
        
        st.markdown("---")
        
        analyze_button = st.button("🔬 Lancer l'analyse", type="primary", use_container_width=True)
        
        if analyze_button:
            try:
                f, f_expr = parse_function(fonction_input)
                st.success(f"✅ Fonction f(x) parsée: {f_expr}")
                
                # Sélection de la classe
                methode_classes = {
                    "Dichotomie": DichotomieSolver,
                    "Newton-Raphson": NewtonRaphsonSolver,
                    "Point Fixe": PointFixeSolver,
                    "Sécante": SecanteSolver
                }
                
                methode_class = methode_classes[methode_analyse]
                
                with st.spinner("🔄 Analyse en cours..."):
                    resultats_arrondi = analyze_rounding_errors(f, methode_class, param, tolerance)
                
                if resultats_arrondi is not None:
                    st.success("✅ Analyse terminée!")
                    
                    # Affichage des résultats
                    st.markdown("---")
                    st.subheader("📊 Résultats de l'analyse")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        st.metric("Racine (Double)", f"{resultats_arrondi['double']['racine']:.15f}")
                        st.caption(f"ε machine: {resultats_arrondi['double']['epsilon']:.2e}")
                    
                    with col_r2:
                        st.metric("Racine (Simple)", f"{resultats_arrondi['simple']['racine']:.15f}")
                        st.caption(f"ε machine: {resultats_arrondi['simple']['epsilon']:.2e}")
                    
                    with col_r3:
                        st.metric("Différence absolue", f"{resultats_arrondi['difference']:.2e}")
                        ratio = resultats_arrondi['difference'] / resultats_arrondi['simple']['epsilon']
                        st.caption(f"Ratio / ε_simple: {ratio:.2f}")
                    
                    # Graphiques comparatifs
                    st.markdown("---")
                    st.subheader("📈 Comparaison de convergence")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Convergence en double précision
                    df_double = pd.DataFrame(resultats_arrondi['double']['historique'])
                    ax1.semilogy(df_double['Itération'], df_double['Erreur'], 
                                'bo-', linewidth=2, markersize=6, label='Double précision')
                    
                    # Convergence en simple précision
                    df_simple = pd.DataFrame(resultats_arrondi['simple']['historique'])
                    ax1.semilogy(df_simple['Itération'], df_simple['Erreur'], 
                                'ro-', linewidth=2, markersize=6, label='Simple précision')
                    
                    ax1.axhline(y=resultats_arrondi['double']['epsilon'], 
                               color='blue', linestyle='--', alpha=0.5, label='ε double')
                    ax1.axhline(y=resultats_arrondi['simple']['epsilon'], 
                               color='red', linestyle='--', alpha=0.5, label='ε simple')
                    
                    ax1.grid(True, alpha=0.3, which='both')
                    ax1.set_xlabel('Itération', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Erreur (échelle log)', fontsize=12, fontweight='bold')
                    ax1.set_title('Convergence selon la précision', fontsize=13, fontweight='bold')
                    ax1.legend(loc='best', fontsize=10)
                    
                    # Comparaison des erreurs
                    min_len = min(len(df_double), len(df_simple))
                    iterations_comp = range(min_len)
                    diff_erreurs = [abs(df_double.iloc[i]['Erreur'] - df_simple.iloc[i]['Erreur']) 
                                   for i in iterations_comp]
                    
                    ax2.semilogy(iterations_comp, diff_erreurs, 'go-', 
                                linewidth=2, markersize=6, label='|Erreur_double - Erreur_simple|')
                    ax2.grid(True, alpha=0.3, which='both')
                    ax2.set_xlabel('Itération', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Différence d\'erreur', fontsize=12, fontweight='bold')
                    ax2.set_title('Impact de la précision', fontsize=13, fontweight='bold')
                    ax2.legend(loc='best', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tableaux détaillés
                    st.markdown("---")
                    st.subheader("📋 Historiques détaillés")
                    
                    tab1, tab2 = st.tabs(["Double précision", "Simple précision"])
                    
                    with tab1:
                        st.dataframe(df_double, use_container_width=True, height=400)
                    
                    with tab2:
                        st.dataframe(df_simple, use_container_width=True, height=400)
                    
                    # Conclusion
                    st.markdown("---")
                    st.subheader("📝 Conclusions")
                    
                    if resultats_arrondi['difference'] < resultats_arrondi['simple']['epsilon']:
                        st.success("""
                        ✅ **La différence entre les deux précisions est négligeable**
                        
                        La méthode est peu sensible aux erreurs d'arrondi pour ce problème.
                        """)
                    elif resultats_arrondi['difference'] < 1e-6:
                        st.info("""
                        ℹ️ **La différence est faible mais mesurable**
                        
                        Les erreurs d'arrondi ont un impact limité mais détectable.
                        """)
                    else:
                        st.warning("""
                        ⚠️ **La différence est significative**
                        
                        Les erreurs d'arrondi affectent notablement la précision finale.
                        La double précision est recommandée pour ce problème.
                        """)
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                st.exception(e)
    
    # Informations de la sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ À propos")
    st.sidebar.info("""
    **Méthodes implémentées:**
    - **Dichotomie**: Convergence linéaire, robuste
    - **Point Fixe**: x = g(x), convergence conditionnelle
    - **Newton-Raphson**: Convergence quadratique
    - **Sécante**: Convergence super-linéaire (φ ≈ 1.618)
    
    **Fonctionnalités:**
    - ✅ Résolution simple et comparative
    - ✅ Analyse des erreurs d'arrondi
    - ✅ Visualisations avancées
    - ✅ Export de données
    
    **Projet:** Calculs Numériques  
    **Année:** 2025
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Astuce:**")
    st.sidebar.caption("""
    Utilisez le mode *Comparaison* pour évaluer toutes les méthodes simultanément 
    et identifier la plus adaptée à votre problème.
    """)


if __name__ == "__main__":
    main()