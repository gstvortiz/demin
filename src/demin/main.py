import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from itertools import combinations
import pyomo.environ as pyo
import periodictable

class MiningVisualizer():
    def __init__(self, data, subset = ['X', 'Y', 'Z'], figsize = 8, fontsize = 15, s = 30, elev = 30, azim = -75, labelpad = 10, cmap = 'turbo', colorbar = True):
        self.data = data.dropna(subset = subset)
        self.subset = subset
        self.figsize = figsize
        self.fontsize = fontsize
        self.labelpad = labelpad
        self.s = s
        self.elev = elev
        self.azim = azim
        self.cmap = cmap
        self.colorbar = colorbar

    def Plot(self, var = None, var_kind = None):
        fig = plt.figure(figsize=(self.figsize, self.figsize))
        ax = fig.add_subplot(projection='3d')
        ax.set_title(var, fontsize = self.fontsize, pad = self.labelpad)
        ax.set_xlabel(self.subset[0], fontsize = self.fontsize, labelpad = self.labelpad)
        ax.set_ylabel(self.subset[1], fontsize = self.fontsize, labelpad = self.labelpad)
        ax.set_zlabel(self.subset[2], fontsize = self.fontsize, labelpad = self.labelpad)
        ax.view_init(elev = self.elev, azim = self.azim)

        if var_kind == 'Discreta':
            scatter_list = []
            data = self.data.dropna(subset = var)
            grupos = sorted(data[var].unique())
            colors = sns.color_palette(self.cmap, n_colors=len(grupos))
            for i, grupo in enumerate(grupos):
                grupo_data = data[data[var] == grupo]
                scatter = ax.scatter(data=grupo_data, xs = self.subset[0], ys = self.subset[1], zs = self.subset[2], s = self.s, color = colors[i], depthshade=False, label = grupo)
                scatter_list.append(scatter)
            ax.legend(fontsize=self.fontsize*0.75, markerscale=2)

        elif var_kind == 'Contínua':    
            data = self.data.dropna(subset = var)
            scatter = ax.scatter(data = data, xs = self.subset[0], ys = self.subset[1], zs = self.subset[2], c = var, cmap = self.cmap, s = self.s, depthshade=False)
            if self.colorbar:
                plt.subplots_adjust(left = 0, right = 0.8, bottom = 0, top = 1)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.6])
                cbar = fig.colorbar(scatter, cax = cbar_ax)
                
        else:
            scatter = ax.scatter(data = self.data, xs = self.subset[0], ys = self.subset[1], zs = self.subset[2], s = self.s, depthshade=False)
            ax.set_title(None)
        return fig, ax
    
    def Expand(self, hue):
        fig, axs = plt.subplots(1, 3, figsize = (20, 6))
        fig.suptitle(f'Geospatial Visualization: {hue}')
        combinacoes = list(combinations(self.subset, 2))
        for i, (x, y) in enumerate(combinacoes):
            sns.scatterplot(data = self.data, x = x, y = y, hue = hue, palette='turbo', s = 40, ax = axs.ravel()[i])

class MineralogicalConversion:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = None

    def MSE(self, model):
        return sum((model.y[i] - sum(model.X[i, j] * model.w[j] for j in model.J))**2 for i in model.I)
    
    def solve(self, bounds = [0, 1], tee = False):
        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(self.X.shape[0])
        model.J = pyo.RangeSet(self.X.shape[1])
        model.X = pyo.Param(model.I, model.J, initialize = lambda model, i, j: self.X.iloc[i-1, j-1])
        model.w = pyo.Var(model.J, within = pyo.NonNegativeReals, bounds = bounds)
        model.y = pyo.Param(model.I, initialize = lambda model, i: self.y.iloc[i-1])
        model.obj = pyo.Objective(rule = self.MSE, sense = pyo.minimize)
        pyo.SolverFactory('ipopt', executable = 'ipopt').solve(model, tee = tee)
        self.w = np.array([model.w[j]() for j in model.J])

    def summary(self):
        frx_df = pd.DataFrame([self.y, self.X.dot(self.w)], index = ['FRX_true', 'FRX_pred']).round(2)
        frx_df['[SUM]'] = frx_df.sum(axis = 1).round(2)
        mineral_df = pd.DataFrame(self.w, index = self.X.columns, columns = ['Predicted(%)']).rename_axis('Mineral').round(3).T * 100
        mineral_df['[SUM]'] = mineral_df.sum(axis = 1).round(2)
        display(frx_df, mineral_df)
        return frx_df, mineral_df

def mass_percentage_distribution(chemical_composition):
    elements = {}
    total_mass = 0
    # Calculate the total mass and mass of each element
    for element, quantity in chemical_composition.items():
        element_obj = getattr(periodictable, element)
        element_mass = element_obj.mass * quantity
        elements[element] = element_mass
        total_mass += element_mass
    # Calculate the mass percentage distribution for each element
    percentage_distribution = {element: (mass / total_mass) * 100 for element, mass in elements.items()}
    return percentage_distribution

def calculate_compositions(compositions):
    data = {}
    for composition_name, composition in compositions.items():
        data[composition_name] = mass_percentage_distribution(composition)
    df = pd.DataFrame(data).fillna(0).round(2).rename_axis('Elements')    
    return df.sort_index()[sorted(data.keys())]

def Fragmentate(df, compositions):
    result_df = calculate_compositions(compositions)
    for index, row in result_df.iterrows():
        df[index] = (df[row.index] * row / 100).sum(axis = 1)
    return df[sorted(df.columns)].drop(result_df, axis = 1)