#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:48:42 2022
@author: laportem
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
import numpy as np
import os
import math




def filter_simulations(samples_df, filter_Z=10):
    """
    Filters the simulation data based on a depth error threshold.
    
    Args:
        samples_df (pd.DataFrame): DataFrame containing all sample results and errors.
        filter_Z (float): Depth error threshold (km). Keeps only rows with ErrZ < filter_Z.
    
    Returns:
        pd.DataFrame: Filtered DataFrame with depth error less than the threshold.
    """
    # Filter the rows where absolute depth error is within the threshold
    return samples_df[abs(samples_df['ErrZ']) < filter_Z]



def choose_subplot_dimensions(k):
    if k < 4:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/5), 5


def generate_subplots(k, row_wise=False):
    nrow, ncol = choose_subplot_dimensions(k)
    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(3, 5,figsize=(15,12))

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=('C' if row_wise else 'F'))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        return figure, axes
    
    
    
def gaussian_points(xo,yo):
    """ Gives the density of a population of points
    inputs : floats: xo,yo initial population
    returns : x, y, z 
    x, y : initial population sorted by density
    z : density of the population """
    x=np.array(xo)
    y=np.array(yo)
    xy = np.vstack([xo,yo])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z = [a/max(z) for a in z]
    return x,y,z



def EE_subplots(samples, samples_names, param_output, path_out, name_simu):
    """ Generate n_variables (13) scatter subplots to display the distributions of one output parameter
    (example: ErrZ, Z, La, Err_Lat...) according to all random variables (gap1....)
    Comments : - Color by density of points 
                - May be impossible to generate for a too large number of samples --> EE_subplots_imshow
    Args:
        samples (matrix n_samples*n_variables of values of random variables ) : 
        samples_names (list n_variables) : names of problem variables
        param_output (2D array) : Array of values and names of the output parameter considered
        path_out (string) path for figure output
        name_simu (string) name of the figure 
        
    Returns :  a large figure in path_out describing the distribution
    of one output (x-axis) according to all random variables (y_axis) 
    as scatter plots 
    """
    # Generate subplots based on the number of variables
    fig, axes = generate_subplots(len(samples[0])+1)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.9, wspace=0.3, hspace=0.3)
    

    n_variables = len(samples_names)
    # Generate a subplot/random variable for the param_output chosen
    for i, ax in zip(range(n_variables), axes):
        
        x, y, z = gaussian_points(samples[:, i], param_output[1])
        im = ax.scatter(x, y, c=z, s=10, cmap='jet', alpha=0.7, label=samples_names[i])
        
        ax.set_xlabel(sample_names[i], fontsize=16, fontstyle='italic')
    
    # Add color bar to the last axis
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.025, pad=0.1)
    cbar.set_label("Density", fontsize=16)
    
    # Set y-label for the entire figure
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel(param_output[0], fontsize=20, labelpad=8, fontweight='bold')
    plt.grid(False)
    
    # Save figures
    name_fig_pdf = f"F7_Distribution_per_variables_scatter_{param_output[0]}.pdf"
    name_fig_jpg = f"F7_Distribution_per_variables_scatter_{param_output[0]}.jpg"
    plt.savefig(name_fig_pdf)
    plt.savefig(name_fig_jpg)
    plt.close(fig)
    
    

def EE_subplots_imshow(samples, samples_names, param_output, path_out):
    """
    Generate n_variables (13)  imshow subplots displaying the distribution of one output parameter
    (e.g., ErrZ, Z, La, Err_Lat) against all random variables (e.g., gap1...).
    Comments : - Colors represent the density of points.
               - Use in case n_samples is too large
    Args:
        samples (matrix n_samples*n_variables of values of random variables ) : 
        samples_names (list n_variables) : names of problem variables
        param_output (2D array) : Array of values and names of the output parameter considered
        path_out (string) path for figure output
        name_simu (string) name of the figure 
        
    Returns :  a large figure in path_out describing the distribution
    of one output (x-axis) according to all random variables (y_axis) 
    as imshow plots 
    """
    fig, axes = generate_subplots(len(samples[0])+1, row_wise=True)
    y = np.array(param_output[1])
    n_variables = len(samples_names)
    for i, ax in zip(range(n_variables), axes):
        x = np.array(samples[:, i])
        bounds = [[x.min(), x.max()], [y.min(), y.max()]]
        
        # Compute 2D histogram for density visualization
        im, x_edges, y_edges = np.histogram2d(x, y, range=bounds, bins=50)
        bounds2 = (x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max())
        
        # Display density using imshow
        ax.imshow(im.T[::-1, :], extent=bounds2, cmap='jet', interpolation='gaussian')
        ax.set_aspect('auto')
        ax.grid(linewidth=0.1, linestyle=':')
        ax.set_xlabel(samples_names[i], fontsize=16)
        ax.tick_params(labelsize=12)
    
    # Add density histogram on the final subplot
    lim0, lim1 = min(y), max(y)
    bins = np.linspace(lim0, lim1, 21)
    axes[-1].hist(y, bins=bins, color='steelblue', density=True)
    axes[-1].set_xlabel(param_output[0], fontsize=16)
    axes[-1].set_ylabel('Density of solutions')
    
    # General plot adjustments
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel(param_output[0], fontsize=20)
    fig.tight_layout()
    plt.grid(False)
    
    # Save the figure
    name_fig = os.path.join(path_out, f"F7_Distribution_per_variables_imshow_{param_output[0]}.pdf")
    plt.savefig(name_fig)
    
    
def EE_hist(samples,samples_names, param_output, path_out):
    """ Plot histograms of all random variables distributions (should correspond to 
    their distribution set in the problem)  
    if not : add more samples 
    In addition plot histogram of one param_output (example : Z)
    Args:
        samples (matrix n_samples*n_variables of values of random variables ) : 
        samples_names (list n_variables) : names of problem variables
        param_output (2D array) : Array of values and names of the output parameter considered
        path_out (string) path for figure output
        name_simu (string) name of the figure 
    Returns : A figure to verify the random variables distribution """
    
    fig, axes = generate_subplots(len(samples[0])+1, row_wise=True)
    
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.3,hspace=0.3)
    n_variables = len(samples_names)
    for i, ax in zip(range(n_variables+1), axes):
        #ax = fig.add_subplot(3,5,i)
        ax = axes[i]
        
        if i<=n_variables:
            x = samples[:,i]
            lim0=min(x)
            lim1=max(x)
            bins = np.linspace(lim0, lim1, 21)
            ax.hist(x, bins=bins , color='steelblue',)
            ax.set_xlabel(sample_names[i])
            ax.set_ylabel('Probability distribution')

        else :
            z = param_output[1]
            binsz=np.linspace(-20,20,41)
            ax.hist(z, bins=binsz, color='salmon')
            
            ax.set_xlabel(param_output[0])
            ax.set_ylabel('Probability distribution')
    name_fig = 'Histograms_distributions'+str(param_output[0])+'.pdf'
    plt.savefig(name_fig)