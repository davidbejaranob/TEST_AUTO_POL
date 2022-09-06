#!~/anaconda3/envs/MIO_env/bin/python

###################--------------------------------------
#Piece of code for fitting HG Polarization phase curves from asteroids data
#User:Lab Avanzado
#
#The formulation is based on:
#http://gcpsj.sdf-eu.org/catalogo.html
#
#
#By: JHCCH 12-nov-2021
##########################-----------------------------
import numpy as np
import matplotlib.pyplot as plt
from lmfit.model import Model, save_model
from scipy.interpolate import interp1d
from scipy.stats import chisquare
#import time
import glob

def modelFase(A,c1,c2,c3):
   '''model to ajust Polarization fase curves in asteroids
   A--> phase angle
   c1--> paramater 1
   c2--> parameter 2
   c3--> parameter 3
   OUT --> Pr(alpha), reduced Polarization'''
   
   Pr=c1*(np.exp(-(A/c2))-1)+c3*A
   
   return(Pr)

def P_fit(alpha,magPr):
   '''Fitting function for modelFase 
   alpha-->observed phase
   magVr--> observed Pr'''
   X=(alpha,alpha)
   model = Model(modelFase)
   model.set_param_hint('c1', value=0.1,min=0,max=15)
   model.set_param_hint('c2',value=0.1,min=0,max=15)
   model.set_param_hint('c3',value=0.1,min=0,max=15)
   
   params = model.make_params()
   # Fitting
   model_fit = model.fit(magPr, params,
               A=X,verbose=True,max_nfev=10000)

   return model_fit

#Cargar datos 
#arch='a21900.txt'
#lista=glob.glob('PolFase_try2.csv')
arch=input("Archivo con datos de polarimetria (CSV): ")
plt.figure();plt.clf()
#co=0
#color = iter(plt.cm.Paired(np.linspace(0, 1, len(lista)+1)))

dato=np.loadtxt(arch,delimiter=',')
k0=dato[dato[:,0].argsort()] 
al=k0[:,0];va=k0[:,1]
funF=interp1d(al,va,kind='linear')
al0=np.arange(al[0],al[-1],.1)
Mag0=funF(al0)
LC_fit=P_fit(al0,Mag0)
al1=np.arange(0,35,.1)
Fit=LC_fit.eval(X=(al,al0))
#print parameters
print(arch+':');print(LC_fit.params);print(':\n\n')
c1= LC_fit.params['c1'].value
c2= LC_fit.params['c2'].value
c3= LC_fit.params['c3'].value
mod=modelFase(al1,c1,c2,c3)
#co=next(color)
plt.plot(al1,mod,'--',label='Model')
plt.plot(al,va,'*',label=arch)
plt.plot(al0,Mag0,'-.',label='interp')
plt.ylim(-1,1)
plt.xlim(0,35)
#plt.plot(Y/v,F0,'-k',label='Synthetic_optimize')
plt.xlabel(r'Phase Angle [$\alpha$]');plt.ylabel(r'Pr($\alpha$)' )
plt.legend()

#invertir ejes
#ax = plt.gca()
#ax.set_ylim(ax.get_ylim()[::-1])
plt.show()
