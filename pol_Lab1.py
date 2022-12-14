import matplotlib.pyplot as plt
import numpy as np



def polr(obs,sun,pol):
	"""
	Funcion que permite calcular la polarizacion y el angulo de posicion referidos a la normal al plano de scattering (plano Sol-Objeto-Tierra).

	Usage::
	  pp,th=polr(obs,sun,pol)


	Parameters
	----------

	obs     : angulo de posicion observado respecto al punto cardinal norte en grados. [float]
=
	sun     : angulo de posicion del plano de scattering respecto al punto cardinal norte en grados. [float]

	pol     : modulo del vector de polarizacion observado en %. [float]


	Returns
	-------

	pp      : valor del vector de polarizacion reducido respecto a la normal al plano de scattering en %. [float]

	th      : angulo de posicion respecto a la normal al plano de scattering en grados. [float]
	

	Notes
	-----

	rgh - Octubre 2018
	"""
	#ojo: el valor resultante de th debería ser menor de 90...
	th=obs-(sun-90.)
	
	pp=pol*np.cos(2.*th*np.pi/180.)
	return pp,th
	
	
#def getpol_tangra(arch,faseIn=10,paso=2.5):
''' this function will allow to calculate the polarization-fase
curve from data reduced with Tangra from Laboratorio Avanzado
we assume four angles taken in order 0, 45, 90 and 135
arch-->dat archive: expected to be CSV as obtained from Tangra
faseIn--> Angulo de fase en el cual se inicia la medicion
paso--> paso en angulo de fase
OUT--> grafica, archivo csv con la fase y Pr y arreglos de fase y Pr
'''

arch=input("Archivo de datos fotometricos (CSV): ")
faseIn=input("Angulo de fase inicial: ")
faseIn=float(faseIn)
paso=input("Paso en angulo de fase: ")
paso=float(paso)
dat1=np.loadtxt(arch,delimiter=',') #datos en formato numpy
dat=dat1[:,1]
PolR=[]

ln=len(dat)  
k=0
while k<ln:
	x0=dat[k];k=k+1
	x45=dat[k];k=k+1
	x90=dat[k];k=k+1
	x135=dat[k];k=k+1
	q=((x0-x90)/(x0+x90))
	u=((x45-x135)/(x45+x135))
	P=np.sqrt(q**2+u**2) #porcentaje de pol
	A=((np.arctan2(u,q))*180/np.pi)/2 #angulo entre 0 y 180
	if A < 0: A=A+180. #NO SE REQUIERE AL CALCULAR LA BAJA
	if A >= 180: A=A-180.
	'''####FALTA CALCULAR EL ERR DE POL...
	n2=(x0+x90)**2
	eq=np.sqrt((((x0/n2)**2)*ex90**2+((x90/n2)**2)*ex0**2))
	#eq=np.sqrt(eq**2+sqs**2)*100#calculo del nuevo error q
	n2=(x45+x135)**2
	eu=np.sqrt((((x45/n2)**2)*ex135**2+((x135/n2)**2)*ex45**2))
	#eu=np.sqrt(eu**2+sus**2)*100#calculo del nuevo error u
	eP=np.sqrt(eq**2+eu**2) # Error de pol'''
	#Calculo de error del angulo
	r=u/q
	'''sigma_r=np.abs(r)*np.sqrt((eq/q)**2+(eu/u)**2)
	eA =0.5*sigma_r/(1.0+r*r)      
	eA *= 180.0/np.pi'''
	#Calcular Pr y Th 
	su=90
	Pr,Th=polr(A,su,P)
	PolR.append(Pr)

pr=np.array(PolR)
fas=np.arange(faseIn,len(pr)*paso+faseIn,paso)
np.savetxt('PolFase_'+arch,np.array([fas,pr]).T,delimiter=',')
print("Guardando archivo: "+'PolFase_'+arch)
print(np.array([fas,pr]).T)
plt.figure();plt.clf()
plt.plot(fas,pr,'.-')
plt.xlabel(r'Phase Angle [$\alpha$]');plt.ylabel(r'Pr($\alpha$)' )
plt.title('PolFase  '+arch)
plt.show()


 
#return fas,pr
