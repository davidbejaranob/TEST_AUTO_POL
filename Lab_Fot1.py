from importlib.resources import path
from tabnanny import check
import astropy.io.fits as ft
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation as ipl
import scipy.ndimage as nd
import os
import subprocess
import glob
from clint.textui import colored
clear = lambda: os.system("cls")  # On Windows System


#############################################################
#
# funcion que permite lee una imagen FITS y su cabecera
# desde su archivo en disco
#
#
def leefits(arch, uint=True, plano=0):
    """
    Funcion que permite leer una imagen FITS y su cabecera desde su archivo y asignarla a objetos en memoria. Si la imagen es multidimensional se puede indicar cual es el plano que interesa leer.

    Usage::
      img,hdr=leefits(arch,uint=True,plano=0)


    Parameters
    ----------

    arch     : Archivo a leer. (string)

    uint     : Si la cabecera tiene los keywords BSCALE y BZERO puede considerar la imagen como de enteros sin signo. El default es uint=True (bool)

    plano    : En un cubo multidimensional indica cual es el plano a leer. El default es plano=0. (int)


    Returns
    -------

    img      : El array con la imagen.

    hdr      : La cabecera.


    Notes
    -----

    rgh - Ago 2016
    """
    f = ft.open(arch, uint=uint)

    # esto es para evitar problemas con keywords no standard en la cabecera
    #
    f.verify("silentfix")

    img = f[0].data
    hdr = f[0].header
    f.close()
    return img, hdr


#############################################################
#
# calculo del total de ADUs en una apertura circular considerando
# la contribucion en el borde por pixels fraccionales
#
#
#
def aper(img, coo, cr):
    """
    Calcula el numero total de ADUs de un objeto en una apertura circular considerando la contribucion en el borde de la apertura por pixels fraccionales. Si el pixel esta en el borde, el calculo estima el brillo en forma proporcional al area exacta del pixel dentro de la apertura.

    Usage::
      val=aper(img,coo,cr)


    Parameters
    ----------

    img     : imagen donde esta el objeto. [array]

    coo     : tuple con coordenadas Y y X del objeto. Ojo con el orden.  [int o float]

    cr      : radio de la apertura. [float]


    Returns
    -------

    val     : vector con el radio, el area total, el total de ADUs en la apertura y la desviacion standard asumiendo una distr. de Poisson. [float]


    Notes
    -----

    rgh - Marzo 2016
    """

    # dimensiones de la imagen, lado del box de extraccion y control de
    # parametros ingresados
    dim = np.shape(img)
    box = int(cr + 0.5) + 1

    assert cr >= 1.0, "Radio de apertura < 1. Aborta."
    assert (coo[0] >= 0.0) & (coo[1] >= 0.0), "Coord. del objeto incorrectas. Aborta"
    assert (
        (coo[0] >= box)
        & (coo[1] >= box)
        & (dim[0] - 1 - coo[0] >= box)
        & (dim[1] - 1 - coo[1] >= box)
    ), "Objeto muy cerca del borde. Aborta."

    # hace un shift para centrar el objeto en un pixel entero
    dyx = (int(coo[0]) - coo[0], int(coo[1]) - coo[1])
    img1 = nd.shift(img, dyx)

    # seccion de la imagen con el objeto. Las coord. ingresadas
    # estan en el centro de la seccion
    sec = img1[
        int(coo[0]) - box : int(coo[0]) + box + 1,
        int(coo[1]) - box : int(coo[1]) + box + 1,
    ]

    # arma una mascara con el peso del pixel dentro
    # de la apertura
    mask = np.ones((2 * box + 1, 2 * box + 1), dtype=float)

    for i in np.arange(2 * box + 1):
        for j in np.arange(2 * box + 1):

            # distancia desde el centro a los vertices del pixel
            r1 = np.sqrt((i + 0.5 - box) ** 2 + (j + 0.5 - box) ** 2)
            r2 = np.sqrt((i - 0.5 - box) ** 2 + (j + 0.5 - box) ** 2)
            r3 = np.sqrt((i - 0.5 - box) ** 2 + (j - 0.5 - box) ** 2)
            r4 = np.sqrt((i + 0.5 - box) ** 2 + (j - 0.5 - box) ** 2)

            if (r1 > cr) | (r2 > cr) | (r3 > cr) | (r4 > cr):
                # los cuatro vertices estan fuera de la apertura
                if (r1 > cr) & (r2 > cr) & (r3 > cr) & (r4 > cr):
                    mask[j, i] = 0.0
                else:
                    # los cuatro posibles puntos de corte con los
                    # lados del pixel
                    pyi = cr**2 - (i - 0.5 - box) ** 2
                    pyd = cr**2 - (i + 0.5 - box) ** 2
                    pxu = cr**2 - (j - 0.5 - box) ** 2
                    pxd = cr**2 - (j + 0.5 - box) ** 2

                    # marca como negativos los que no corresponden
                    if pyi > 0:
                        ppp = -np.sqrt(pyi) + box
                        pyi = np.sqrt(pyi) + box
                        if np.abs(pyi - j) > 0.5:
                            pyi = ppp
                            if np.abs(pyi - j) > 0.5:
                                pyi = -1.0
                    if pyd > 0:
                        ppp = -np.sqrt(pyd) + box
                        pyd = np.sqrt(pyd) + box
                        if np.abs(pyd - j) > 0.5:
                            pyd = ppp
                            if np.abs(pyd - j) > 0.5:
                                pyd = -1.0
                    if pxu > 0:
                        ppp = -np.sqrt(pxu) + box
                        pxu = np.sqrt(pxu) + box
                        if np.abs(pxu - i) > 0.5:
                            pxu = ppp
                            if np.abs(pxu - i) > 0.5:
                                pxu = -1.0
                    if pxd > 0:
                        ppp = -np.sqrt(pxd) + box
                        pxd = np.sqrt(pxd) + box
                        if np.abs(pxd - i) > 0.5:
                            pxd = ppp
                            if np.abs(pxd - i) > 0.5:
                                pxd = -1.0

                    # para cada posible caso calcula el area del sector
                    # circular y del pixel que esta dentro de la apertura

                    # corta a izquierda y derecha
                    if (pyd > 0) & (pyi > 0):
                        alp = np.sqrt((pyd - pyi) ** 2 + 1.0)
                        areac = (
                            np.arcsin(alp / 2.0 / cr) * cr**2
                            - alp * np.sqrt(cr**2 - (alp / 2.0) ** 2) / 2.0
                        )
                        if pyi < pyd:
                            area = (pyd - pyi) / 2.0 + (j + 0.5 - pyd)
                        else:
                            area = (pyi - pyd) / 2.0 + (j + 0.5 - pyi)
                        if r1 <= cr:
                            mask[j, i] = area + areac
                        else:
                            mask[j, i] = 1.0 - area + areac

                    # corta arriba y abajo
                    elif (pxu > 0) & (pxd > 0):
                        alp = np.sqrt((pxd - pxu) ** 2 + 1.0)
                        areac = (
                            np.arcsin(alp / 2.0 / cr) * cr**2
                            - alp * np.sqrt(cr**2 - (alp / 2.0) ** 2) / 2.0
                        )
                        if pxu < pxd:
                            area = (pxd - pxu) / 2.0 + (i + 0.5 - pxd)
                        else:
                            area = (pxu - pxd) / 2.0 + (i + 0.5 - pxu)
                        if r1 <= cr:
                            mask[j, i] = area + areac
                        else:
                            mask[j, i] = 1.0 - area + areac

                    # corta a izquierda y arriba
                    elif (pyi > 0) & (pxu > 0):
                        alp = np.sqrt((i - 0.5 - pxu) ** 2 + (j - 0.5 - pyi) ** 2)
                        areac = (
                            np.arcsin(alp / 2.0 / cr) * cr**2
                            - alp * np.sqrt(cr**2 - (alp / 2.0) ** 2) / 2.0
                        )
                        area = (
                            (pxu - (i - 0.5)) * (pyi - (j - 0.5)) / 2.0
                            + (pxu - (i - 0.5)) * (j + 0.5 - pyi)
                            + (i + 0.5 - pxu)
                        )
                        if r1 <= cr:
                            mask[j, i] = area + areac
                        else:
                            mask[j, i] = 1.0 - area + areac

                    # corta a izquierda y abajo
                    elif (pyi > 0) & (pxd > 0):
                        alp = np.sqrt((i - 0.5 - pxd) ** 2 + (j + 0.5 - pyi) ** 2)
                        areac = (
                            np.arcsin(alp / 2.0 / cr) * cr**2
                            - alp * np.sqrt(cr**2 - (alp / 2.0) ** 2) / 2.0
                        )
                        area = (
                            (pxd - (i - 0.5)) * (j + 0.5 - pyi) / 2.0
                            + (pxd - (i - 0.5)) * (pyi - (j - 0.5))
                            + (i + 0.5 - pxd)
                        )
                        if r4 <= cr:
                            mask[j, i] = area + areac
                        else:
                            mask[j, i] = 1.0 - area + areac

                    # corta a derecha y arriba
                    elif (pyd > 0) & (pxu > 0):
                        alp = np.sqrt((i + 0.5 - pxu) ** 2 + (j - 0.5 - pyd) ** 2)
                        areac = (
                            np.arcsin(alp / 2.0 / cr) * cr**2
                            - alp * np.sqrt(cr**2 - (alp / 2.0) ** 2) / 2.0
                        )
                        area = (
                            (i + 0.5 - pxu) * (pyd - (j - 0.5)) / 2.0
                            + (i + 0.5 - pxu) * (j + 0.5 - pyd)
                            + (pxu - (i - 0.5))
                        )
                        if r2 <= cr:
                            mask[j, i] = area + areac
                        else:
                            mask[j, i] = 1.0 - area + areac

                    # corta a derecha y abajo
                    elif (pyd > 0) & (pxd > 0):
                        alp = np.sqrt((i + 0.5 - pxd) ** 2 + (j + 0.5 - pyd) ** 2)
                        areac = (
                            np.arcsin(alp / 2.0 / cr) * cr**2
                            - alp * np.sqrt(cr**2 - (alp / 2.0) ** 2) / 2.0
                        )
                        area = (
                            (i + 0.5 - pxd) * (j + 0.5 - pyd) / 2.0
                            + (i + 0.5 - pxd) * (pyd - (j - 0.5))
                            + (pxd - (i - 0.5))
                        )
                        if r3 <= cr:
                            mask[j, i] = area + areac
                        else:
                            mask[j, i] = 1.0 - area + areac

    val = [cr, np.sum(mask), np.sum(mask * sec), np.sqrt(np.sum(mask * sec))]
    return val


#############################################################
#
# fotometria precisa en una serie de aperturas
#
#
#
def fot(
    img,
    obj,
    coo,
    rr,
    ri,
    re,
    tint=1.0,
    cte=22.0,
    gain=1.0,
    rdnoise=1.0,
    ring=True,
    modo="mean",
    fon=False,
):
    """
    Calcula el flujo total en e- y la magnitud de un objeto en una serie de aperturas circulares considerando la contribucion en el borde de la apertura por pixels fraccionales.

    Usage::
      val,sky=fot(img,obj,coo,rr,ri,re,tint=1.,cte=22.,gain=1.,rdnoise=1.,ring=True,modo='mean',fon=False)


    Parameters
    ----------

    img     : imagen donde esta el objeto. [array]

    obj     : numero de identificacion del objeto. [int]

    coo     : tuple con coordenadas Y y X del objeto. Ojo con el orden.  [int o float]

    rr      : tuple con el listado de aperturas. [float]

    ri      : radio interno de la apertura. radi tiene que ser menor que re. [float]

    re      : radio externo de la apertura. rade tiene que ser mayor que ri. [float]

    tint    : tiempo de integracion utilizado en seg. El default es tint=1 [float]

    cte     : constante para el calculo de la magnitud. El default es cte=22.0 [float]

    gain    : ganancia del detector. El default es gain=1 [float]

    rdnoise : ruido de lectura del detector. El default es rdnoise=1 [float]

    ring    : flag que indica si en la estimacion del cielo considera todo el anillo y desprecia pixeles altos (True), o hace un calculo tomando octavos del anillo. El default es ring=True. [bool]

    modo    : indica si en la estimacion del cielo usa la mediana ('median') o el valor medio ('mean'). El default es modo='mean'. [string]

    fon     : si es True utiliza el valor en re como valor medio del fondo de cielo. El default es fon=False. [bool]


    Returns
    -------

    val     : lista donde cada elemento es un vector que contiene:
                0)	la identificacion del objeto,
                1) el radio de la apertura,
                2) el area total,
                3) el flujo neto,
                4) la magnitud,
                5) su error
                6) y la relacion S/N.
        [todos float menos el nro. de identificacion que es int]

    sky     : vector que contiene:
                0) el area del anillo donde se estimo el cielo,
                1) el flujo total,
                2) el valor del flujo de cielo por pixel,
                3)y los radios interno
                4) y externo utilizados para el anillo.
        Si se utilizo un valor constante para el cielo ambos radios son cero. [float]


    Notes
    -----

    rgh - Abril 2018
    rgh - Agosto 2018: Modificado para que cada elemento de VAL sea una lista y no un string
    rgh - Octubre 2018: modificado para permitir diferentes formas de estimar el cielo (modificacion de la funcion ANILLO).
    rgh - Noviembre 2018: Modificado para permitir utilizar un valor constante para el cielo.
    """

    # dimensiones de la imagen, lado del box de extraccion y control de
    # parametros ingresados
    dim = np.shape(img)
    box = re + 1

    assert (ri >= 1.0) & (re > 1.0), "Radio de anillo para el cielo < 1. Aborta."
    assert ri < re, "Radio exterior es menor que el interior. Aborta."
    assert ri + 1 < re, "Seccion minima de 2 pix para el anillo. Aborta."
    assert np.max(rr) <= ri, "Aperturas muy grandes para ese anillo. Aborta."
    assert np.min(rr) >= 1.0, "Aperturas muy peque??as (< 1 pix). Aborta."
    assert (coo[0] >= 0.0) & (coo[1] >= 0.0), "Coord. del objeto incorrectas. Aborta"
    assert (
        (coo[0] >= box)
        & (coo[1] >= box)
        & (dim[0] - 1 - coo[0] >= box)
        & (dim[1] - 1 - coo[1] >= box)
    ), "Objeto muy cerca del borde. Aborta."
    assert tint > 0.0, "Tiempo de integracion debe ser positivo. Aborta."
    assert gain > 0.0, "Ganancia debe ser positiva. Aborta."
    assert rdnoise > 0.0, "Read-noise debe ser positivo. Aborta."

    fotom = []

    # estima valores para el cielo utilizando el valor medio o la mediana.
    #
    if fon:
        sky = re * gain
    else:
        ss = anillo(img, coo, ri, re, full=ring)
        if modo == "median":
            sky = ss[3] * gain
        else:
            sky = ss[2] * gain

    # bucle para cada apertura donde calcula la magnitud, el
    # error y la relacion S/N.
    for cr in rr:
        val = aper(img, coo, cr)
        flx = val[2] * gain - val[1] * sky
        if flx > 0.0:
            mag = cte - 2.5 * np.log10(flx / tint)
            sn = flx / np.sqrt(
                flx + val[1] * (1.0 + 1.0 / ss[0]) * (sky + rdnoise**2)
            )
            emag = 2.5 * np.log10(np.e) / sn
        else:
            mag = 99.0
            sn = 0.0
            emag = 9.999

        fin = [obj, cr, val[1], flx, mag, emag, sn]
        fotom.append(fin)

    if fon:
        return fotom, [1.0, sky, sky, 0.0, 0.0]
    else:
        return fotom, [ss[0], ss[1] * gain, sky, ri, re]


# Funcion para seleccionar N posiciones de objeto
#
class mou_clickN:
    def __init__(self, img, Zsc=True, N=3):
        """img--> matriz de datos de la imagen
        Zsc--> bandera para mejorar contraste tipo ZSCALE como DS9
             N--> Numero de Clicks en la imagen antes de cerrar
             OUT-->coords, pares de numeros que representan la coordenada extraida"""
        # Simple mouse click function to store coordinates
        def onclickN(event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            # assign global variable to access outside of function
            global coords
            coords.append((ix, iy))

            # Disconnect after 2 clicks
            if len(coords) == N:
                fig.canvas.mpl_disconnect(cid)
                plt.close(1)
            return

        print("1)Click sobre objeto\n ")
        fig = plt.figure(1)
        # ax = fig.add_subplot(111)
        # ax.imshow(img**0.5)
        if Zsc == True:
            a, b = zscale(img)
        else:
            a = np.min(img)
            b = np.max(img)
        plt.imshow(img, vmin=a, vmax=b)  # mejorar el contraste
        # plt.Circle(co,ro)
        # plt.Circle(cf,rf)
        # plt.gray()

        # Call click func
        global coords
        coords = []
        cid = fig.canvas.mpl_connect("button_press_event", onclickN)
        plt.show(block=True)  # Bloquear prompt en la imagen para interactuar

        self.xy = coords


#############################################################
#
# funcion para encontrar el baricentro fotometrico de la
# imagen de un objeto utilizando promedios pesados en una seccion
# de la imagen centrada en el objeto.
#
#
def centro(img, coo, box=9, dmax=1, sat=62000, mod="margen"):
    """
    Encuentra el baricentro fotometrico de una imagen a partir de coordenadas aproximadas usando distribuciones marginales en lineas y columnas pesados con el flujo, o procesos morfologicos.
    La funcion no controla si la caja de busqueda esta completamente dentro de la imagen y es posible que en ciertos casos la caja varie sus dimensiones para no sobrepasar los bordes.
    En la salida se dan las coordenadas (Y, X) del baricentro fotometrico, las coordenadas (X, Y) para ds9, y flags de distancia respecto a las coord. iniciales y saturacion.

    Usage::
      cen=centro(img,coo,box=7,dmax=1,sat=62000,mod='margen')

    Parameters
    ----------

    img    : imagen o array a procesar.  [float o int]

    coo    : tuple con coordenadas (Y, X) aproximadas del centro de la imagen. Ojo con el orden. [float o int]

    box    : lado de la caja de busqueda. El default es box=9. [int]

    dmax   : numero maximo de pixels entre las posiciones inicial y final. El defaul es dmax=1. [float o int]

    sat    : nivel de saturacion. El default es sat=62000. [int]

    mod    : modo para determinar el baricentro. Puede ser 'margen' o 'morf'para metodos de sumas marginales o procesos morfologicos, respectivamente. El default es mod='margen'. [str]

    Returns
    -------

    cen    : lista con las coordenadas (Y, X) del baricentro fotometrico (en ese orden, float), las coordenadas (X,Y) para ds9 (se le suma 1px, float), un string que indica si la diferencia entre las coord. iniciales y finales es mayor que dmax ('Ok' o 'Err'), y un flag que indica si la fuente es util o esta saturada ('Ok' o 'Sat').

    Notes
    -----

    rgh - Ago 2015
    rgh - Abr 2016: Modifico calculo del baricentro fotometrico.
    rgh - Nov 2017: Python 3.5
    rgh - Abr 2018: opcion de calculo mediante procesos morfologicos.
    rgh - Sep 2018: las sumas marginales las centro en pixel de mayor valor.
    """

    # asegura que las coordenadas aproximadas sean enteras
    #
    xa = int(coo[1] + 0.5)
    ya = int(coo[0] + 0.5)

    # box entero e impar
    #
    box = int(box + 0.5)
    box = (box // 2) * 2 + 1

    # dimensiones de la imagen y area de la caja de busqueda
    #
    dim = np.shape(img)

    x0 = max(xa - box // 2, 0)
    x1 = min(xa + box // 2 + 1, dim[1])
    y0 = max(ya - box // 2, 0)
    y1 = min(ya + box // 2 + 1, dim[0])

    sec = np.copy(img[y0:y1, x0:x1])

    # resto valor minimo dentro de la caja de busqueda
    #
    sec -= np.min(sec)

    if mod == "margen":

        # asume como centro el pixel con el mayor valor
        # en la caja de busqueda
        #
        dim0 = np.shape(sec)
        cen = np.argmax(sec)
        yc = cen // dim0[0] + y0
        xc = cen % dim0[1] + x0

        # hace una nueva busqueda centrando la caja en el
        # pixel encontrado con mayor valor
        #
        x0 = max(xc - box // 2, 0)
        x1 = min(xc + box // 2 + 1, dim[1])
        y0 = max(yc - box // 2, 0)
        y1 = min(yc + box // 2 + 1, dim[0])

        sec = np.copy(img[y0:y1, x0:x1])

        # calcula las sumas marginales dentro de la
        # caja de busqueda
        #
        yy, xx = np.indices(sec.shape)
        yc = np.sum(sec * yy) / np.sum(sec) + y0
        xc = np.sum(sec * xx) / np.sum(sec) + x0

    else:

        # define el objeto mediante procedimientos morfologicos
        # y luego determina el baricentro
        #
        thr = img[ya, xa] / 3.0
        sec1, nn = nd.label(sec > thr, structure=np.ones((3, 3)))
        cen = nd.center_of_mass(sec, sec1, [1])
        yc = cen[0][0] + y0
        xc = cen[0][1] + x0

    # flag de distancia respecto del centro del box
    #
    if np.sqrt((xa - xc) ** 2 + (ya - yc) ** 2) < dmax:
        flg = "Ok"
    else:
        flg = "Err"

    # flag de saturacion
    #
    if img[int(yc + 0.5), int(xc + 0.5)] < sat:
        flg1 = "Ok"
    else:
        flg1 = "Sat"

    return [(yc, xc), (xc + 1, yc + 1), flg, flg1]


##################################################################
MAX_REJECT = 0.5
MIN_NPIXELS = 5
GOOD_PIXEL = 0
BAD_PIXEL = 1
KREJ = 2.5
MAX_ITERATIONS = 5


def zscale(image, nsamples=1000, contrast=0.25):
    """Implement IRAF zscale algorithm
    nsamples=1000 and contrast=0.25 are the IRAF display task defaults
    image is a 2-d numpy array
    returns (z1, z2)
    """

    # Sample the image
    samples = zsc_sample(image, nsamples)

    return zscale_samples(samples, contrast=contrast)


####################################################################
def zsc_sample(image, maxpix, bpmask=None, zmask=None):

    # Figure out which pixels to use for the zscale algorithm
    # Returns the 1-d array samples
    # Don't worry about the bad pixel mask or zmask for the moment
    # Sample in a square grid, and return the first maxpix in the sample
    nc = image.shape[0]
    nl = image.shape[1]
    stride = max(1.0, np.sqrt((nc - 1) * (nl - 1) / float(maxpix)))
    stride = int(stride)
    samples = image[::stride, ::stride].flatten()
    # remove NaN and Inf
    samples = samples[np.isfinite(samples)]
    return samples[:maxpix]


#####################################################################
def zscale_samples(samples, contrast=0.25):
    samples = np.asarray(samples, dtype=float)
    npix = len(samples)
    samples.sort()
    zmin = samples[0]
    zmax = samples[-1]
    # For a zero-indexed array
    center_pixel = int((npix - 1) // 2)
    if npix % 2 == 1:
        median = samples[center_pixel]
    else:
        median = 0.5 * (samples[center_pixel] + samples[center_pixel + 1])

    #
    # Fit a line to the sorted array of samples
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    ngrow = max(1, int(npix * 0.01))
    ngoodpix, zstart, zslope = zsc_fit_line(samples, npix, KREJ, ngrow, MAX_ITERATIONS)
    # print "slope=%f intercept=%f" % (zslope, zstart)

    if ngoodpix < minpix:
        z1 = zmin
        z2 = zmax
    else:
        if contrast > 0:
            zslope = zslope / contrast
        z1 = max(zmin, median - (center_pixel - 1) * zslope)
        z2 = min(zmax, median + (npix - center_pixel) * zslope)
    return z1, z2


###################################################################
def zsc_fit_line(samples, npix, krej, ngrow, maxiter):
    if npix <= 1:
        return npix, 0, 1

    #
    # First re-map indices from -1.0 to 1.0
    xscale = 2.0 / (npix - 1)
    xnorm = np.arange(npix)
    xnorm = xnorm * xscale - 1.0

    ngoodpix = npix
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    last_ngoodpix = npix + 1

    # This is the mask used in k-sigma clipping.  0 is good, 1 is bad
    badpix = np.zeros(npix, dtype=int)

    #
    #  Iterate

    for niter in range(maxiter):

        if (ngoodpix >= last_ngoodpix) or (ngoodpix < minpix):
            break

        # Accumulate sums to calculate straight line fit
        goodpixels = np.where(badpix == GOOD_PIXEL)
        sumx = xnorm[goodpixels].sum()
        sumxx = (xnorm[goodpixels] * xnorm[goodpixels]).sum()
        sumxy = (xnorm[goodpixels] * samples[goodpixels]).sum()
        sumy = samples[goodpixels].sum()
        sum = len(goodpixels[0])

        delta = sum * sumxx - sumx * sumx
        # Slope and intercept
        intercept = (sumxx * sumy - sumx * sumxy) / delta
        slope = (sum * sumxy - sumx * sumy) / delta

        # Subtract fitted line from the data array
        fitted = xnorm * slope + intercept
        flat = samples - fitted

        # Compute the k-sigma rejection threshold
        ngoodpix, mean, sigma = zsc_compute_sigma(flat, badpix, npix)

        threshold = sigma * krej

        # Detect and reject pixels further than k*sigma from the fitted line
        lcut = -threshold
        hcut = threshold
        below = np.where(flat < lcut)  # else:
        above = np.where(flat > hcut)

        badpix[below] = BAD_PIXEL
        badpix[above] = BAD_PIXEL

        # Convolve with a kernel of length ngrow
        kernel = np.ones(ngrow, dtype=int)
        badpix = np.convolve(badpix, kernel, mode="same")

        ngoodpix = len(np.where(badpix == GOOD_PIXEL)[0])

        niter += 1

    # Transform the line coefficients back to the X range [0:npix-1]
    zstart = intercept - slope
    zslope = slope * xscale

    return ngoodpix, zstart, zslope


#####################################################################
def zsc_compute_sigma(flat, badpix, npix):

    # Compute the rms deviation from the mean of a flattened array.
    # Ignore rejected pixels

    # Accumulate sum and sum of squares
    goodpixels = np.where(badpix == GOOD_PIXEL)
    sumz = flat[goodpixels].sum()
    sumsq = (flat[goodpixels] * flat[goodpixels]).sum()
    ngoodpix = len(goodpixels[0])
    if ngoodpix == 0:
        mean = None
        sigma = None
    elif ngoodpix == 1:
        mean = sumz
        sigma = None
    else:
        mean = sumz / ngoodpix
        temp = sumsq / (ngoodpix - 1) - sumz * sumz / (ngoodpix * (ngoodpix - 1))
        if temp < 0:
            sigma = 0.0
        else:
            sigma = np.sqrt(temp)

    return ngoodpix, mean, sigma


#############################################################
#
# calculo del total de ADUs y sus parametros estadisticos
# en una apertura anular
#
#
#
def anillo(img, coo, cri, cre, full=True):
    """
            Calcula el numero total de ADUs en una apertura anular, su valor medio, la mediana, la desviacion standard asumiendo una distribucion de Poisson, y el numero de pixels. Solo se
    consideran pixels enteros que se encuentren completamente dentro del anillo ya que lo que se pretende es obtener un valor representativo del brillo por pixel.

            Si el campo es muy denso es posible calcular por octavos de anillo y tomar los parametros estadisticos del tercero con menor suma total.

            Usage::
              val=anillo(img,coo,cri,cre,full=True)


            Parameters
            ----------

            img     : imagen donde esta el objeto. [array]

            coo     : tuple con coordenadas Y y X del objeto. Ojo con el orden.  [int o float]

            cri     : radio interno de la apertura. radi tiene que ser menor que cre. [float]

            cre     : radio externo de la apertura. rade tiene que ser mayor que cri. [float]

            full    : decide si toma todo el anillo (True) despreciando el 20% de los pixels con mayor valor o analiza en octavos y se queda con los valores correspondientes al tercero con menor suma total. El default es full=True. [bool]

            Returns
            -------

            val    : vector con: numero de pixels utilizado para el calculo, numero total de ADUs, valor medio, mediana y desviacion standard. [float]


            Notes
            -----

            rgh - Ago 2015
            rgh - Abril 2016
            rgh - Julio 2018: desprecia un porcentaje de los pixeles altos
            rgh - Oct 2018: modifico calculo de pixels altos, agrego posibilidad de calcular en octavos y una estimacion de la moda
    """

    # dimensiones de la imagen, lado del box de extraccion y control de
    # parametros ingresados
    dim = np.shape(img)
    box = (int(cre + 0.5) // 2 + 1) * 2

    assert (cri >= 1.0) & (cre > 1.0), "Radio de apertura < 1. Aborta."
    assert cri < cre, "Radio exterior es menor que el interior. Aborta."
    assert cri + 1 <= cre, "Seccion minima de 1 pixels. Aborta."
    assert (coo[0] >= 0.0) & (coo[1] >= 0.0), "Coord. del objeto incorrectas. Aborta"
    assert (
        (coo[0] >= box)
        & (coo[1] >= box)
        & (dim[0] - 1 - coo[0] >= box)
        & (dim[1] - 1 - coo[1] >= box)
    ), "Objeto muy cerca del borde. Aborta."

    # seccion de la imagen con el objeto. Las coord. ingresadas
    # estan en el centro de la seccion
    sec = img[
        int(coo[0] + 0.5) - box : int(coo[0] + 0.5) + box + 1,
        int(coo[1] + 0.5) - box : int(coo[1] + 0.5) + box + 1,
    ]

    # arma una mascara para determinar los pixeles a
    # considerar porque aportan flujo
    iy, ix = np.mgrid[-box : box + 1, -box : box + 1]
    ii = np.sqrt(ix**2 + iy**2)

    # selecciona los pixels dentro del anillo
    #
    inx = np.where((ii <= cre) & (ii >= cri))

    # decide como va a calcular
    #
    if full:
        # desprecia un 20% de los pixeles mas altos
        # para evitar contaminacion por otros objetos
        #
        ring0 = np.sort(sec[inx])
        nro = int(len(ring0) * 0.2 + 0.5)

        ring = ring0[:-nro]

        val = [len(ring), np.sum(ring), np.mean(ring), np.median(ring), np.std(ring)]

    else:

        # calcula en octavos del anillo y elige como
        # representativo el segundo mas bajo
        #
        vlen = []
        vsum = []
        vmea = []
        vmed = []
        vstd = []

        # arma array con angulos respecto de una direccion
        # arbitraria
        #
        ang = np.arctan2(iy, ix) * 180.0 / np.pi
        inx = np.where(ang < 0.0)
        ang[inx] += 360.0

        # determina el anillo en la imagen
        #
        msk = (ii <= cre) & (ii >= cri)

        # toma octavos del anillo y calcula la suma,
        # valor medio, mediana, moda y desviacion standard
        # para cada uno
        #
        for jj in range(6):
            msk1 = (ang >= jj * 60.0) & (ang < (jj + 1) * 60.0)
            inx = np.where(msk * msk1 == 1)
            ring0 = sec[inx]

            vlen.append(len(ring0))
            vsum.append(np.sum(ring0))
            vmea.append(np.mean(ring0))
            vmed.append(np.median(ring0))
            vstd.append(np.std(ring0))

        # elige los valores correspondientes al sector
        # con la tercera suma mas baja
        #
        vv = np.sort(np.array(vsum))
        jj = vsum.index(vv[2])
        val = [vlen[jj], vsum[jj], vmea[jj], vmed[jj], vstd[jj]]

    return val


def txt_lists(path, arch):
    # Get object frames list
    fp = glob.glob(path + arch +"_???.fits")
    with open(path + arch + ".txt", "w") as f:
        for item in fp:
            f.write("%s\n" % item)
    # Get dark frames list
    fp = glob.glob(path + "*dark" + "?"*(len(arch)+5) +".fits")
    with open(path + "dark.txt", "w") as f:
        for item in fp:
            f.write("%s\n" % item)
    # Get flat frames list
    fp = glob.glob(path + "*flat" + "?"*(len(arch)+5) +".fits")
    with open(path + "flat.txt", "w") as f:
        for item in fp:
            f.write("%s\n" % item)


#########CODIGO DE PATY.........####################################################
###############################################################################
##############################main@#########################################
#%
# refot.reducir('flats.dat','bias.dat','objetos.dat') #falta descomentar para otros archivos flats bias y objetos.

#%%

clear()

print("\nObjetos observados:\n")
print(colored.yellow(subprocess.check_output(["dir", "datos\\"]).decode()))  # Cambiar

arch = input("Ingresa el nombre de un objeto: ")

cwd = os.getcwd().format()  # Get the current working directory
path = cwd + "\datos\\" + arch + "\\" # Object data directory

txt_lists(path, arch)

obj = []
with open(path + arch + ".txt", "r") as f:
    objetos = f.readlines()
    for linea in objetos:
        obj.append(linea.strip("\n"))
# print(obj)

# n=int(input('Numero de objetos a seleccionar (siendo el primer seleccionado el objeto de inter??s): \n'))
n = 1  # numero de objetos
k = 1  # contador de imagenes
box = 30  # caja para buscar centroide
rr = (5, 10, 15, 20)  # Aperturas para fotometria
ri = 25  # Radio interno anillo de cielo
re = 30  # Radio externo del anillo de cielo
Flux = []
Imnum = []
for item in obj:
    if k == 1:  # Solo abrir imagen la primera vez
        print("\n" + item + "\n")
        im, hd = leefits(item)
        out = mou_clickN(im, N=n)
        x = out.xy[0][0]
        y = out.xy[0][1]
        yx = centro(im, (y, x), box=box)
        # print("Centroide: ({:.2},{:.2})".format(yx[0][0],yx[0][1]))
        val, sky = fot(
            img=im, obj=1, coo=yx[0], rr=rr, ri=ri, re=re, tint=hd["EXP-TIME"]
        )
        R = []  # genera una nueva lista con los ultimos elementos
        for i in range(len(val)):
            R.append(val[i][5])  # Buscando el error minimo en la fotometria
        mvalor = min(R)  # val[R.index(min(R))]
        area = val[R.index(min(R))][2]
        flux = val[R.index(min(R))][3]
        fondo = sky[2] * area
        Flux.append(flux)  # -fondo)
        Imnum.append(k)
        print(
            "Centroide: ({:.2},{:.2})-->  Error en Fotometria: {:.2}".format(
                yx[0][0], yx[0][1], mvalor
            )
        )
        # print("Error en Fotometria: {:}".format(mvalor))
        k = k + 1
    else:
        print(item)
        im, hd = leefits(item)
        yx = centro(im, (yx[0]), box=box)
        val, sky = fot(
            img=im, obj=1, coo=yx[0], rr=rr, ri=ri, re=re, tint=hd["EXP-TIME"]
        )
        R = []  # genera una nueva lista con los ultimos elementos
        for i in range(len(val)):
            R.append(val[i][5])  # Buscando el error minimo en la fotometria
        mvalor = min(R)  # val[R.index(min(R))]
        area = val[R.index(min(R))][2]
        flux = val[R.index(min(R))][3]
        fondo = sky[2] * area
        Flux.append(flux)  # -fondo)
        Imnum.append(k)
        k = k + 1
        print("Centroide: ({:})-->  Error en Fotometria: {:}".format(yx[0], mvalor))
        # print("Error en Fotometria: {:}".format(mvalor))

flujo = np.array(Flux)
imagen = np.array(Imnum)

plt.figure()
plt.clf()
plt.plot(imagen, flujo, ".-")
plt.xlabel("Numero de imagen")
plt.ylabel("Flujo")
plt.show()
print(
    colored.yellow("Grabando Datos: {:}".format(path + "Fotom_" + arch + ".csv\n"))
)
np.savetxt(
    path + "Fotom_" + arch + ".csv", np.array([imagen, flujo]).T, delimiter=","
)

