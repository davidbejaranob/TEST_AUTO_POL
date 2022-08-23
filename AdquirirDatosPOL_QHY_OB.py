#!/bin/python3
import ctypes
from ctypes import *
import numpy as np
import time
import cv2 as cv
from matplotlib import pyplot as plt
from libqhy import *

# qhyccd = CDLL('libqhyccd.so.6.0.4')
# qhyccd = CDLL('pkg_win/x64/qhyccd.dll')#locaL
qhyccd = CDLL("./qhyccd.dll")  # Desde la carpeta que contiene el proyecto
# qhyccd = CDLL("C:/Program Files/QHYCCD/AllInOne/sdk/x64/qhyccd.dll")  # iNSTALADO
qhyccd.GetQHYCCDParam.restype = c_double
qhyccd.OpenQHYCCD.restype = ctypes.POINTER(c_uint32)
# ref: https://www.qhyccd.com/bbs/index.php?topic=6356.0
# H.F. @20191115

import re, readline, time
from lib import argv
from lib.communicate import Communicate
from clint.textui import colored, puts

from astropy.time import Time
import astropy.io.fits as ft


def init_cam():
    # INICIALIZAR CAMARA QHY174M
    ret = -1
    qhyccd.InitQHYCCDResource()
    qhyccd.ScanQHYCCD()
    type_char_array_32 = c_char * 32
    id = type_char_array_32()
    qhyccd.GetQHYCCDId(c_int(0), id)  # open the first camera
    print(id.value)
    cam = qhyccd.OpenQHYCCD(id)
    qhyccd.SetQHYCCDStreamMode(cam, 0)  # 0 for single frame
    qhyccd.InitQHYCCD(cam)

    # OBTENER  PARAMETROS GENERALES DE LA CAMARA
    chipw = c_double()
    chiph = c_double()
    w = c_uint()
    h = c_uint()
    pixelw = c_double()
    pixelh = c_double()
    bpp = c_uint()
    channels = c_uint32(1)
    qhyccd.GetQHYCCDChipInfo(
        cam,
        byref(chipw),
        byref(chiph),
        byref(w),
        byref(h),
        byref(pixelw),
        byref(pixelh),
        byref(bpp),
    )

    # FIJAR TEMPERATURA DE TRABAJO
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_COOLER, c_double(-5.0))
    print("Current temp: ", qhyccd.GetQHYCCDParam(cam, CONTROL_ID.CONTROL_CURTEMP))
    print("Target temp: ", qhyccd.GetQHYCCDParam(cam, CONTROL_ID.CONTROL_COOLER))

    # OBTENER IMAGEN DE PRUEBA FULL FRAME
    roi_w = c_uint(1920)
    roi_h = c_uint(1200)

    imgdata = (ctypes.c_uint16 * roi_w.value * roi_h.value)()
    qhyccd.SetQHYCCDResolution(cam, 0, 0, roi_w, roi_h)
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_EXPOSURE, c_double(9000))  # unit: us
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_GAIN, c_double(100))
    # qhyccd.SetQHYCCDParam(cam, CONTROL_ID., c_double(100))

    ret = qhyccd.ExpQHYCCDSingleFrame(cam)
    ret = qhyccd.GetQHYCCDSingleFrame(
        cam, byref(roi_w), byref(roi_h), byref(bpp), byref(channels), imgdata
    )
    # x = np.asarray(imgdata)
    # plt.imshow(x)
    return cam


def show_cam_img(cam, et=0.5, gan=1, n_imgs=1):
    # SOLO DESPLIEGUE DE IMAGENES
    bpp = c_uint()
    channels = c_uint32(1)
    # x='$J=G91G21X-20F100';serial.run(x)
    # n_imgs=5
    # 1920x1200
    roi_w = c_uint(1920)
    roi_h = c_uint(1200)
    expt = et * 1e6
    # gan=50
    print("Current temp: ", qhyccd.GetQHYCCDParam(cam, CONTROL_ID.CONTROL_CURTEMP))
    imgdata = (ctypes.c_uint16 * roi_w.value * roi_h.value)()
    qhyccd.SetQHYCCDResolution(cam, 0, 0, roi_w, roi_h)
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_EXPOSURE, c_double(expt))  # unit: us
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_GAIN, c_double(gan))
    plt.figure(1)

    for kk in range(n_imgs):
        ret = qhyccd.ExpQHYCCDSingleFrame(cam)
        ret = qhyccd.GetQHYCCDSingleFrame(
            cam, byref(roi_w), byref(roi_h), byref(bpp), byref(channels), imgdata
        )
        x = np.asarray(imgdata)
        plt.clf()
        # time.sleep(0.001)
        plt.imshow(x)
        plt.gray()
        plt.title("Img Num: {:}".format(kk))
        plt.draw()
        plt.pause(0.1)
    # return(x)


def get_cam_array(cam, et=0.5, gan=1):
    bpp = c_uint()
    channels = c_uint32(1)
    roi_w = c_uint(1920)
    roi_h = c_uint(1200)
    expt = et * 1e6
    # print("Current temp: ",qhyccd.GetQHYCCDParam(cam,CONTROL_ID.CONTROL_CURTEMP))
    imgdata = (ctypes.c_uint16 * roi_w.value * roi_h.value)()
    qhyccd.SetQHYCCDResolution(cam, 0, 0, roi_w, roi_h)
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_EXPOSURE, c_double(expt))  # unit: us
    qhyccd.SetQHYCCDParam(cam, CONTROL_ID.CONTROL_GAIN, c_double(gan))

    ret = qhyccd.ExpQHYCCDSingleFrame(cam)
    ret = qhyccd.GetQHYCCDSingleFrame(
        cam, byref(roi_w), byref(roi_h), byref(bpp), byref(channels), imgdata
    )
    x = np.asarray(imgdata)
    # time.sleep(0.001)
    return x


def init_openbuilds():
    # INICIALIZAR COMUNICACIÓN CON LA MESA LINEAL DE OPENBUILDS
    # OJO REVISAR EL PUERTO SERIAL EN EL SW DE OPENBUILDS
    with Communicate("COM3", 115200, timeout=0.70, debug=0, quiet=0) as serial:
        x = "$$"
        serial.run(x)
        x = "$X"
        serial.run(x)  # Desbloquear
    return serial


def close_cam(cam):
    # PREFERENTEMENTE CORRER ESTA CELDA PARA LIBERAR LA CAMARA
    qhyccd.CloseQHYCCD(cam)
    qhyccd.ReleaseQHYCCDResource()


#################################################################
#################Clase para obtener las coordenadas del#########
############## click en una imagen desplegada en zscale#############
class mou_click:
    def __init__(self, img, Zsc=True):

        # Simple mouse click function to store coordinates
        def onclick(event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            # assign global variable to access outside of function
            global coords
            coords.append((ix, iy))

            # Disconnect after 1 clicks
            if len(coords) == 1:
                fig.canvas.mpl_disconnect(cid)
                plt.close(1)
            return

        # print("Click sobre Objeto")
        # fig = plt.figure(1)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        fig = plt.figure(1, figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # ax.imshow(img**0.5)
        if Zsc == True:
            a, b = zscale(img)
        else:
            a = np.min(img)
            b = np.max(img)

        plt.imshow(img, vmin=a, vmax=b)  # mejorar el contraste
        plt.jet()

        # Call click func
        global coords
        coords = []
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show(block=True)  # Bloquear para interactuar

        self.xy = coords


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
    samples = np.asarray(samples, dtype=np.float)
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
    badpix = np.zeros(npix, dtype=np.int)

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
        kernel = np.ones(ngrow, dtype=np.int)
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


#######################################################################
def rebin(arr, new_shape):
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)


###################################################################
def corte_xy(img, x, y, boxG=140, bi=(2, 2)):
    """Funcion para recortar y salvar imagen procesada
    img-->corregida por flat y bias ya es NUMPY array
    boxG--> tam de la caja a guardar
    ang--> angulo polarizador string
    x--> centoride X python del cometa
    y--> centoride Y python del cometa

    OUT--> salva archivo y regresa el numpy array de la img recortada
    """
    if np.size(img) == 1:
        print("Error en corte arreglo IMG vacio")
        return 0
    b2 = int(boxG / 2)  # mitad de la caja
    x = int(x)
    y = int(y)
    imr = img[y - b2 : y + b2, x - b2 : x + b2]  # recorte
    # Evitar valores negativos
    m = np.min(imr)
    if m < 0:
        m = m * -1
    else:
        m = 0
    imr = imr + m
    ######################
    # Aplicar Binning
    imrb = rebin(imr, bi)
    return imr


################################################################
def create_header():
    # Creacion del Header
    hdu = ft.PrimaryHDU()
    hdu.header["BITPIX"] = 16
    hdu.header["EXP-TIME"] = 1
    hdu.header["OBJECT"] = ""
    hdu.header["OBSERVER"] = "Equipo X"
    hdu.header["GAIN"] = 1  # ganancia de la QHY
    hdu.header["JD"] = Time.now().jd
    hdu.header["DATE-OBS"] = Time.now().isot
    hdu.header["NAME"] = ""
    hdu.header["GNO"] = 0
    hdu.header["POL"] = 0
    h = hdu.header
    return h


def mover_OB(serial, gra, axi="Y", di="+", speed=100, m=5):
    z = "$J=G91G21" + axi + str(gra / m) + "F" + str(speed)
    serial.run(z)


###############################################################################################################
##################################INICIA CODIGO##############################################################
###############################################################################################################

cam = init_cam()
serial = init_openbuilds()
h = create_header()
gan = 1
et = 0.5
h["GAIN"] = gan
h["EXP-TIME"] = et
pre_arch = input("Prefijo Archivo FITS: ")
h["OBJECT"] = pre_arch
gno_ini = 3.5
sle_time = 7
gno_paso = 5
mgno = 4.4903  # Grados por mm
gno_tot = 90
pol_paso = 45
mpol = 4.97  # grados por mm
pol_tot = 135
img = get_cam_array(cam, et=et, gan=gan)
print("CLICK en el Centro de la imagen")
f = mou_click(img)
oxy = np.array(f.xy)
x0 = int(oxy[0, 0])
y0 = oxy[0, 1]
kk = 1
for k in range(int((gno_tot - gno_ini) / gno_paso) + 1):  # Mover Goniometro
    if k == 0:
        y = "$J=G91G21Y" + str(0) + "F100"
    else:
        y = "$J=G91G21Y" + str(gno_paso / mgno) + "F100"
    gno_pos = (k) * gno_paso + gno_ini
    print("Muevo Goniometro: {:} grados".format(gno_pos))
    serial.run(y)  # para acomodar posiciones
    time.sleep(sle_time / 2)
    for l in range(int(pol_tot / pol_paso) + 1):  # Rotar Polarizador
        if l == 0:
            z = "$J=G91G21Z" + str(0) + "F200"
        else:
            z = "$J=G91G21Z" + str(pol_paso / mpol) + "F200"
        pol_pos = l * pol_paso
        print("Muevo Polarizador: {:} grados".format(pol_pos))
        serial.run(z)  # para acomodar posiciones
        time.sleep(sle_time)
        # Capturar y desplegar imagen
        # show_cam_img(cam)
        img = get_cam_array(cam, et=et, gan=gan)

        TT = Time.now()
        imr = corte_xy(img, x=x0, y=y0)
        plt.figure(1)
        plt.imshow(imr)
        plt.gray()
        plt.title("GNO: {:}, POL: {:}".format(gno_pos, pol_pos))
        plt.draw()
        plt.pause(1)
        h["GNO"] = gno_pos
        h["POL"] = pol_pos
        h["JD"] = TT.jd  # Obtener el tiempo de adquisicion de la imagen
        h["DATE-OBS"] = TT.isot
        h["NAME"] = "{:}_{:03}.fits".format(pre_arch, kk)
        ft.writeto(
            "{:}_{:03}.fits".format(pre_arch, kk), imr, header=h, overwrite=True
        )  # >Grabar imagen
        time.sleep(0.05)
        kk = kk + 1

    z = "$J=G91G21Z-" + str((pol_tot - 1) / mpol) + "F300"
    serial.run(z)  # para regresar a 0 ojo el -1 es por backlash
    time.sleep(sle_time)
y = "$J=G91G21Y-" + str(((k) * gno_paso - 2) / mgno) + "F100"
serial.run(y)  # para regresar a 0 el -2 es por backlash
time.sleep(sle_time * 2)
close_cam(cam)
