# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Definición de variables
global CoordenadaX,CoordenadaY
global circulos, maxCirculos,C


#Funciones
def GetPupil(frame):#Función para obtener los contornos del ojo y la pupila
    global pupilImg
    global pupil
    mascara = cv2.Canny(frame, 30,80)
    im2, contours, hierarchy = cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    pupilImg = frame.copy()
    pupil = contours
    cv2.drawContours(pupilImg, pupil,-1, (255,0,0), 2, cv2.FILLED)
    return pupilImg#Contornos

def HoughTransform(frame1,frame2):#Función para detectar círculos en la imagen
    global circulos
    global CoordenadaX
    global CoordenadaY,C
    C=C+1
    CirclesIMG=frame2
    circles=cv2.HoughCircles(CirclesIMG, cv2.HOUGH_GRADIENT, 24, 20, param1=30, param2=33, minRadius=15,maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0,:]:
            cv2.circle(frame1, (c[0],c[1]), c[2], (255,0,0),2)
            circulos=circulos+1
            print ("Coordenada X: " + str(c[0]))
            
            print ("Coordenada Y: " + str(c[1]))
            if(C%2==0):
                CoordenadaX.append(c[0])
                CoordenadaY.append(c[1])
            if(circulos>maxCirculos):
                break
    cv2.imshow("Circulos",frame1)#Graficar los círculos detectados sobre la interpolación de la imagen original
        
    
#Código   
#video = cv2.VideoCapture('C:/Users/Sergio Sanchez/Documents/PIS/PISPrueba12.avi')
video=cv2.VideoCapture(0) #Selecciona la cámara predeterminada como dispositivo de grabación por el PC

#Esta sección se utiliza para realizar las grabaciones de prueba

'''fourcc = cv2.VideoWriter_fourcc(*'XVID')#Define el formato del video
out = cv2.VideoWriter('PISPrueba13.avi',fourcc, 20.0, (640,480))#Define la salida y el nombre del video'''

#Inicialización de Variables que se utilizan en el código
#Para ojo derecho
c1=145
r1=225

#Para ojo izquierdo
'''c1=145+80
r1=225'''
C=0
circulos=1
maxCirculos=2
Band=True
CoordenadaX=[]
CoordenadaY=[]

while Band==True:
    #Obtención de la imagen en tiempo real (Video)
    ret, frame = video.read()

#Sección para grabar videos de prueba
    '''if ret==True:
        frame = cv2.flip(frame,1)
        out.write(frame)'''

    img = frame
    if ret==False:
        cv2.destroyAllWindows()
        video.release()
        break
    
    cv2.imshow("Original",img)#Graficar imagen obtenida por la cámara o archivo de video
    
    #Conversión de la imagen a escala de grises para procesamiento digital
    framegrey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Interpolación y selección del ROI para un solo ojo
    img = framegrey[c1:c1+35,r1:r1+60]
    inter= cv2.resize(img, None, fx = 3, fy = 5, interpolation = cv2.INTER_CUBIC)#cv2.INTER_LANCZOS4)
    img1 = frame[c1:c1+35,r1:r1+60]
    inter1= cv2.resize(img1, None, fx = 3, fy = 5, interpolation = cv2.INTER_CUBIC)#cv2.INTER_LANCZOS4)
    
    #Interpolación y selección del ROI para ambos ojos
    '''img = framegrey[c1:c1+35,r1:r1+140]
    inter= cv2.resize(img, None, fx = 3, fy = 5, interpolation = cv2.INTER_CUBIC)#cv2.INTER_LANCZOS4)
    img1 = frame[c1:c1+35,r1:r1+140]
    inter1= cv2.resize(img1, None, fx = 3, fy = 5, interpolation = cv2.INTER_CUBIC)#cv2.INTER_LANCZOS4)'''
    
    #Filtros de imagen
    kernel = np.ones((3,3),np.float32)/9
    framegrey = cv2.filter2D(inter,-1,kernel)
    framegrey = cv2.GaussianBlur(framegrey, (1,1), 2)
    framegrey = cv2.medianBlur(framegrey,5)    

    #Función de obtención de contornos de la pupila
    Pupila=GetPupil(inter)
    HoughTransform(inter1,Pupila)
    #Sección para dibujar y graficar los contornos sobre la imagen original
    '''cv2.drawContours(inter1, pupil,-1, (255,0,0), 2, cv2.FILLED)
    cv2.imshow("Contours Original",inter1)'''
    
    key = cv2.waitKey(1)
    if key==27:
        Band=False
    
video.release()
#out.release()#Se debe descomentar esta linea para poder grabar los videos de prueba
cv2.destroyAllWindows()

#Gráficos de las coordenadas de manera individual en cada ojo.
plt.figure('Figura 1')
plt.plot(CoordenadaX,'o',linewidth=0.2,color=(0.2,0.1,0.4))
plt.plot(CoordenadaX,linewidth=1,color='r')
plt.title('Ojo Derecho Derecha-Izquierda')
plt.ylabel('Coordenada X [Pixel]')
plt.figure('Figura 2')
plt.plot(CoordenadaY,'o',linewidth=0.2,color=(0.2,0.1,0.4))
plt.plot(CoordenadaY,linewidth=1,color='r')
plt.title('Ojo Derecho Arriba-Abajo')
plt.ylabel('Coordenada Y [Pixel]')
plt.show()


