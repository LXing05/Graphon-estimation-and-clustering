import numpy as np
from math import log, exp, sin, cos,pi

def phi1(x):
    return 1-x

def phi2(x):
    return 2*x/(x+1)

def phi3(x):
    return x**2

def phi4(x):
    return 1/(x+1)

def phi5(x):
    return x**3



def W1_0(u,v):
    return sin(5*pi*(u+v-1)+1)/2+0.5

def W1_1(u,v):
    return sin(5*pi*(phi1(u)+phi1(v)-1)+1)/2+0.5


def W1_2(u,v):
    return sin(5*pi*(phi2(u)+phi2(v)-1)+1)/2+0.5


def W1_3(u,v):
    return sin(5*pi*(phi3(u)+phi3(v)-1)+1)/2+0.5


def W1_4(u,v):
    return sin(5*pi*(phi4(u)+phi4(v)-1)+1)/2+0.5


def W1_5(u,v):
    return sin(5*pi*(phi5(u)+phi5(v)-1)+1)/2+0.5



def W2_0(u,v):
    return 1-1/(1+exp(15*(0.8*abs(u-v))**(4/5)-0.1))


def W2_1(u,v):
    return 1-1/(1+exp(15*(0.8*abs(phi1(u)-phi1(v)))**(4/5)-0.1))



def W2_2(u,v):
    return 1-1/(1+exp(15*(0.8*abs(phi2(u)-phi2(v)))**(4/5)-0.1))



def W2_3(u,v):
    return 1-1/(1+exp(15*(0.8*abs(phi3(u)-phi3(v)))**(4/5)-0.1))



def W2_4(u,v):
    return 1-1/(1+exp(15*(0.8*abs(phi4(u)-phi4(v)))**(4/5)-0.1))



def W2_5(u,v):
    return 1-1/(1+exp(15*(0.8*abs(phi5(u)-phi5(v)))**(4/5)-0.1))




def W3_0(u,v):
    return (u**2+v**2)/3*cos(1/(u**2+v**2))+0.15 





def W3_1(u,v):
    return (phi1(u)**2+phi1(v)**2)/3*cos(1/(phi1(u)**2+phi1(v)**2))+0.15 



def W3_2(u,v):
    return (phi2(u)**2+phi2(v)**2)/3*cos(1/(phi2(u)**2+phi2(v)**2))+0.15 




def W3_3(u,v):
    return (phi3(u)**2+phi3(v)**2)/3*cos(1/(phi3(u)**2+phi3(v)**2))+0.15 




def W3_4(u,v):
    return (phi4(u)**2+phi4(v)**2)/3*cos(1/(phi4(u)**2+phi4(v)**2))+0.15 




def W3_5(u,v):
    return (phi5(u)**2+phi5(v)**2)/3*cos(1/(phi5(u)**2+phi5(v)**2))+0.15 



def W4_0(x,y):
    return (x**2+y**2)/2 


def W4_1(x,y):
    return (phi1(x)**2+phi1(y)**2)/2 



def W4_2(x,y):
    return (phi2(x)**2+phi2(y)**2)/2 



def W4_3(x,y):
    return (phi3(x)**2+phi3(y)**2)/2 



def W4_4(x,y):
    return (phi4(x)**2+phi4(y)**2)/2 


def W4_5(x,y):
    return (phi5(x)**2+phi5(y)**2)/2 



def W5_0(x,y):
    w=min(x,y)*(1-max(x,y))
    return w


def W5_1(x,y):
    w=min(phi1(x),phi1(y))*(1-max(phi1(x),phi1(y)))
    return w



def W5_2(x,y):
    w=min(phi2(x),phi2(y))*(1-max(phi2(x),phi2(y)))
    return w




def W5_3(x,y):
    w=min(phi3(x),phi3(y))*(1-max(phi3(x),phi3(y)))
    return w



def W5_4(x,y):
    w=min(phi4(x),phi4(y))*(1-max(phi4(x),phi4(y)))
    return w




def W5_5(x,y):
    w=min(phi5(x),phi5(y))*(1-max(phi5(x),phi5(y)))
    return w




def W6_0(x,y):
    return abs(x-y)



def W6_1(x,y):
    return abs(phi1(x)-phi1(y))



def W6_2(x,y):
    return abs(phi2(x)-phi2(y))



def W6_3(x,y):
    return abs(phi3(x)-phi3(y))


def W6_4(x,y):
    return abs(phi4(x)-phi4(y))



def W6_5(x,y):
    return abs(phi5(x)-phi5(y))


