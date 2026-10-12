import numpy as np
from fatiga import PARAM, paso
p=PARAM; F,R,r=p["F"],p["R"],p["r"]
def pico_full(TL,Tw,Tr,ncyc=60):
    x=np.array([0,1.,0]); pk=[]
    for c in range(ncyc):
        m=0
        for t in range(int(Tw)): x=paso(TL,x,1.0,p); m=max(m,x[2])
        for t in range(int(Tr)): x=paso(0.0,x,1.0,p); m=max(m,x[2])
        pk.append(m)
    return pk[-1]
def pico_red(TL,Tw,Tr):
    return F*TL/R*(1-np.exp(-R*Tw))/(1-np.exp(-R*(Tw+r*Tr)))
def rest_min(TL,Tw,th):
    a=1-(F*TL/(R*th))*(1-np.exp(-R*Tw))
    if a<=0: return np.inf
    return max(0,(-np.log(a)-R*Tw)/(r*R))
for TL,Tw,Tr in [(0.1,60,30),(0.2,90,30),(0.3,120,60),(0.15,300,60),(0.4,30,60),(0.05,600,10)]:
    f=pico_full(TL,Tw,Tr); g=pico_red(TL,Tw,Tr)
    print(TL,Tw,Tr,"full %.4f red %.4f relerr %.2f%%"%(f,g,100*(g-f)/f))
for TL,Tw,th in [(0.2,90,0.4),(0.3,60,0.4),(0.1,120,0.25)]:
    Tr=rest_min(TL,Tw,th); print("TL",TL,"Tw",Tw,"theta",th,"Tr_min",round(Tr,1),"check full peak",round(pico_full(TL,Tw,int(np.ceil(Tr))),4))
