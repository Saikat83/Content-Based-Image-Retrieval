import cv2
import numpy as np
import pickle
import operator
class Searcher:
    def __init__(self,index):
        self.index=index
    def search(self,hist):
        results={}
        eps=1e-10
        for (keys,features) in self.index.items():
            results[keys]=0.5*np.sum([((features-hist)**2)/(features+hist+eps)])
        results=sorted([(v,k) for (k,v) in results.items()])
        return results
pickle_in=open('Features.pickle','rb')
features=pickle.load(pickle_in)
print('Enter the query image name:')
name=input()
img=cv2.imread(name,cv2.IMREAD_COLOR)
WindowSize=(512,512)
img=cv2.resize(img,WindowSize,interpolation=cv2.INTER_CUBIC)
HistoGram=cv2.calcHist([img],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
hist=cv2.normalize(HistoGram,HistoGram).flatten()
Searcher_obj=Searcher(features)
resultDict=Searcher_obj.search(hist)
cv2.imshow('Query',img)
montage1=np.zeros([150*5,400,3],dtype="uint8")
montage2=np.zeros([150*5,400,3],dtype="uint8")
for j in range(0,10):
    res=cv2.imread(resultDict[j][1],cv2.IMREAD_COLOR)
    res=cv2.resize(res,(400,150),interpolation=cv2.INTER_CUBIC)
    if j<5:
        montage1[j*150:(j+1)*150,:]=res
    else:
        montage2[(j-5)*150:((j-5)+1)*150,:]=res
cv2.imshow('Top 5 Results',montage1)
cv2.imshow('Last 5 Results',montage2)
cv2.waitKey(0)
cv2.destroyAllWindows()

