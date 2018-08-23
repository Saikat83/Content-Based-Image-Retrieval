import cv2
import pickle
class Histogram:
    def __init__(self,bins):
        self.bins=bins
    def ThreeDColorHistogram(self,img):
        hist=cv2.calcHist([img],[0,1,2],None,self.bins,[0,256,0,256,0,256])
        hist=cv2.normalize(hist,hist)
        return hist.flatten()
dict1={}
for i in range(1,26):
    name='Image'+str(i)+'.png'
    img=cv2.imread(name,cv2.IMREAD_COLOR)
    WindowSize=(512,512)
    img=cv2.resize(img,WindowSize,interpolation=cv2.INTER_CUBIC)
    bins=[8,8,8]
    ColorHist=Histogram(bins)
    dict1['Image'+str(i)+'.png']=ColorHist.ThreeDColorHistogram(img)
pickle_out=open("Features.pickle",'wb')
pickle.dump(dict1,pickle_out)
pickle_out.close()
cv2.destroyAllWindows()
