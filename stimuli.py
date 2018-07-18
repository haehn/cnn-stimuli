import math
import numpy as np
import skimage.draw

class Figure5:

    #SORRY ABOUT PARAMETERS I WILL ADD THEM LATER!
    SIZE = (100, 100)
    BIG_SIZE = (100, 150) #SY, SX
    RANGE = (10, 80) #sizes of angles generated
    POS_RANGE = (20, 80) #position range
    AUTO_SPOT_SIZE = 3

    @staticmethod
    def position_common_scale(spot_size=AUTO_SPOT_SIZE, varspot=False, preset=None):
        if preset is not None:
            img = preset
        else:
            img = np.zeros(Figure5.SIZE)
            ORIGIN = 10 #where the line is
            img[Figure5.POS_RANGE[0]:Figure5.POS_RANGE[1], ORIGIN] = 1
        if varspot:
            sizes = [1, 3, 5, 7, 9, 11]
            spot_size = np.random.choice(sizes)

        X = len(img[0]) / 2
        Y = np.random.randint(Figure5.POS_RANGE[0], Figure5.POS_RANGE[1])
        label = Y - Figure5.POS_RANGE[0]
        
        half_size = spot_size / 2
        img[int(Y-half_size):int(Y+half_size+1), int(X-half_size):int(X+half_size+1)] = 1

        sparse = [Y, X, spot_size]

        return sparse, img, label
    @staticmethod
    def multiple_pcs(num=4, spot_size=AUTO_SPOT_SIZE, varspot=False):
        label_ = []
        sparse_ = []
        img = None
        temp = Figure5.position_common_scale(spot_size, varspot)
        img = temp[1]
        label_.append(temp[2])
        sparse_.append(temp[0])
        for i in range(num-1):
            temp = Figure5.position_common_scale(spot_size, varspot, preset=img)
            label_.append(temp[2])
            sparse_.append(temp[0])
        return sparse_, img, label_
    
    @staticmethod
    def angle(X, Y, preset=None, L=20) :
        if preset is not None:
            img = preset
        else:
            img = np.zeros(Figure5.BIG_SIZE)
        startangle = np.random.randint(0, 360)
        ANGLE = np.random.randint(Figure5.RANGE[0], Figure5.RANGE[1])
        t2 = startangle * (math.pi/180)
        diff2 = ANGLE * (math.pi/180)
        r, c = skimage.draw.line(Y, X, Y+(int)(L*np.sin(t2)), X+(int)(L*np.cos(t2)))
        diffangle = t2+diff2 #angle after diff is added (2nd leg)
        r2, c2 = skimage.draw.line(Y, X, Y+(int)(L*np.sin(diffangle)), X+(int)(L*np.cos(diffangle)))
        img[r, c] = 1
        img[r2, c2] = 1
        sparse = [Y, X, ANGLE, startangle]
        return sparse, img, ANGLE
