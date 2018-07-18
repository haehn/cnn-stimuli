import math
import numpy as np
import skimage.draw

class Figure5:

    #SORRY ABOUT PARAMETERS I WILL ADD THEM LATER!
    SIZE = (100, 100)
    BIG_SIZE = (100, 150) #SY, SX
    RANGE = (10, 80) #sizes of angles generated
    POS_RANGE = (20, 80) #position range
    AUTO_SPOT_SIZE = 3 #how big automatic spot is in pixels
    LENGTH_RANGE = (0.4, 0.9) #how much of length 1 can others be?

    @staticmethod
    def position_variable_scale(diff=None, spot_size=AUTO_SPOT_SIZE, varspot=False, preset=None):
        if preset is not None:
            img = preset
        else:
            img = np.zeros(Figure5.SIZE)
            ORIGIN = 10 #where the line is
            if diff is not None:
                img[Figure5.POS_RANGE[0]+diff:Figure5.POS_RANGE[1]+diff, ORIGIN] = 1
            else:
                diff = np.random.randint(-9, 11)
                img[Figure5.POS_RANGE[0]+diff:Figure5.POS_RANGE[1]+diff, ORIGIN] = 1
        if varspot:
            sizes = [1, 3, 5, 7, 9, 11]
            spot_size = np.random.choice(sizes)

        X = len(img[0]) / 2
        Y = np.random.randint(Figure5.POS_RANGE[0], Figure5.POS_RANGE[1])
        Y = Y + diff
        label = Y - Figure5.POS_RANGE[0] - diff
        
        half_size = spot_size / 2
        img[int(Y-half_size):int(Y+half_size+1), int(X-half_size):int(X+half_size+1)] = 1

        sparse = [Y, X, spot_size]

        return sparse, img, label, diff

    @staticmethod
    def multiple_pvs(diff=None, num=4, spot_size=AUTO_SPOT_SIZE, varspot=False):
        label_ = []
        sparse_ = []
        img = None
        temp = Figure5.position_variable_scale(diff, spot_size, varspot)
        img = temp[1]
        label_.append(temp[2])
        sparse_.append(temp[0])
        diff = temp[3]
        for i in range(num-1):
            temp = Figure5.position_variable_scale(diff, spot_size, varspot, preset=img)
            label_.append(temp[2])
            sparse_.append(temp[0])
        return sparse_, img, label_
    
    @staticmethod
    def position_common_scale(spot_size=AUTO_SPOT_SIZE, varspot=False, preset=None):
        return Figure5.position_variable_scale(diff=0, spot_size=spot_size, varspot=varspot, preset=preset)
    
    @staticmethod
    def multiple_pcs(num=4, spot_size=AUTO_SPOT_SIZE, varspot=False):
        return Figure5.multiple_pvs(diff=0, num=num, spot_size=spot_size, varspot=varspot)
    
    @staticmethod
    def angle(X, Y, preset=None, size=BIG_SIZE, L=10) :
        if preset is not None:
            img = preset
        else:
            img = np.zeros(size)
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

    @staticmethod
    def length(X, Y, preset=None, size=BIG_SIZE, OL=20):
        L = OL
        if preset is not None:
            img = preset
            L = OL * np.random.uniform(Figure5.LENGTH_RANGE[0], Figure5.LENGTH_RANGE[1])
        else:
            img = np.zeros(size)
        half_l = int(L * 0.5)
        img[Y-half_l:Y+half_l, X] = 1
        sparse = [Y, X, L]
        return sparse, img, L

    @staticmethod
    def direction(X, Y, preset=None, size=BIG_SIZE, L=10):
        if preset is not None:
            img = preset
        else:
            img = np.zeros(size)
        angle = np.random.randint(0, 360)
        radangle = angle * math.pi / 180
        r, c = skimage.draw.line(Y, X, Y+int(L*np.sin(radangle)), X+int(L*np.cos(radangle)))
        img[r,c] = 1
        img[Y-1:Y+1, X-1:X+1] = 1
        sparse = [Y, X, angle]
        return sparse, img, angle

    @staticmethod
    def diagonal(stimulus): #draws stimulus 4x in a diagonal
        sparse_ = [] #sparse of all stimuli
        label_ = [] #label of all stimuli
        X = 30
        Y = 20
        temp = stimulus(X, Y, size=Figure5.BIG_SIZE)
        img = temp[1]
        for i in range(3):
            X = X + 30
            Y = Y + 20
            temp = stimulus(X, Y, preset=img)
            sparse_.append(temp[0])
            label_.append(temp[2])
        return sparse_, img, label_
