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
    CURV_WIDTH = 30 #auto curvature width

    @staticmethod
    def position_non_aligned_scale(diff=None, spot_size=AUTO_SPOT_SIZE, varspot=False, preset=None):
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
    def multiple_pnas(diff=None, num=4, spot_size=AUTO_SPOT_SIZE, varspot=False):
        label_ = []
        sparse_ = []
        img = None
        temp = Figure5.position_non_aligned_scale(diff, spot_size, varspot)
        img = temp[1]
        label_.append(temp[2])
        sparse_.append(temp[0])
        diff = temp[3]
        for i in range(num-1):
            temp = Figure5.position_non_aligned_scale(diff, spot_size, varspot, preset=img)
            label_.append(temp[2])
            sparse_.append(temp[0])
        return sparse_, img, label_
    
    @staticmethod
    def position_common_scale(spot_size=AUTO_SPOT_SIZE, varspot=False, preset=None):
        return Figure5.position_non_aligned_scale(diff=0, spot_size=spot_size, varspot=varspot, preset=preset)
    
    @staticmethod
    def multiple_pcs(num=4, spot_size=AUTO_SPOT_SIZE, varspot=False):
        return Figure5.multiple_pnas(diff=0, num=num, spot_size=spot_size, varspot=varspot)
    
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
        radangle = angle * np.pi / 180
        r, c = skimage.draw.line(Y, X, Y+int(L*np.sin(radangle)), X+int(L*np.cos(radangle)))
        img[r,c] = 1
        img[Y-1:Y+1, X-1:X+1] = 1
        sparse = [Y, X, angle]
        return sparse, img, angle

    @staticmethod
    def area(X, Y, preset=None, size=BIG_SIZE):
        if preset is not None:
            img = preset
        else:
            img = np.zeros(size)
        DOF = 19
        radius = np.random.randint(1, DOF+1)
        r, c = skimage.draw.ellipse_perimeter(Y, X, radius, radius)
        img[r, c] = 1
        sparse = [Y, X, radius]
        label = np.pi * radius * radius
        return sparse, img, label

    @staticmethod
    def volume(X, Y, preset=None, size=BIG_SIZE, autosize=20):
        if preset is not None:
            img = preset
        else:
            img = np.zeros(size)
        depth = np.random.randint(1, autosize+1)

        def obliqueProjection(point):
            angle = -45.
            alpha = (np.pi/180.0) * angle
            P = [[1, 0, (1/2.)*np.sin(alpha)], [0, 1, (1/2.)*np.cos(alpha)], [0, 0, 0]]
            ss = np.dot(P, point)
            return [int(np.round(ss[0])), int(np.round(ss[1]))]
        halfdepth = int(depth/2.)
        fbl = (Y+halfdepth, X-halfdepth)
        fbr = (fbl[0], fbl[1]+depth)
    
        r, c = skimage.draw.line(fbl[0], fbl[1], fbr[0], fbr[1])
        img[r, c] = 1

        ftl = (fbl[0]-depth, fbl[1])
        ftr = (fbr[0]-depth, fbr[1])
        
        r, c = skimage.draw.line(ftl[0], ftl[1], ftr[0], ftr[1])
        img[r,c] = 1
        r, c = skimage.draw.line(ftl[0], ftl[1], fbl[0], fbl[1])
        img[r, c] = 1
        r, c = skimage.draw.line(ftr[0], ftr[1], fbr[0], fbr[1])
        img[r, c] = 1

        bbr = obliqueProjection([fbr[0], fbr[1], depth])
        btr = (bbr[0]-depth, bbr[1])
        btl = (btr[0], btr[1]-depth)

        r, c = skimage.draw.line(fbr[0], fbr[1], bbr[0], bbr[1])
        img[r, c] = 1
        r, c = skimage.draw.line(bbr[0], bbr[1], btr[0], btr[1])
        img[r, c] = 1
        r, c = skimage.draw.line(btr[0], btr[1], btl[0], btl[1])
        img[r, c] = 1
        r, c = skimage.draw.line(btl[0], btl[1], ftl[0], ftl[1])
        img[r, c] = 1
        r, c = skimage.draw.line(btr[0], btr[1], ftr[0], ftr[1])
        img[r, c] = 1

        sparse = [Y, X, depth]
        label = depth ** 3
        return sparse, img, label
        

    @staticmethod
    def curvature(X, Y, preset=None, size=BIG_SIZE, varwidth=False):
        if preset is not None:
            img = preset
        else:
            img = np.zeros(size)
        DOF = 30
        depth = np.random.randint(1, DOF+1)
        width = Figure5.CURV_WIDTH
        if varwidth:
            width = np.random.randint(1, width/2)*2
        halfwidth = int(width/2)
        start = (Y, X-halfwidth)
        mid = (Y-depth, X)
        end = (Y, X+halfwidth)
        r, c = skimage.draw.bezier_curve(start[0], start[1], mid[0], mid[1], end[0], end[1], 1)
        img[r, c] = 1
        sparse = [Y, X, depth, width]
        t = 0.5
        P10 = (mid[0] - start[0], mid[1] - start[1])
        P21 = (end[0] - mid[0], end[1] - mid[1])
        dBt_x = 2*(1-t)*P10[1] + 2*t*P21[1]
        dBt_y = 2*(1-t)*P10[0] + 2*t*P21[0]
        dBt2_x = 2*(end[1] - 2*mid[1] + start[1])
        dBt2_y = 2*(end[0] - 2*mid[0] + start[0])
        curvature = np.abs((dBt_x*dBt2_y - dBt_y*dBt2_x) / ((dBt_x**2 + dBt_y**2)**(3/2.)))
        label = np.round(curvature, 3)
        return sparse, img, label

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
