# CrowdNav State Modified Ver
# add Dummy Ped (Vx=Vy=R=0) for sarl
# FOV ROI Applied

import math
import numpy as np
from math import *
verbose = False


def printv(*text):
    if verbose:
        print(*text)

class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.fov =0 

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        printv('selfState', self_state)

        totalObj = len(human_states)

        self_state.theta = np.arctan2(self_state.vy, self_state.vx)
        #human_states.append(dummyState)


        # Dummy State... It will be added if no state in the very moment of Obs State
        dumX = 11.9*cos(self_state.theta) + self_state.py
        dumY = 11.9*sin(self_state.theta) + self_state.px
        dummyState = ObservableState(dumX, dumY, 0, 0 ,0)

        dumY2 = 21.9*cos(self_state.theta) + self_state.py
        dumX2 = 21.9*sin(self_state.theta) + self_state.px

        dummyState2 = ObservableState(dumX2, dumY2, 0, 0 ,0)
        human_states.append(dummyState2)

       
        
        fovState = []

        for human_state in human_states:
            assert isinstance(human_state, ObservableState)
            if self.fovFilter(human_state, self_state):
                printv("In FOV!", human_state.__str__())
                human_state.fov = 1
                fovState.append(human_state)


        tempText = [ x.__str__() for x in fovState]
        tempText2 = [ x.__str__() for x in human_states]

        printv('applied Result', tempText)
        printv('nonapplied Result', tempText2)
        assert(tempText2 != tempText)


        if not len(fovState):
            fovState.append(ObservableState(self_state.px+10,self_state.py+10, 0, 0 ,0))



        printv('cnt', totalObj, len(fovState))
        printv('dum state', dummyState)
        printv('dum state2', dummyState2)


        
        self.self_state = self_state
        self.human_states = fovState
        
    def uniformDummy(self, human_state):

        for i in range(1):
            human_state.append(ObservableState(0,0, 0, 0 ,0))
        print(human_state)
        return human_state
            

    def fovFilter(self, human_state, self_state):
        # Do FOV Filter with D435I Depth Sensor Range =  85.2 Deg and 12m deg

    
        # New Approach, Convert Pos X Y to Polar Coordinate and Check
        tempX, tempY = human_state.px - self_state.px, human_state.py - self_state.py
        #qy, qx = self.rotate((0,0), (tempX, tempY), (radians(90) - self_state.theta ))
        
        


        qx, qy = self.rotate((0,0), (tempX, tempY), (radians(90) - self_state.theta))
        

        cAngle = degrees(np.arctan2(qy, qx)) % 360
        #cAngle
        cRadius2 = qx * qx + qy * qy
        #print(tempX, tempY, qx, qy )
        printv('cAngle', cAngle, 'cRadius2', cRadius2, 'theta', degrees(self_state.theta), self_state.theta)
        
        SA = (degrees(self_state.theta % 360) - 42.6) % 360
        EA = (degrees(self_state.theta % 360) + 42.6) % 360
        printv("SA", SA, "EA", EA)


        if cRadius2 > 12*12: return None
        if EA > SA:
            if SA < cAngle < EA:
                return human_state
        elif SA > EA:
            if cAngle > SA:
                return human_state
            elif cAngle < EA:
                return human_state
            else:
                return None
        else:
            return None
            
        # if cRadius2 < 12*12 and 90 + 90 > degrees(cAngle) > 90 - 42.6:
        #     print(degrees(cAngle))
        #     return human_state
        # else:
        #     return None



        # # 1. shift all pts to make robots's loc as origin
        # # 2. rotate pts (theta - 90) for aligning robots heading toward 90 deg

        # # simple math


        # #printv(qx, qy)
        # if qy < 12 and abs(qx / tan(radians(85.2))) < abs(qx):
        #     # In ROI
        #     printv()
        #     return human_state
        # else:
        #     return None


    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy


