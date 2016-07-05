import numpy

class Bicycle:

    def __init__(self):
        self.dt = 0.01
        self.v = 10.0/3.6	
        self.g = 9.82
        self.dCM = 0.3  
        self.c = 0.66
        self.h = 0.94   
        self.Mc = 15.0
        self.Md = 1.7
        self.Mp = 60.0
        self.M = (self.Mc + self.Mp)
        self.R = 0.34
        self.sigma_dot = self.v / self.R
        self.I_bike = (13.0/3.0)*self.Mc*self.h*self.h + self.Mp*(self.h+self.dCM)*(self.h+self.dCM)
        self.I_dc = (self.Md * self.R * self.R)
        self.I_dv = (3.0/2.0) * self.Md * self.R* self.R
        self.I_dl = (1.0/2.0) * self.Md * self.R*self.R
        self.l = 1.11
        self.maxNoise = 0.02 
        self.default_angle = 0.
        self.x_goal = 1000
        self.y_goal = 0
        self.radius_goal = 10
        self.c_reward = 0.1
        self.currentState = [.0,.0,.0,.0,.0]
        self.M_PI = numpy.pi
        self.shouldTerminate = False
        
        self.iomega = 0
        self.iomega_dot = 1
        self.itheta = 2
        self.itheta_dot = 3
        self.iangleGoal = 4


    def reset(self):
        self.theta = self.theta_dot = self.theta_d_dot = 0.0
        self.omega = self.omega_dot = self.omega_d_dot = 0.0

        self.xb = 0.0
        self.yb = 0.0
        
        self.angle = 0.
        self.xf = self.l * numpy.cos(self.angle)
        self.yf = self.l * numpy.sin(self.angle)
        self.psi = numpy.arctan2(self.yf - self.yb, self.xf - self.xb)
        self.psi_goal = self.calc_angle_to_goal(self.xf, self.xb, self.yf, self.yb)
        self.lastdtg = self.calc_dist_to_goal(self.xf, self.xb, self.yf, self.yb)
        self.shouldTerminate = False


    def step(self, action):
        self.psi_t = self.psi
        discreteActions = [
        [ -2, -0.02 ],     [ -2, 0 ],     [ -2, 0.02 ],
        [ 0, -0.02 ], [ 0, 0 ], [ 0, 0.02 ],
        [ 2, -0.02 ],  [ 2, 0 ],  [ 2, 0.02 ] ]


        T = discreteActions[int(action)][0]
        d = discreteActions[int(action)][1]
        
        noise = numpy.random.rand()
        d = d + self.maxNoise * noise
        if (self.theta == 0):
            self.rCM = self.rf = self.rb = 9999999 #Just a large number
        else:
            self.rCM = numpy.sqrt((self.l-self.c)**2 + self.l*self.l/(numpy.tan(self.theta)**2))
            self.rf = self.l / abs(numpy.sin(self.theta))
            self.rb = self.l / abs(numpy.tan(self.theta))

        phi = self.omega + numpy.arctan(d/self.h)
        self.omega_d_dot = ( self.h*self.M*self.g*numpy.sin(phi)
					- numpy.cos(phi)*(self.I_dc*self.sigma_dot*self.theta_dot
					+ numpy.sign(self.theta)*self.v*self.v*(self.Md*self.R*(1.0/self.rf + 1.0/self.rb)
					+ self.M*self.h/self.rCM) )
				  ) / self.I_bike
        self.theta_d_dot = (T - self.I_dv*self.omega_dot*self.sigma_dot) /  self.I_dl
        self.omega_dot += self.omega_d_dot * self.dt
        self.omega += self.omega_dot * self.dt
        self.theta_dot += self.theta_d_dot * self.dt
        self.theta += self.theta_dot * self.dt
        
        if (abs(self.theta) > 1.3963)	:  #handlebars cannot turn more than 80 degrees
            self.theta = numpy.sign(self.theta) * 1.3963

        #Compute new position of front tire
        temp = self.v*self.dt/(2*self.rf)
        sgn = numpy.sign(self.angleWrapPi(self.psi + self.theta))
        if (temp > 1):
            temp = sgn * self.M_PI/2
        else:
            temp = sgn * numpy.arcsin(temp)
        self.xf += self.v * self.dt * numpy.cos(self.psi + self.theta + temp)
        self.yf += self.v * self.dt * numpy.sin(self.psi + self.theta + temp)

        # Compute new position of back tire
        temp = self.v*self.dt/(2*self.rb)
        if (temp > 1):
            temp = numpy.sign(self.psi) * self.M_PI/2
        else:
            temp = numpy.sign(self.psi) * numpy.arcsin(temp)
        self.xb += self.v * self.dt * (numpy.cos(self.psi + temp))
        self.yb += self.v * self.dt * (numpy.sin(self.psi + temp))

        # Round off errors accumulate so the length of the bike changes over many iterations. The following take care of that by correcting the back wheel:
        temp = numpy.sqrt((self.xf-self.xb)*(self.xf-self.xb)+(self.yf-self.yb)*(self.yf-self.yb))
        if (abs(temp - self.l) > 0.001):
            self.xb += (self.xb-self.xf)*(self.l-temp)/temp
            self.yb += (self.yb-self.yf)*(self.l-temp)/temp
	

        self.psi = numpy.arctan2(self.yf - self.yb, self.xf - self.xb)
        self.psi_goal = self.calc_angle_to_goal(self.xf, self.xb, self.yf, self.yb)
        self.dtg = self.calc_dist_to_goal(self.xf, self.xb, self.yf, self.yb)

        if (abs(self.omega) > (self.M_PI/15)):
            # The bike has fallen over - it is more than 12 degrees from vertical
            self.shouldTerminate = True
            reward = ((-1.0 - self.M_PI*self.M_PI * 3.0 / 4.0) * 0.001) / (1.0 - 0.99)
    
        elif (self.dtg <= self.radius_goal):
            #The bike reached the goal
            reward = 1.0
            self.shouldTerminate = True
        else:
            reward = self.c_reward *(self.angleWrapPi(self.psi_t) - self.angleWrapPi(self.psi))

        self.lastdtg = self.dtg
        self.currentState[self.iomega] = self.omega
        self.currentState[self.iomega_dot] = self.omega_dot
        self.currentState[self.itheta] = self.theta
        self.currentState[self.itheta_dot] = self.theta_dot
        self.currentState[self.iangleGoal] = self.angleWrap(self.psi- self.psi_goal)
        self.normalizeState()
        
        return reward

    def angleWrap(self, x):
        while (x < 0):
    		x += 2.0*self.M_PI
        while (x > 2.0*self.M_PI):
    		x -= 2.0*self.M_PI
        return x
    
    
    def angleWrapPi(self, x):
        while (x < -self.M_PI):
    		x += 2.0*self.M_PI
        while (x > self.M_PI):
    		x -= 2.0*self.M_PI
        return x
    
    def calc_dist_to_goal(self, xf, xb, yf, yb):
        return numpy.sqrt((self.x_goal-xf)*(self.x_goal-xf) + (self.y_goal-yf)*(self.y_goal-yf))
    
    
    def calc_angle_to_goal(self, xf, xb, yf, yb):
        return numpy.arctan2(self.y_goal - yf, self.x_goal - xf)
    
    
    def angleToGoal(self, xf, xb, yf, yb):
    
        norm_x = numpy.sqrt(xf*xf + xb*xb)
        norm_y = numpy.sqrt(yf*yf + yb*yb)
        return numpy.arccos((xf*yf + xb*yb)/(norm_x * norm_y ))
    
    
    def isAtGoal(self):
        return self.dtg <= self.radius_goal
    
    def denormalize(self, v, min_, max_):
        ret = v/2. + 0.5	#agai between 0 and 1
        return ret*(max_-min_) + min_
    
    
    def normalize(self, v, min_, max_):
        min_out = -1.
        max_out = 1.
        temp = (v-min_)/(max_-min_)
        return  temp * (max_out-min_out) + min_out
    
    def normalizeState(self):
        self.currentState[self.itheta] = self.normalize(self.currentState[self.itheta],-1.3963,1.3963)
        self.currentState[self.itheta_dot] = self.normalize(self.currentState[self.itheta_dot],-9.40283,7.9616)
        self.currentState[self.iomega] = self.normalize(self.currentState[self.iomega],-0.209355,0.209291)
        self.currentState[self.iomega_dot] = self.normalize(self.currentState[self.iomega_dot],-1.60456,1.87696)
        self.currentState[self.iangleGoal] = self.normalize(self.currentState[self.iangleGoal],-3.1407,3.14151)
    
    
    def denormalizeState(self):
        self.currentState[self.itheta] = self.denormalize(self.currentState[self.itheta],-1.3963,1.3963)
        self.currentState[self.itheta_dot] = self.denormalize(self.currentState[self.itheta_dot],-9.40283,7.9616)
        self.currentState[self.iomega] = self.denormalize(self.currentState[self.iomega],-0.209355,0.209291)
        self.currentState[self.iomega_dot] = self.denormalize(self.currentState[self.iomega],-1.60456,1.87696)
        self.currentState[self.iangleGoal] = self.denormalize(self.currentState[self.iangleGoal],-3.1407,3.14151)
    
    
    def isAbsorbing(self):
        return self.shouldTerminate
        
    def getState(self):
        #return state already normalized
        return self.currentState
                