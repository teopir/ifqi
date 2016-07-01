import numpy

class Bicycle:

    def __init__(self, **kwargs):
        self.noise = kwargs.setdefault('noise', 0.04)
        self.random_start = kwargs.setdefault('random_start', False)
    
        self.state = numpy.zeros((5,)) # omega, omega_dot, omega_ddot, theta, theta_dot
        self.position = numpy.zeros((5,)) # x_f, y_f, x_b, y_b, psi
        self.state_range = numpy.array([[-numpy.pi * 12./180., numpy.pi * 12./180.],
                                        [-numpy.pi * 2./180., numpy.pi * 2./180.],
                                        [-numpy.pi, numpy.pi],
                                        [-numpy.pi * 80./180., numpy.pi * 80./180.],
                                        [-numpy.pi * 2./180., numpy.pi * 2./180.]])
        self.psi_range = numpy.array([-numpy.pi, numpy.pi])
    
        self.reward_fall = -1.0
        self.reward_goal = 0.01
        self.goal_rsqrd = 1000.0 # Square of the radius around the goal (10m)^2
        self.navigate = kwargs.setdefault('navigate', False)
        if not self.navigate:
            # Original balancing task
            self.reward_shaping = 0.001
        else:
            self.reward_shaping = 0.00004
    
        self.goal_loc = numpy.array([1000., 0])
        # Units in Meters and Kilograms
        self.c = 0.66       # Horizontal dist between bottom of front wheel and center of mass
        self.d_cm = 0.30    # Vertical dist between center of mass and the cyclist
        self.h = 0.94       # Height of the center of mass over the ground
        self.l = 1.11       # Dist between front tire and back tire at point on ground
        self.M_c = 15.0     # Mass of bicycle
        self.M_d = 1.7      # Mass of tire
        self.M_p = 60       # Mass of cyclist
        self.r = 0.34       # Radius of tire
        self.v = 10.0 / 3.6 # Velocity of bicycle (converted from km/h to m/s)
    
        # Useful precomputations
        self.M = self.M_p + self.M_c
        #SQRT
        self.Inertia_bc = (13./3.) * self.M_c * self.h**2 + self.M_p * (self.h + self.d_cm)**2
        self.Inertia_dv = self.M_d * self.r**2
        self.Inertia_dl = .5 * self.M_d * self.r**2
        self.sigma_dot = self.v / self.r
    
        # Simulation Constants
        self.gravity = 9.8
        self.delta_time = 0.01
        self.sim_steps = 1
        
        self.isAbs = False

    
    def setNewState(self,state, x1, y1, orientation):
        x2 = x1+numpy.cos(orientation) * self.l
        y2 = y1+numpy.sin(orientation) * self.l
        z = numpy.arctan((self.position[1]-self.position[0])/(self.position[2] - self.position[3]))
        self.setState(state + [x1,y1, x2, y2, z])
        
    #Azione ha intervallo da 1 a 9
    def step(self, intAction):
        T = 2. * ((int(intAction)/3) - 1) # Torque on handle bars
        d = 0.02 * ((intAction % 3) - 1) # Displacement of center of mass (in meters)
        print "T: ", T
        print "d: ", d        
        #print "azioni prese:" , T, d
        #if self.noise > 0:
          #d += (numpy.random.random()-0.5)*self.noise # Noise between [-0.02, 0.02] meters

        omega, omega_dot, omega_ddot, theta, theta_dot = tuple(self.state)
        x_f, y_f, x_b, y_b, psi = tuple(self.position)

        for step in range(self.sim_steps):
            if theta == 0: # Infinite radius tends to not be handled well
                r_f = r_b = r_CM = 9999999 #1.e8
            else:
                r_f = self.l / float(numpy.abs(numpy.sin(theta)))
                r_b = self.l / float(numpy.abs(numpy.tan(theta)))
                r_CM = numpy.sqrt((self.l - self.c)**2 + (self.l**2 / numpy.tan(theta)**2))

            varphi = omega + numpy.arctan(d / self.h)
            
            #print "varphi: ", varphi
    

            omega_ddot = self.h * self.M * self.gravity * numpy.sin(varphi)
            
            omega_ddot -= numpy.cos(varphi) * (self.Inertia_dv * self.sigma_dot * theta_dot + numpy.sign(theta)*self.v**2*(self.M_d * self.r *(1./r_f + 1./r_b) + self.M*self.h/r_CM))
            omega_ddot /= self.Inertia_bc

            #print "omega_ddot: ", omega_ddot
            theta_ddot = (T - self.Inertia_dv * self.sigma_dot * omega_dot) / self.Inertia_dl

            #print "theta_ddot: ", theta_ddot
            df = (self.delta_time / float(self.sim_steps))
            omega_dot += df * omega_ddot
            omega += df * omega_dot
            theta_dot += df * theta_ddot
            theta += df * theta_dot

            #print "omega_dot: ", omega_dot
            #print "   omega    : ", omega
            #print "      theta_dot: ", theta_dot
            #print "         theta    : ", theta
            
            # Handle bar limits (80 deg.)
            theta = numpy.clip(theta, self.state_range[3,0], self.state_range[3,1])

            #print "theta: ", theta
            # Update position (x,y) of tires
            front_term = psi + theta + numpy.sign(psi + theta)*numpy.arcsin(self.v * df / (2.*r_f))
            back_term = psi + numpy.sign(psi)*numpy.arcsin(self.v * df / (2.*r_b))
           
            #print "front_term: ", front_term
            #print "    back_term: ", back_term
            
            x_f += -numpy.sin(front_term)
            y_f += numpy.cos(front_term)
            x_b += -numpy.sin(back_term)
            y_b += numpy.cos(back_term)
            
            

            # Handle Roundoff errors, to keep the length of the bicycle constant
            dist = numpy.sqrt((x_f-x_b)**2 + (y_f-y_b)**2)
            if numpy.abs(dist - self.l) > 0.01:
                x_b += (x_b - x_f) * (self.l - dist)/dist
                y_b += (y_b - y_f) * (self.l - dist)/dist

            #print "x_b: ", x_b
            #print "    y_b: ", y_b
            
            # Update psi
            if x_f==x_b and y_f-y_b < 0:
                psi = numpy.pi
            elif y_f - y_b > 0:
                psi = numpy.arctan((x_b - x_f)/(y_f - y_b))
            else:
                psi = numpy.sign(x_b - x_f)*(numpy.pi/2.) - numpy.arctan((y_f - y_b)/(x_b-x_f))

        #print "(" , omega , omega_dot, omega_ddot, theta, theta_dot, " )"
        self.state = numpy.array([omega, omega_dot, omega_ddot, theta, theta_dot])
        self.position = numpy.array([x_f, y_f, x_b, y_b, psi])

        if numpy.abs(omega) > self.state_range[0,1]: # Bicycle fell over
            self.isAbs = True
            return -1.0
        elif self.isAtGoal():
            self.isAbs = True
            return self.reward_goal
        elif not self.navigate:
            self.isAbs = False
            return self.reward_shaping
        else:
            goal_angle = self.vector_angle(self.goal_loc, numpy.array([x_f-x_b, y_f-y_b])) * numpy.pi / 180.
            #print "goal_angle: ", goal_angle            
            self.isAbs = False
            return (4. - goal_angle**2) * self.reward_shaping

    def isAbsorbing(self):
        return self.isAbs
        
    def setState(self, state):
        self.state = numpy.array(state[0:5])
        self.position = numpy.array(state[5:10])

    def isAtGoal(self):
        # Anywhere in the goal radius
        if self.navigate:
            return numpy.sqrt(max(0.,((self.position[:2] - self.goal_loc)**2).sum() - self.goal_rsqrd)) < 1.e-5
        else:
            return False
            
    def vector_angle(self,u, v):
        return numpy.arccos(numpy.dot(u,v)/(numpy.linalg.norm(u)*numpy.linalg.norm(v)))*180.0/numpy.pi
        
    def actionOverState(self, state, action):
        self.setState(state)
        reward, fine = self.takeAction(action)
        return self.state.tolist() +  self.position.tolist() , action, reward, fine
        
    def generateState(self):
        rnd_state = [0,0,0,0,0]
        rnd_pos = [0,0,0,0,0]
        #genero uno stato abbastanza casuale, che esca anche dalle condizioni di stabilità
        for i in range(0,5):
            rnd_state[i] = self.state_range[i,0] + numpy.random.rand() * (self.state_range[i,1] -self.state_range[i,0] )
        for i in range(0,5):
            rnd_pos[i] = self.psi_range[0] + numpy.random.rand() * (self.psi_range[1] -self.psi_range[0])
        return rnd_state + rnd_pos
    
    def getState(self):
        x_f, y_f, x_b, y_b, psi = tuple(self.position)
        return self.state.tolist()  +  [self.vector_angle(self.goal_loc, numpy.array([x_f-x_b, y_f-y_b])) * numpy.pi / 180.]
        
    def reset(self):
        self.isAbs = False
        self.state.fill(0.0)
        self.position.fill(0.0)
        self.position[3] = self.l
        self.position[4] = numpy.arctan((self.position[1]-self.position[0])/(self.position[2] - self.position[3]))

    def episode(self, actor):
        self.state=numpy.zeros((5,))
        self.reset()
        ret = []
        for i in range(0,1000):
            #s = self.getState()
            s1, a, r, f = self.actionOverState(self.getState(),actor.exploitAction(self.getState()))
            #ret.append([s, a, r, s1, f, self.isAtGoal()])
            ret.append(r)
            if(f):
                break
        return ret
            
