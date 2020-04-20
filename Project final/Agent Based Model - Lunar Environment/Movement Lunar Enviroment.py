from mesa import Agent, Model 
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from matplotlib.colors import colorConverter

# The enviroment where the agents exist 
crop = np.load('crop2.npy')
crop = np.flip(crop,0)


# Specifying the number of agents and the number of steps they take
num_of_agents = 10
N = num_of_agents
num_of_steps = 10


apath = []
yy = []
xx = []



class AgentClass(Agent): 
    """An agent with fixed initial wealth."""
    
    # def is how you create a function. A function in a class describes the objects behaivour of the class
    #__init__ sets the initial values of an object. It does not need to be called to be initialised
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
   
    
    def decider(self, ModelEnviroment, position):
        j , i = position
         # retrieves the cell neighbors 
        region = ModelEnviroment[max(0,j-1) : j+2,max(0,i-1) : i+2]        
       
        # This line decides if the agents want to move to larger values or smaller values 
        PossipleSteps = np.where(region ==1 )
        stepsj, stepsi = PossipleSteps
        stepsarrayi1 = stepsj.shape
        rand = stepsarrayi1[0] 
        x = random.sample(range(rand),1)
        np.array(x)
        x = x[0]
        xx = stepsi[x]
        yy = stepsj[x]
        
       
        while yy == 1 and xx == 1:
            x = random.sample(range(rand),1)
            x = x[0]
            xx = stepsi[x]
            yy = stepsj[x]

            if yy != 1 or xx !=1:
                break 

        

        if yy == 0 and xx == 0:
            return 2
        elif yy == 0 and xx == 1:
            return 4
        elif yy == 0 and xx == 2:
            return 7
        elif yy == 1 and xx == 0:
            return 1
        elif yy == 1 and xx == 2:
            return 6
        elif yy == 2 and xx == 0:
            return 0
        elif yy == 2 and xx == 1:
            return 3
        elif yy == 2 and xx == 2:
            return 5
    
          


    # The following function allows the agent to choose its new position on the grid
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
        self.pos,
        moore=True, 
        include_center=False)

         # This is where the number of the cell will be inputted
        new_position = possible_steps[self.decider(crop, self.pos)]
        self.model.grid.move_agent(self, new_position)

    
    # This function saves the path of the agents to the apath variable
    def agentpath(self):
        apath.append(self.pos)    
        
    
    def step(self):
        self.move()
        self.agentpath()
        #print(self.decider(crop, self.pos))
        


class ModelClass(Model):
    """A model with some number of agents."""
    
    def __init__(self, N, elevation):
        # num_agents is a parameter and stays constant throughout the simulation
        self.num_agents = N
        self.width = elevation.shape[0]
        self.height = elevation.shape[1]
        self.z = elevation   
        # Adds the grid to the model object 
        self.grid = MultiGrid(self.width, self.height, True)
        # Adds the scheduler to the model object 
        self.schedule = RandomActivation(self)
        
        # create agents (for loop hence multiple agents) and add them to the schedular (one at a time) 
        for i in range(self.num_agents):
            # The money agent class (object) is being computed for every value of i
            a = AgentClass(i, self)
            self.schedule.add(a)
            # Add the agents to a random cell 
            # x = self.random.randrange(self.width)
            # y = self.random.randrange(self.height)
            self.grid.place_agent(a, (54,30))


    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()

# Calls the class and the init function 
model1 = ModelClass(num_of_agents , crop)

# Runs the model for as many times as stated in num_of_steps
for i in range(num_of_steps):
    model1.step()

Apath = np.array(apath)

# print(Apath.shape)
agent1y = Apath[:,0]
agent1x = Apath[:,1]
agent1pathplot = agent1y ,agent1x
pathplot = np.zeros((75,75), dtype = np.uint8)
pathplot[agent1pathplot] = 1
np.save('pathLE.npy', pathplot)


zvals = crop
zvals2 = pathplot

# generate the colors for your colormap
color1 = colorConverter.to_rgba('white')
color2 = colorConverter.to_rgba('red')

# make the colormaps
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init()
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas


img2 = plt.imshow(zvals, interpolation='nearest', cmap=cmap1, origin='lower')
img3 = plt.imshow(zvals2, interpolation='nearest', cmap=cmap2, origin='lower')

plt.show()
