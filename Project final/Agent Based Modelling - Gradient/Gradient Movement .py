from mesa import Agent, Model 
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from matplotlib.colors import colorConverter


gradient_environment = np.zeros((100,100), dtype = np.uint8)
gradient_environment[:] = np.arange(100)
gradient_environment = np.rot90(gradient_environment) 

# Specifying the number of agents and the number of steps they take
num_of_agents = 5
N = num_of_agents
num_of_steps = 49



# Variable is ready to collect the data from self.pos
apath = []
xx = []
yy = []



class AgentClass(Agent): 
    """An agent with fixed initial wealth."""
    
    # def is how you create a function. A function in a class describes the objects behaivour of the class
    #__init__ sets the initial values of an object. It does not need to be called to be initialised
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
   
    
    def decider(self, Environment, position):
        i , j = position
        
        # retrieves the cell neighbors 
        region = Environment[max(0,i-1) : i+2,max(0,j-1) : j+2] 
        current = region[1,1]
     
             
        # This line decides if the agents want to move to larger values or smaller values 
        PossibleSteps = np.where(region > current )
        row, column = PossibleSteps
        row_len =  row.shape
        rand = row_len[0] 

        # A posssible step is chosen at random 
        x = random.sample(range(rand),1)
        np.array(x)
        x = x[0]

        # The index selected has its colunm and row data assigned to yy and xx
        yy = column[x]
        xx =    row[x]
        
        # The following code dictates what is returned when the function is executed
        if xx == 0 and yy == 0:
            return 2
        elif xx == 0 and yy == 1:
            return 4
        elif xx == 0 and yy == 2:
            return 7
        elif xx== 1 and yy== 0:
            return 1
        elif xx == 1 and yy == 2:
            return 6
        elif xx == 2 and yy == 0:
            return 0
        elif xx == 2 and yy == 1:
            return 3
        elif xx == 2 and yy == 2:
            return 5
    
    # The following function allows the agent to choose its new position on the grid
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
        self.pos,
        moore=True, 
        include_center=False)
        
        # This is where the tuple is selected in "possible_steps" and assigned to "new_position"
        new_position = possible_steps[self.decider(gradient_environment,self.pos)]
        # Move agent executes the movement of the agent on the grid
        self.model.grid.move_agent(self, new_position)

    
    # This function saves the path of the agents to the apath list
    def agentpath(self):
        apath.append(self.pos)    
        
    # Step is called for each agent and activates the "move" and "agentpath" functions
    def step(self):
        self.move()
        self.agentpath()
        
        

# The object in this class is the ModelClass
class ModelClass(Model):
    """A model with some number of agents."""
    
    def __init__(self, N, Grid):
        # num_agents is a parameter and stays constant throughout the simulation
        self.num_agents = N
        self.width = Grid.shape[1]
        self.height = Grid.shape[0] 
   
        # Adds the grid to the model 
        self.grid = MultiGrid(self.width, self.height, True)
        
        # The schedule is re-shuffled for each itteration of the model  
        self.schedule = RandomActivation(self)
        
        # Creates the agents and adds them to the schedule 
        for i in range(self.num_agents):
            a = AgentClass(i, self)
            self.schedule.add(a)
            # Specifies where the agent starts on the grid
            self.grid.place_agent(a, (50,50))


    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()

# Calls the class and the init function 
model1 = ModelClass(num_of_agents , gradient_environment)

# Runs the model for as many times as stated in num_of_steps
for i in range(num_of_steps):
    model1.step()

Apath = np.array(apath)

# print(Apath.shape)
agent1y = Apath[:,0]
agent1x = Apath[:,1]
agent1pathplot = agent1y ,agent1x
pathplot = np.zeros((100,100), dtype = np.uint8)
pathplot[agent1pathplot] = 1

# create dummy data
zvals = gradient_environment
zvals2 = pathplot

# generate the colors for your colormap
color1 = colorConverter.to_rgba('white')
color2 = colorConverter.to_rgba('red')

# make the colormaps
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black','white'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init()
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas


img2 = plt.imshow(zvals, interpolation='nearest', cmap=cmap1, origin='lower')
img3 = plt.imshow(zvals2, interpolation='nearest', cmap=cmap2, origin='lower')

plt.show()

