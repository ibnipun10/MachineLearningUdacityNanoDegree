import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.1    # learning rate
        self.gamma = 0.4    # future rewards rate
        self.actions = ('left', 'right', 'forward', None)
        self.lights = ('red','green')
        self.epsilon = .1
        #self.headings = { '(1, 0)':'east', '(0, -1)':'north', '(-1, 0)':'west', '(0, 1)':'south'}
        #self.headingIndex = ('east','north','west','south')
        self.NoneCtr = 0
        self.maxConsequtiveNone = 3
        self.trials = 0
        self.deadlinesCount = 0        
        
        # action : {left, right, forward, none}
        # light :  { red, green }
        # traffic :  { ongoing, left, right}
        # traffic directions :  { left, right, forward, none}
        # waypoint : { left, right, forward }
        self.qMatrix = np.zeros([4,2,4,4,4,3], dtype=float)
        self.unlearnedStatesCount = self.qMatrix.size

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trials = self.trials + 1
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.ReduceEpsilon()
        
    def getIndexInMatrix(self, value, listforSearch):
        for index, val in enumerate(listforSearch):
            if(value == val):
                return index
    
    def getHeadingIndexInMatrix(self, state):
        heading = self.headings[str(state['heading'])]
        headingIndex = self.getIndexInMatrix(heading, self.headingIndex)
        return headingIndex
        
    def getQMatrixPos(self, inputs, state): 
        
        light = self.getIndexInMatrix(inputs['light'], self.lights)
        oncoming = self.getIndexInMatrix(inputs['oncoming'], self.actions)
        right = self.getIndexInMatrix(inputs['right'], self.actions)
        left = self.getIndexInMatrix(inputs['left'], self.actions)
        waypoint = self.getIndexInMatrix(self.next_waypoint, self.actions)   

        return (light, oncoming, left, right, waypoint)
    
    def IncreaseFutureRewards(self):        
        if self.gamma < 0.90:
            self.gamma = self.gamma + .025
    
    def ReduceEpsilon(self):
        #So once the agent has learned why do we need such a high epsilon
        self.unlearnedStatesCount = self.unlearnedStatesCount - self.deadlinesCount
        
        if self.unlearnedStatesCount < 0:
            self.epsilon = 0
        
        self.deadlinesCount = (self.env.agent_states[self])['deadline']
        
        print "epsilon : ", self.epsilon, " unlearnedStates count : ", self.unlearnedStatesCount
        
        
    def checkIndexinarray(self,index, array):
        for val in np.nditer(array):
            if val == index:
                return True
        
        return False
    
    def getMaxQvalueAction(self, inputs, state, current = True):
        matrixPos  = self.getQMatrixPos(inputs, state)
        action = self.actions[0]
        
        #light, oncoming, left, right, waypoint
        Qactions = self.qMatrix[:,matrixPos[0], matrixPos[1],matrixPos[2],matrixPos[3],matrixPos[4]]
        
        if current is True and (random.random() < self.epsilon):
            action = random.choice(self.actions)            
        else:
            maxIndexes = np.where(Qactions == Qactions.max())[0]
            
            # if it gets multiple same Qvalues and if waypoint is there as an action then choose that.
            waypoint = self.getIndexInMatrix(self.next_waypoint, self.actions)
            if current is True and self.checkIndexinarray(waypoint, maxIndexes):              
                action = self.actions[waypoint]
                print "waypoint : ", waypoint, " maxIndexes ", maxIndexes               
            else:
                randIndex = random.choice(maxIndexes)
                action = self.actions[randIndex]
            
        return action 
    
    def chooseAction(self, inputs, state):
        action = self.getMaxQvalueAction(inputs, state)         
        return action
         
    def learnQvalue(self, state1, input1, action1, reward, state2, input2):
            
        initalagentPos = self.getQMatrixPos(input1, state1)
        
        actionPos = self.getIndexInMatrix(action1, self.actions)
  
        #old q Value
        oldqValue = self.qMatrix[actionPos, initalagentPos[0],initalagentPos[1],initalagentPos[2],initalagentPos[3],initalagentPos[4]]
        
        
        #get the next q value
        nextagentPos = self.getQMatrixPos(input2, state2)
        
        tmaxAction = self.getMaxQvalueAction(input2, state2, False)
        actionPos1 = self.getIndexInMatrix(tmaxAction, self.actions)
        nextqValue = self.qMatrix[actionPos1, nextagentPos[0],nextagentPos[1],nextagentPos[2],nextagentPos[3],nextagentPos[4]]
        
        #get the new q value
        newqValue = oldqValue + self.alpha * ( reward + (self.gamma * (nextqValue)) - oldqValue)
        
        
        #set the new q value in the matrix
        self.qMatrix[actionPos, initalagentPos[0],initalagentPos[1],initalagentPos[2],initalagentPos[3],initalagentPos[4]] = newqValue 
        print "action, waypoint, location, and value ", action1, self.next_waypoint,  initalagentPos[0],initalagentPos[1],initalagentPos[2],initalagentPos[3],initalagentPos[4], " old value ",  oldqValue, "new value ", newqValue
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self).copy()
        deadline = self.env.get_deadline(self)
        self.state = self.env.agent_states[self]

        # TODO: Update state     
        state = self.env.agent_states[self].copy()               
        
        # TODO: Select action according to your policy
        action = self.chooseAction(inputs, state)
        
        print state
        
        
        # Execute action and get reward
        reward = self.env.act(self, action)
       
        # TODO: Learn policy based on state, action, reward
        #new state
        newState = self.env.agent_states[self]
        #new Input
        newInputs = self.env.sense(self)
        self.learnQvalue(state, inputs, action, reward, newState, newInputs)
        self.state = newState
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
