#
# This Agent demonstrates use of a Keras centred Q-network estimating the Q[S,A] Function from a few basic Features
#
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# as employed against OpenAI  Gym Cart Pole examples
#  requires keras [and hence Tensorflow or Theono backend]
# ==============================================================================
import random, numpy, math, pickle
import sys
#
import keras.callbacks
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
#
#%% ==========================================================================
#  Keras based Nueral net Based Brain Class
class Brain:
        def __init__(self, NbrStates, NbrActions):
                self.NbrStates = NbrStates
                self.NbrActions = NbrActions

                self.model = self._createModel()
               # self.tbCallBack = keras.callbacks.TensorBoard(log_dir='./log')

        def _createModel(self):
                model = Sequential()

                # Simple Model with Two Hidden Layers and a Linear Output Layer. The Input layer is simply the State input.
                model.add(Dense(units=64, activation='relu', input_dim=self.NbrStates))
                model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=self.NbrActions, activation='linear'))                            # Linear Output Layer as we are estimating a Function Q[S,A]

                model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment

                return model

        def train(self, x, y, epoch=1, verbose=0):
                
                self.model.fit(x, y, batch_size=256, epochs=epoch, verbose=verbose)#, callbacks = [self.tbCallBack],validation_split = 0.2)

        def predict(self, s):
		
                return self.model.predict(s)

        def predictOne(self, s):
		
                return self.predict(s).flatten()

# =======================================================================================
# A simple Experience Replay memory
#  DQN Reinforcement learning performs best by taking a batch of training samples across a wide set of [S,A,R, S'] expereiences
#
class ExpReplay:   # stored as ( s, a, r, s_ )
        samples = []

        def __init__(self, capacity):
                self.capacity = capacity

        def add(self, sample):
                self.samples.append(sample)

                if len(self.samples) > self.capacity:
                        self.samples.pop(0)

        def sample(self, n):
                n = min(n, len(self.samples))
                return random.sample(self.samples, n)

# =====================================================================================
# DQN Reinforcement Learning Algorithm  Hyper Parameters
ExpReplay_CAPACITY = 2000
OBSERVEPERIOD = 4100             # Period actually start real Training against Experieced Replay Batches
BATCH_SIZE = 4096
GAMMA = 0.95                            # Q Reward Discount Gamma
MAX_EPSILON = 1
MIN_EPSILON = 0.05
LAMBDA = 0.0001     
P_BATCH_SIZE = 128            # Speed of Epsilon decay
EXP_PROCESS_RATIO = 16
# ============================================================================================
class Agent:
        def __init__(self, NbrStates, NbrActions):
                self.NbrStates = NbrStates
                self.NbrActions = NbrActions

                self.brain = Brain(NbrStates, NbrActions)
                self.ExpReplay = ExpReplay(ExpReplay_CAPACITY)
                self.steps = 0
                self.epsilon = MAX_EPSILON
		self.ExpCount = 0
        # ============================================
	def Load(self, num):
		self.brain.model = keras.models.load_model("./model/testModel_MBSub{}".format(num))
                with open("./model/testModel_MBSub{}.steps".format(num),"r") as f:
                        self.steps = int(f.read())
                        print(self.steps)
                with open("./model/testModel_MBSub{}.exp".format(num),"rb") as f:
                        self.ExpReplay.samples =  pickle.load(f)
                        
                        
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * (self.steps-OBSERVEPERIOD))
        # ============================================
	def Save(self, num):
                self.brain.model.save("./model/testModel_MBSub{}".format(num))
                with open("./model/testModel_MBSub{}.steps".format(num),"w") as f:
                        f.write('{}'.format(num))
                        print("WriteSub!")
                        print(num)
                        
                with open("./model/testModel_MBSub{}.exp".format(num),"wb") as f:        
                        pickle.dump(self.ExpReplay.samples,f)

        # ============================================
        # Return the Best Action  from a Q[S,A] search.  Depending upon an Epslion Explore/ Exploitaiton decay ratio
        def ballysub(self, s):
		#print 1111
                if (self.steps < OBSERVEPERIOD): #random.random() < self.epsilon or 
			#print "rand"
                        return random.uniform(0, 1)                                             # Explore
              
		else:
			return self.brain.predictOne([s[2],s[0]])
			
                                  # Exploit Brain best Prediction

        # ============================================
        def CaptureSample(self, sample):  # in (s, a, r, s_) format
                self.ExpReplay.add(sample)

                # slowly decrease Epsilon based on our eperience
                self.steps += 1
                if(self.steps>OBSERVEPERIOD):
                        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * (self.steps-OBSERVEPERIOD))

        # ============================================
        # Perform an Agent Training Cycle Update by processing a set of sampels from the Experience Replay memory
        def Process(self):
		self.ExpCount += 1
		if(self.ExpCount%EXP_PROCESS_RATIO!=0):
                        return 0
                batch =  self.ExpReplay.samples[-BATCH_SIZE:] 
                batchLen = len(batch)
		
                no_state = numpy.zeros(self.NbrStates)

                states = numpy.zeros((BATCH_SIZE,2))

		states_ = numpy.zeros((BATCH_SIZE,2))
		for i in range(batchLen):
			states[i][0] = batch[i][0][2]
			states[i][1] = batch[i][0][0]
			states_[i][0] = 0 if batch[i][3] is None else batch[i][3][2]
			states_[i][1] = 0 if batch[i][3] is None else batch[i][3][0]
			


                
                predictedQ = self.brain.predict(states)        

		                                 # Predict from keras Brain the current state Q Value
                predictedNextQ = self.brain.predict(states_)                            # Predict from keras Brain the next state Q Value		
		

                x = []
                y = []

                #  Now compile the Mini Batch of [States, TargetQ] to Train an Target estimator of Q[S,A]	
		#print batch[0][0][6], "speed"
	
                for i in range(batchLen-1):
			#print batch[i][0][6], "speed", batch[i][0][4], "direction",batch[i][0][2] , "ball_X_position"
			if((batch[i+1][0][3]*batch[i+1][0][3]>batch[i][0][3]*batch[i][0][3] and batch[i+1][0][1]<=0.1 and batch[i+1][0][3]==(-1)*batch[i][0][3]*1.05 and batch[i+1][0][4]==batch[i][0][4]*1.05) or((batch[i+1][0][3]*batch[i+1][0][3])<(batch[i][0][3]*batch[i][0][3])and batch[i][0][1]<=0.1 and batch[i+1][0][1]>=0.45 and batch[i+1][0][1]<=0.55)):
# ):
				#print 99999999999999999999999
                        	batchitem = batch[i]
                        	state = (batchitem[0][2],batchitem[0][0]); a = batchitem[1]; reward = 1/(float)(1+(predictedQ[i]-states_[i])*(predictedQ[i]-states_[i])); nextstate = (batchitem[3][2],batchitem[3][0])

                       	 	targetQ = reward
	                        x.append([batch[i][0][2],batch[i][0][0]])
        	                y.append(batch[i+1][0][2])
		#if(batchLen>=128): 
		#	sys.exit(0)
		
		if(len(x)!=0):
		 print(i)
                 self.brain.train(x, y)                                          #  Call keras DQN to Train against the Mini Batch set
# =======================================================================
