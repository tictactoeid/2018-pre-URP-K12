#
# This Agent demonstrates use of a Keras centred Q-network estimating the Q[S,A] Function from a few basic Features
#
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# as employed against OpenAI  Gym Cart Pole examples
#  requires keras [and hence Tensorflow or Theono backend]
# ==============================================================================
import random, numpy, math, pickle
#
import keras.callbacks
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
# =====================================================================================
# DQN Reinforcement Learning Algorithm  Hyper Parameters
ExpReplay_CAPACITY = 20000
OBSERVEPERIOD = 20000             # Period actually start real Training against Experieced Replay Batches
BATCH_SIZE = 2048
GAMMA = 0.995                            # Q Reward Discount Gamma
MAX_EPSILON = 1
MIN_EPSILON = 0.05
LAMBDA = 0.00001                 # Speed of Epsilon decay
EXP_PROCESS_RATIO = 2000
# Profiling Parameters
P_BATCH_SIZE = 2048
#%% ==========================================================================
#  Keras based Nueral net Based Brain Class
class Brain:
        def __init__(self, NbrStates, NbrActions):
                self.NbrStates = NbrStates
                self.NbrActions = NbrActions

                self.model = self._createModel()
                self.tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=P_BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        def _createModel(self):
                model = Sequential()

                # Simple Model with Three Hidden Layers and a Linear Output Layer. The Input layer is simply the State input.
                # model.add(Conv1D(filters = 32, kernel_size = 3, strides = 1))
                model.add(Dense(units=64, activation='relu', input_dim=self.NbrStates))
                model.add(Dense(units=32, activation='relu'))
                # model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=self.NbrActions, activation='linear'))                            # Linear Output Layer as we are estimating a Function Q[S,A]

                model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))     # use adam as an alternative optimsiuer as per comment

                return model

        def train(self, x, y, epoch=1, verbose=0):
                self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose, initial_epoch = epoch-1)#, callbacks = [self.tbCallBack],validation_split = 0.2)

        def predict(self, s):
                return self.model.predict(s)

        def predictOne(self, s):
                return self.predict(s.reshape(1, self.NbrStates)).flatten()

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

#self.numpy.array([ batchitem[0][2] for batchitem in self.ExpReplay.samples[-128:] ])
# ============================================================================================
class Agent:
        def __init__(self, NbrStates, NbrActions):
                self.NbrStates = NbrStates
                self.NbrActions = NbrActions
                self.ExpCount = 0
                self.brain = Brain(NbrStates, NbrActions)
                self.ExpReplay = ExpReplay(ExpReplay_CAPACITY)
                self.steps = 0
                self.epsilon = MAX_EPSILON
        # ============================================
        def Load(self, num):
                self.brain.model = keras.models.load_model("./model/testModel{}".format(num))
                with open("./model/testModel{}.steps".format(num),"r") as f:
                        self.steps = int(f.read())
                        print(self.steps)
                with open("./model/testModel{}.exp".format(num),"rb") as f:
                        self.ExpReplay.samples =  pickle.load(f)
                        
                        
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * (self.steps-OBSERVEPERIOD))
        # ============================================
        def Save(self, num):
                self.brain.model.save("./model/mf_mfModel{}".format(num))
                with open("./model/mf_mfModel{}.steps".format(num),"w") as f:
                        f.write('{}'.format(self.steps))
                        print("Write!")
                        print(self.steps)
                        
                with open("./model/mf_mfModel{}.exp".format(num),"wb") as f:        
                        pickle.dump(self.ExpReplay.samples,f)
                        
                        #학습 중간에 멈췄다가 로딩할 때만 필요
                        #Python 2.7로 save된 model을 Python 3.x에서 load하면 오류가 생길 수 있음.
                        #버전의 차이로 오류가 생기는 부분은 여기뿐임
        # ============================================
        # Return the Best Action  from a Q[S,A] search.  Depending upon an Epslion Explore/ Exploitaiton decay ratio
        def Act(self, s):
                if (random.random() < self.epsilon or self.steps < OBSERVEPERIOD):
                        return random.randint(0, self.NbrActions-1)                                             # Explore
                else:
                        return numpy.argmax(self.brain.predictOne(s))                                   # Exploit Brain best Prediction

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
                self.ExpCount+=1
                if(self.ExpCount%EXP_PROCESS_RATIO!=0):
                        return 0
               
        
                batch = self.ExpReplay.sample(BATCH_SIZE)
                batchLen = len(batch)

                no_state = numpy.zeros(self.NbrStates)

                states = numpy.array([ batchitem[0] for batchitem in batch ])
                states_ = numpy.array([ (no_state if batchitem[3] is None else batchitem[3]) for batchitem in batch ])

                predictedQ = self.brain.predict(states)                                         # Predict from keras Brain the current state Q Value
                predictedNextQ = self.brain.predict(states_)                            # Predict from keras Brain the next state Q Value

                x = numpy.zeros((batchLen, self.NbrStates))
                y = numpy.zeros((batchLen, self.NbrActions))

                #  Now compile the Mini Batch of [States, TargetQ] to Train an Target estimator of Q[S,A]
                for i in range(batchLen):
                        batchitem = batch[i]
                        state = batchitem[0]; a = batchitem[1]; reward = batchitem[2]; nextstate = batchitem[3]

                        targetQ = predictedQ[i]
                        if nextstate is None:
                                targetQ[a] = reward                                                                                             # An End state Q[S,A]assumption
                        else:
                                targetQ[a] = reward + GAMMA * numpy.amax(predictedNextQ[i])     # The core Q[S,A] Update recursive formula

                        x[i] = state
                        y[i] = targetQ

                self.brain.train(x, y, self.ExpCount)                                          #  Call keras DQN to Train against the Mini Batch set
# =======================================================================
