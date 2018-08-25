#
# This Agent demonstrates use of a Keras centred Q-network estimating the Q[S,A] Function from a few basic Features
#
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# as employed against OpenAI  Gym Cart Pole examples
#  requires keras [and hence Tensorflow or Theono backend]
# ==============================================================================
import random, numpy, math
import MyPongfi
class Agent:
        def __init__(self, NbrStates, NbrActions):
                self.NbrStates = NbrStates
                self.NbrActions = NbrActions

                self.steps = 0
        
        # ============================================
        # Return the Best Action  from a Q[S,A] search.  Depending upon an Epslion Explore/ Exploitaiton decay ratio
        def Act(self, s):
                if(s[0]>s[2]+float(MyPongfi.BALL_HEIGHT)/MyPongfi.WINDOW_HEIGHT):
                        return 1
                
                if(s[0]+float(MyPongfi.PADDLE_HEIGHT)/MyPongfi.WINDOW_HEIGHT<s[2]):
                        return 2
                return 0
        # ============================================
        def CaptureSample(self, sample):  # in (s, a, r, s_) format
                doNothing = 0
        # ============================================
        # Perform an Agent Training Cycle Update by processing a set of sampels from the Experience Replay memory
        def Process(self):
                doNothing = 0
