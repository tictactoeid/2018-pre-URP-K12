#
#  MyPong DQN Reinforcement Learning Experiment
#
#  Plays Pong Game (DQN control of Left Hand Yellow Paddle)
#  Objective is simply measured as succesfully returning of the Ball 
#  The programed oponent player is a pretty hot player. Imagine success as being able to return ball served from Serena Williams)
#  Moving Average Score from [-10, +10] from Complete failure to return the balls, to full success in returning the Ball.
#
#  This Reinforcement learning employs Direct Features [ Paddle Y, Ball X, Y and Ball X,Y Directions feeding into DQN Nueral net Estimator of Q[S,A] function.
#  So this is NOT a Convolutional Network based RL, based Game Video Frame states [Which in my experience takes much too Long to Learn on standard PCs] 
#  So unfortunaly this is Game Specific DQN Reinforecment Learning, and cannot be generalised to other games. Requires specific Features to be identified. 
#      
# This experiment demonstrates DQN based agent, improves from poor performace ~ 5.0 towards reasonably good +8.0 (Fluctuating) 
# return rate in around 15,000 game cycles [Not returns]
#
#  The  code is based upon Siraj Raval's inspiring vidoes on Machine learning and Reinforcement Learning [ Which is full convolutional DQN example] 
#  https://github.com/llSourcell/pong_neural_network_live
# 
#  requires pygame, numpy, matplotlib, keras [and hence Tensorflow or Theono backend] 
# ==========================================================================================
import MyPongMBMF # My PyGame Pong Game 
import MyAgentMB
import MyAgentMF # My DQN Based Agent

import numpy as np 
import random 
import matplotlib.pyplot as plt
import MyAgentMBsub_brain
import csv

#
# =======================================================================
#   DQN Algorith Paramaters 
ACTIONS = 3 # Number of Actions.  Acton istelf is a scalar:  0:stay, 1:Up, 2:Down
STATECOUNT = 5 # Size of State [ PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection, OpponentYPos] 
TOTAL_GAMETIME = 1000000000000000000000000000000000000000000000000000

# =======================================================================
# Normalise GameState

TheAgentsub1 = MyAgentMBsub_brain.Agent(2, 1)

def CaptureNormalisedState1(PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed):

        gstate1 = np.zeros([STATECOUNT])
        gstate1[0] = PlayerYPos/400.0    # Normalised PlayerYPos
        gstate1[1] = BallXPos/400.0      # Normalised BallXPos
        gstate1[2] = BallYPos/400.0      # Normalised BallYPos
        gstate1[3] = BallXDirection*BallSpeed  # Normalised BallXDirection
        gstate1[4] = BallYDirection*BallSpeed  # Normalised BallYDirection

        gstate = np.zeros([STATECOUNT+1])
        gstate[0] = PlayerYPos/400.0    # Normalised PlayerYPos
        gstate[1] = BallXPos/400.0      # Normalised BallXPos
        gstate[2] = BallYPos/400.0      # Normalised BallYPos
        gstate[3] = BallXDirection*BallSpeed  # Normalised BallXDirection
        gstate[4] = BallYDirection*BallSpeed  # Normalised BallYDirection
        gstate[5] = TheAgentsub1.ballysub(gstate1)

        return gstate

def CaptureNormalisedState2(PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed):
        gstate = np.zeros([STATECOUNT])
        gstate[0] = PlayerYPos/400.0    # Normalised PlayerYPos
        gstate[1] = (MyPongMBMF.WINDOW_WIDTH - BallXPos + MyPongMBMF.BALL_WIDTH)/400.0      # Normalised BallXPos
        gstate[2] = BallYPos/400.0      # Normalised BallYPos
        gstate[3] = -BallXDirection*BallSpeed  # Normalised BallXDirection
        gstate[4] = BallYDirection*BallSpeed  # Normalised BallYDirection

        return gstate



# =====================================================================
# Main Experiment Method 
def PlayExperiment():
        GameTime = 0
    
        GameHistory = []
        
        #Create our PongGame instance
        TheGame = MyPongMBMF.PongGame()
    # Initialise Game
        TheGame.InitialDisplay()
        #
        #  Create our Agent (including DQN based Brain)
        TheAgent1 = MyAgentMB.Agent(STATECOUNT+1, ACTIONS)
       
        TheAgent2 = MyAgentMF.Agent(STATECOUNT, ACTIONS)
        TheAgent1.Load(10000000)
        TheAgentsub1.Load(10000000)
        TheAgent2.Load(10000000)
        GameTime = 0
        # Initialise NextAction  Assume Action is scalar:  0:stay, 1:Up, 2:Down
        # Initialise current Game State ~ Believe insigificant: (Player1YPos, Player2YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed)
        GameState1 = CaptureNormalisedState1(200.0, 200.0, 200.0, 0, 0, 1.0)
        GameState2 = CaptureNormalisedState2(200.0, 200.0, 200.0, 0, 0, 1.0)
        
        
    # =================================================================
        #Main Experiment Loop
        while (GameTime < TOTAL_GAMETIME):
                # First just Update the Game Display
                if GameTime % 100 == 0:
                        TheGame.UpdateGameDisplay(GameTime)

                # Determine Next Action From the Agent
                BestAction1 = TheAgent1.Act(GameState1)
                BestAction2 = TheAgent2.Act(GameState2)
                # =================
                # Uncomment this out to Test Game Engine:  Player Paddle then Acts the same way as Right Hand programmed Player
                # Get Current Game State
                #[PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection] = TheGame.ReturnCurrentState()
                
                # Move up if ball is higher than Openient Paddle
                #if (PlayerYPos + 30 > BallYPos + 5):
                #       BestAction = 1
                # Move down if ball lower than Opponent Paddle
                #if (PlayerYPos + 30 < BallYPos + 5):
                #       BestAction = 2
                # =============================
                
                #  Now Apply the Recommended Action into the Game       
                [ReturnScore1,ReturnScore2, Player1YPos, Player2YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed]= TheGame.PlayNextMove(BestAction1, BestAction2)
                NextState1 = CaptureNormalisedState1(Player1YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed)
                NextState2 = CaptureNormalisedState2(Player2YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed)
                
                # Capture the Sample [S, A, R, S"] in Agent Experience Replay Memory 
                TheAgent1.CaptureSample((GameState1,BestAction1,ReturnScore1,NextState1))
                TheAgent2.CaptureSample((GameState2,BestAction2,ReturnScore2,NextState2))
                #  Now Request Agent to DQN Train process  Against Experience
                #TheAgent1.Process()
                #TheAgentsub1.Process()
                #TheAgent2.Process()
                
                # Move State On
                GameState1 = NextState1
                GameState2 = NextState2
                
                # Move GameTime Click
                GameTime += 1

        #print our where wer are after saving where we are
                if GameTime % 500000 == 0:
                        doNothing = 0        		
                        #TheAgent1.Save(GameTime)
                        
                        item_length = len(GameHistory)
                        with open('test.csv', 'wb') as test_file:
                                file_writer = csv.writer(test_file)
                                for i in range(item_length):
                                        file_writer.writerow([GameHistory[i][0],GameHistory[i][1],GameHistory[i][2]])


                if GameTime % 1200 == 0:
                        s = len([x for x in TheGame.recentScores if x > 0])
                        print("Game Time: ", GameTime,"  Game Score1: ", "{0}".format(TheGame.Display_Score1),"  Game Score2: ", "{0}".format(TheGame.Display_Score2)," Recent Rate: ","{0}:{1}".format(s,MyPongMBMF.RECENT_SCORE-s) )
                        GameHistory.append((GameTime,TheGame.Display_Score1,TheGame.Display_Score2))
                        
        # ===============================================
        # End of Game Loop  so Plot the Score vs Game Time profile

        
        #TheAgent.Save()
        
        x_val = [x[0] for x in GameHistory]
        y_val = [x[1] for x in GameHistory]

        plt.plot(x_val,y_val)
        plt.xlabel("Game Time")
        plt.ylabel("Score1")
        plt.show()
        

        
        # =======================================================================
def main():
    #
        # Main Method Just Play our Experiment
        PlayExperiment()
        
        # =======================================================================
if __name__ == "__main__":
    main()
