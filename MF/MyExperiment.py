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
import MyPong # My PyGame Pong Game 
import MyAgent # My DQN Based Agent
import MyAgent2 # My DQN Based Agent
import numpy as np 
import random 
import matplotlib.pyplot as plt

#
# =======================================================================
#   DQN Algorith Paramaters 
ACTIONS = 3 # Number of Actions.  Acton istelf is a scalar:  0:stay, 1:Up, 2:Down
STATECOUNT = 7 # Size of State [ PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection, OpponentYPos] 
TOTAL_GAMETIME = 3000000
# =======================================================================
# Normalise GameState
def CaptureNormalisedState(Player1YPos, Player2YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed):
        gstate = np.zeros([STATECOUNT])
        gstate[0] = Player1YPos/400.0    # Normalised PlayerYPos
        gstate[1] = Player2YPos/400.0  # Normalised PlayerYPos
        gstate[2] = BallXPos/400.0      # Normalised BallXPos
        gstate[3] = BallYPos/400.0      # Normalised BallYPos
        gstate[4] = BallXDirection/1.0  # Normalised BallXDirection
        gstate[5] = BallYDirection/1.0  # Normalised BallYDirection
        gstate[6] = BallSpeed  # can't normalise
        return gstate
# =====================================================================
# Main Experiment Method 
def PlayExperiment():
        GameTime = 0
    
        GameHistory = []
        
        #Create our PongGame instance
        TheGame = MyPong.PongGame()
    # Initialise Game
        TheGame.InitialDisplay()
        #
        #  Create our Agent (including DQN based Brain)
        TheAgent1 = MyAgent.Agent(STATECOUNT, ACTIONS)
        #TheAgent1.Load()
        TheAgent2 = MyAgent2.Agent(STATECOUNT, ACTIONS)
        GameTime = 0
        # Initialise NextAction  Assume Action is scalar:  0:stay, 1:Up, 2:Down
        # Initialise current Game State ~ Believe insigificant: (Player1YPos, Player2YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed)
        GameState = CaptureNormalisedState(200.0, 200.0, 200.0, 200.0, 0, 0, 1.0)
        
        
    # =================================================================
        #Main Experiment Loop
        while (GameTime<TOTAL_GAMETIME):
                # First just Update the Game Display
                if GameTime % 100 == 0:
                        TheGame.UpdateGameDisplay(GameTime)

                # Determine Next Action From the Agent
                BestAction1 = TheAgent1.Act(GameState)
                BestAction2 = TheAgent2.Act(GameState)
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
                NextState = CaptureNormalisedState(Player1YPos, Player2YPos, BallXPos, BallYPos, BallXDirection, BallYDirection, BallSpeed)
                
                # Capture the Sample [S, A, R, S"] in Agent Experience Replay Memory 
                TheAgent1.CaptureSample((GameState,BestAction1,ReturnScore1,NextState))
                TheAgent2.CaptureSample((GameState,BestAction2,ReturnScore2,NextState))
                #  Now Request Agent to DQN Train process  Against Experience
                TheAgent1.Process()
                TheAgent2.Process()
                
                # Move State On
                GameState = NextState
                
                # Move GameTime Click
                GameTime = GameTime+1

        #print our where wer are after saving where we are
                if GameTime % 5000 == 0:
                        TheAgent1.Save(GameTime)
            

                if GameTime % 200 == 0:
                        print("Game Time: ", GameTime,"  Game Score1: ", "{0:.2f}".format(TheGame.GScore1),"  Game Score2: ", "{0:.2f}".format(TheGame.GScore2) )
                        GameHistory.append((GameTime,TheGame.GScore1,TheGame.GScore2))
                        
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
