#
# Simple Pong Game based upon PyGame
# My Pong Game, simplify Pong to play with Direct Ball, Pass Paddle and Ball as direct Features into DQN
# 
# Yellow Left Hand Paddle is the DQN Agent Game Play
# A Red Ball return meant the Player missed the last Ball
# A Blue Ball return meant a successful return
#
#  Based upon Siraj Raval's inspiring Machine Learning vidoes  
#  This is based upon Sirajs  Pong Game code 
#  https://github.com/llSourcell/pong_neural_network_live
#
# Note needs imporved frame rate de sensitivition so as to ensure DQN perfomance across all computer types
# Currently Delta Time RATE fixed on each componet update to 7.5 !  => May ned to adjust increase/reduce depending upon perfomance 
# ============================================================================================
import pygame 
import random 
import math
import time

#frame rate per second
SPF = 20    #  Experiment Performance Seems rather sensitive to Computer performance (As Ball as rate vs Paddle rate sensitivity)  

#size of our window
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

#size of our paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
#distance from the edge of the window
PADDLE_BUFFER = 15

#size of our ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

#speeds of our paddle and ball
PADDLE_SPEED = 0.5
INIT_BALL_SPEED = 1
SPEED_GAMMA = 1.1

#number of games from now, to caculate recent score
RECENT_SCORE = 100

#RGB colors for our paddle and ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)

RENDER = 0
#initialize our screen using width and height vars
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# ===============================================================
#Paddle 1 is our learning agent/us
#paddle 2 is the oponent  AI

#draw our ball
def drawBall(ballXPos, ballYPos, BallCol):
    #small rectangle, create it
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    #draw it
    pygame.draw.rect(screen, BallCol, ball)


def drawPaddle1(paddle1YPos):
    #create it
    paddle1 = pygame.Rect(PADDLE_BUFFER, paddle1YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    #draw it
    pygame.draw.rect(screen, YELLOW, paddle1)


def drawPaddle2(paddle2YPos):
    #create it, opposite side
    paddle2 = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    #draw it
    pygame.draw.rect(screen, WHITE, paddle2)



#update the ball, using the paddle posistions the balls positions and the balls directions
def updateBall(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection, ballSpeed, dft,BallColour):
    #update the x and y position
    #dft : delta frame time
    ballXPos = ballXPos + ballXDirection * ballSpeed*dft
    ballYPos = ballYPos + ballYDirection * ballSpeed*dft
    score1 = 0
    score2 = 0
    NewBallColor = BallColour
    #checks for a collision, if the ball hits the Gamer Player side, our Learning agent
    if (ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= paddle1YPos and ballYPos <= paddle1YPos + PADDLE_HEIGHT and ballXDirection <0):
        #switches directions
        angle = +(ballYPos*2.0 - paddle1YPos*2.0 + BALL_HEIGHT - PADDLE_HEIGHT)/(BALL_HEIGHT+PADDLE_HEIGHT)*math.pi*5.0/12        
        ballXDirection = math.cos(angle)
        ballYDirection = math.sin(angle)
        ballSpeed *= SPEED_GAMMA
        ballXPos = 2*(PADDLE_BUFFER + PADDLE_WIDTH)-ballXPos
        #  Player returned the Ball Make the Objective Score (Reward) whenever Returns the Ball  aka playing Serena
        NewBallColor = BLUE
    # Check if Ball past Player
    elif (ballXPos <= 0 and ballXDirection<0):
        #negative score
        # Player Missed the Ball, so negative Score Reward
        score2 = 1
        score1 = -1
        NewBallColor = RED
        
        return [score1, score2, ballXPos, ballYPos, ballXDirection, ballYDirection, ballSpeed, NewBallColor]

    #check if hits the AI Player
    if (ballXPos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER - BALL_WIDTH and ballYPos + BALL_HEIGHT >= paddle2YPos and ballYPos <= paddle2YPos + PADDLE_HEIGHT and ballXDirection >0):
        #switch directions
        angle = math.pi-(ballYPos*2.0 - paddle2YPos*2.0 + BALL_HEIGHT - PADDLE_HEIGHT)/(BALL_HEIGHT+PADDLE_HEIGHT)*math.pi*5.0/12
        ballXDirection = math.cos(angle)
        ballYDirection = math.sin(angle)
        ballSpeed *= SPEED_GAMMA
        ballXPos = 2*(WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER - BALL_WIDTH)-ballXPos
        NewBallColor = WHITE
    #past it
    elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH and ballXDirection>0):
        #positive score
        score1 = 1
        score2 = -1
        NewBallColor = WHITE
        return [score1, score2, ballXPos, ballYPos, ballXDirection, ballYDirection, ballSpeed, NewBallColor]

    #if it hits the top move down
    if (ballYPos <= 0):
        ballYPos = 0
        ballYDirection = -ballYDirection
    #if it hits the bottom, move up
    elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ballYPos = WINDOW_HEIGHT - BALL_HEIGHT
        ballYDirection = -ballYDirection
    return [score1, score2, ballXPos, ballYPos, ballXDirection, ballYDirection, ballSpeed, NewBallColor]
# ========================================================
#update the paddle position
def updatePaddle1(action, paddle1YPos,dft):
    # Assume Action is scalar:  0:stay, 1:Up, 2:Down
    #if move up
    if (action == 1):
        paddle1YPos = paddle1YPos - PADDLE_SPEED*dft
    #if move down
    if (action == 2):
        paddle1YPos = paddle1YPos + PADDLE_SPEED*dft

    #don't let it move off the screen
    if (paddle1YPos < 0):
        paddle1YPos = 0
    if (paddle1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return paddle1YPos


def updatePaddle2(action, paddle2YPos,dft):
    # Assume Action is scalar:  0:stay, 1:Up, 2:Down
    #if move up
    if (action == 1):
        paddle2YPos = paddle2YPos - PADDLE_SPEED*dft
    #if move down
    if (action == 2):
        paddle2YPos = paddle2YPos + PADDLE_SPEED*dft

    #don't let it move off the screen
    if (paddle2YPos < 0):
        paddle2YPos = 0
    if (paddle2YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return paddle2YPos

# =========================================================================
#game class
class PongGame:
    # Initialise Game
    def StartNewGame(self):
     #initialize positions of paddle
     self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
     self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
     self.ballSpeed = INIT_BALL_SPEED
     #random number for initial direction of ball
     # pi/2
     num = random.uniform(0,2*math.pi)
     while (abs(math.cos(num))<0.258819):
      num = random.uniform(0,2*math.pi)

     self.ballXDirection = math.cos(num)
     self.sum+=1
     self.ballYDirection = math.sin(num)
     num = random.randint(0,9)
     #where it will start, y part
     self.ballXPos = WINDOW_WIDTH/2 - BALL_WIDTH/2
     self.ballYPos = (WINDOW_HEIGHT - BALL_HEIGHT)/2

    def __init__(self):
    
        # Initialise pygame
        pygame.init()
        pygame.display.set_caption('Pong DQN Experiment')
        #random number for initial direction of ball
        self.sum = -1
        self.recentScores = []

        self.paddle1YPos = 0
        self.paddle2YPos = 0
        #and ball direction
        self.ballSpeed = 0
        self.ballXDirection = 0
        self.ballYDirection = 0
        #starting point
        self.ballXPos = 0

        self.clock = pygame.time.Clock()
        self.BallColor = WHITE
        self.GTimeDisplay = 0
        self.GScore1 = 0.0
        self.GScore2 = 0.0
        self.Display_Score1 = 0
        self.Display_Score2 = 0


        self.numScorings1 = 0
        self.numScorings2 = 0       
        self.font = pygame.font.SysFont("calibri",20)

        self.StartNewGame()


    def InitialDisplay(self):
        #for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        #make the background black
        screen.fill(BLACK)
        #draw our paddles
        drawPaddle1(self.paddle1YPos)
        drawPaddle2(self.paddle2YPos)
        #draw our ball
        drawBall(self.ballXPos, self.ballYPos,WHITE)
        #
        #updates the window
        pygame.display.flip()
       

    #  Game Update Inlcuding Display
    def PlayNextMove(self, action1, action2):
        # Calculate DeltaFrameTime
        DeltaFrameTime = SPF#self.clock.tick(FPS)
                
        pygame.event.pump()
        score1 = 0
        score2 = 0
        screen.fill(BLACK)
        #update our paddle
        self.paddle1YPos = updatePaddle1(action1, self.paddle1YPos,DeltaFrameTime)
        
        #update evil AI paddle
        self.paddle2YPos = updatePaddle2(action2, self.paddle2YPos,DeltaFrameTime)
        
        #update our vars by updating ball position
        [score1, score2, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection, self.ballSpeed, self.BallColor] = updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection, self.ballSpeed, DeltaFrameTime,self.BallColor)
        #draw the ball
       
        #
        startNewGame = 0
        # Uddate Game Score Moving Average only if Hit or Miss Return
        if(score1 != 0):
            self.GScore1 += score1
            self.numScorings1 += 1            
            startNewGame = 1
            if(score1>0):
                self.recentScores.append(1)
                self.Display_Score1+=1

        if(score2 != 0):
            self.GScore2 += score2
            self.numScorings2 += 1
            startNewGame = 1   
            if(score2>0):
                self.recentScores.append(-1)
                self.Display_Score2+=1

        if(startNewGame==1):
            if(self.sum>=RECENT_SCORE):
                self.recentScores.pop(0)
            self.StartNewGame()
            
        #  Display Parameters
        if(RENDER!=0):
         drawPaddle1(self.paddle1YPos)
         drawPaddle2(self.paddle2YPos)
         drawBall(self.ballXPos, self.ballYPos,self.BallColor)
         pygame.display.flip()
        ScoreDisplay = self.font.render("Score: "+ str("{0}".format(self.Display_Score1))+"  :  "+str("{0}".format(self.Display_Score2)), True,(255,255,255))
        TimeDisplay = self.font.render("Time: "+ str(self.GTimeDisplay), True,(255,255,255))
        GScoreDisplay = self.font.render("GScore: "+ str("{0}".format(self.GScore1))+"  :  "+str("{0}".format(self.GScore2)), True, (255,255,255))
        #BallSpeedDisplay = self.font.render("Speed: "+ str("{0}".format(self.ballSpeed))+"  Direction:  "+str("{0:.3f},{1:.3f}".format(self.ballXDirection,self.ballYDirection)), True, (255,255,255))
        #print("Speed: "+ str("{0}".format(self.ballSpeed))+"  Direction:  "+str("{0:.3f},{1:.3f}".format(self.ballXDirection,self.ballYDirection)))
        screen.blit(ScoreDisplay,(50.,20.))
        screen.blit(TimeDisplay,(50.,40.))
        screen.blit(GScoreDisplay,(50.,60.))
        #screen.blit(BallSpeedDisplay,(50.,100.))
        if(self.sum>RECENT_SCORE):
         s = len([x for x in self.recentScores if x > 0])
         RScoreDisplay = self.font.render("Recent Score: "+ str("{0}".format(s))+"  :  "+str("{0}".format(RECENT_SCORE-s)), True, (255,255,255))
         screen.blit(RScoreDisplay,(50.,80.))
        #update the Game Display
        

        #return the score and the Player Paddle, Ball Position adn Direction 
        return [score1, score2, self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection, self.ballSpeed]

    # Return the Curent Game State
    def ReturnCurrentState(self):
        # Simply return state
        score = 0
        return [self.paddle1YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection]
        
    def UpdateGameDisplay(self,GTime):
        self.GTimeDisplay = GTime
        
