from agents.agent import Agent
from store import register_agent
import copy
import random
import time
import math

#HELPERS
def argmax(anylist):
    maximum=max(anylist)
    return anylist.index(maximum)

def remove(list_, element_):
    if len(list_)<=1:
        return list_
    
    new_list=[[[-1, -1],'a']]*(len(list_)-1)
    i=0
    found=False
    while i<len(list_):
        if not found and element_!=list_[i]:
            new_list[i]=list_[i]
        elif element_==list_[i] and not found:
            found=True
        else:
            new_list[i-1]=list_[i]
        i+=1
    return new_list

#TREE CREATION SECTION
class Node():
    
    def __init__(self, explored, won, my_pos, adv_pos, chess_board, max_step):
        self.explored=explored
        self.won=won
        self.my_pos=my_pos
        self.adv_pos=adv_pos
        self.chess_board=chess_board
        self.max_step=max_step
        self.children=[]
        
    def addNode(self, obj):
        self.children.append(obj)
          
        
    def AddGame(self, n):
        self.explored+=1
        if n>0:
            self.won+=1
            
    def getInfo(self):
        return [self.explored, self.won]
    
    def printInfo(self):
        print('My pos: '+str(self.my_pos))
        print('Adv pos: '+str(self.adv_pos))
        print('Explored: '+str(self.explored))
        print('Won: '+str(self.won))
         

class Tree():
    def __init__(self, root):
        self.root=root





#EVALUATION FUNCTION
        
def fullSight(my_pos, chess_board, adv_pos, radius):
    add_on=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    block=[0, 1, 2, 3]
    to_check=[[my_pos,0]]
    parallel=[my_pos]
    reachable=[[my_pos, 0]]
    depth=0
    while to_check and depth<=radius:
        pos, depth=to_check.pop(0)    
        for i in block:
            new_pos=[pos[0]+add_on[i][0],pos[1]+add_on[i][1]]
            if (new_pos not in parallel) and (not chess_board[pos[0]][pos[1]][i]) and adv_pos!=new_pos and depth<radius:
                to_check+=[[new_pos,depth+1]]
                reachable+=[[new_pos,depth+1]]
                parallel+=[new_pos]
                
    return reachable

def Eval(my_pos, chess_board, adv_pos, max_step, point):
    
    size=len(chess_board[0])
    mySight=fullSight(my_pos, chess_board, adv_pos, 100)
    advSight=fullSight(adv_pos, chess_board, my_pos, 100)
    
    myDistanceToCenter=(abs(point[0]-my_pos[0])+abs(point[1]-my_pos[1]))/(size-1)
    advDistanceToCenter=(abs(point[0]-adv_pos[0])+abs(point[1]-adv_pos[1]))/(size-1)
    distanceAdvantage=advDistanceToCenter-myDistanceToCenter
    
    myFullSpace=len(mySight)/(size**2)
    advFullSpace=len(advSight)/(size**2)
    fullSpaceAdvantage=myFullSpace-advFullSpace
    
    myReachableSpace, advReachableSpace = 0,0
    for j in range(len(mySight)):
            if mySight[j][1]<=max_step:
                myReachableSpace+=1
    for k in range(len(advSight)):
            if advSight[k][1]<=max_step:
                advReachableSpace+=1
    reachableSpaceAdvantage=(myReachableSpace-advReachableSpace)/(size**2)
    
    evaluation=distanceAdvantage*0.25+fullSpaceAdvantage*0.40+reachableSpaceAdvantage*0.35
    
    return evaluation

def cheap_Eval(my_pos, chess_board, adv_pos, max_step, point):
    
    size=len(chess_board[0])
    mySight=fullSight(my_pos, chess_board, adv_pos, max_step)
    advSight=fullSight(adv_pos, chess_board, my_pos, max_step)
    
    myDistanceToCenter=(abs(point[0]-my_pos[0])+abs(point[1]-my_pos[1]))/(size-1)
    advDistanceToCenter=(abs(point[0]-adv_pos[0])+abs(point[1]-adv_pos[1]))/(size-1)
    distanceAdvantage=advDistanceToCenter-myDistanceToCenter
    
    mySpace=len(mySight)/(size**2)
    advSpace=len(advSight)/(size**2)
    spaceAdvantage=mySpace-advSpace
    
    evaluation=distanceAdvantage*0.25+spaceAdvantage*0.75
    
    return evaluation

def Alt_Eval(my_pos, chess_board, adv_pos, max_step, point):
    
    size=len(chess_board[0])
    mySight=fullSight(my_pos, chess_board, adv_pos, 100)
    advSight=fullSight(adv_pos, chess_board, my_pos, 100)
    
    myDistanceToNeck=(abs(point[0]-my_pos[0])+abs(point[1]-my_pos[1]))/(size-1)
    
    myFullSpace=len(mySight)/(size**2)
    advFullSpace=len(advSight)/(size**2)
    fullSpaceAdvantage=myFullSpace-advFullSpace
    
    myReachableSpace, advReachableSpace = 0,0
    for j in range(len(mySight)):
            if mySight[j][1]<=max_step:
                myReachableSpace+=1
    for k in range(len(advSight)):
            if advSight[k][1]<=max_step:
                advReachableSpace+=1
    reachableSpaceAdvantage=(myReachableSpace-advReachableSpace)/(size**2)
    
    evaluation=myDistanceToNeck*0.4+reachableSpaceAdvantage*0.25+fullSpaceAdvantage*0.35
    
    return evaluation





#CHEAP MOVE POLICY
def DFSpath(my_pos, chess_board, adv_pos, radius):
    moveset=[[-1, 0], [0, 1], [1, 0], [0, -1]]
    
    numMoves=list(range(radius+1))
    
    moveWeights=copy.deepcopy(numMoves)
    for j in range(len(moveWeights)):
        moveWeights[j]+=1
    
    numberOfMoves=random.choices(numMoves,weights=moveWeights,k=1)
    
    init_pos=my_pos
    
    i=0
    max_trials=0
    while i<numberOfMoves[0] and max_trials<(2*radius):
        direction=random.randint(0, 3)
        new_pos=[my_pos[0]+moveset[direction][0], my_pos[1]+moveset[direction][1]]
        
        newDis=(abs(init_pos[0]-new_pos[0])+abs(init_pos[1]-new_pos[1]))
        oldDis=(abs(init_pos[0]-my_pos[0])+abs(init_pos[1]-my_pos[1]))
        
        if (newDis>oldDis) and (not chess_board[my_pos[0]][my_pos[1]][direction]) and (not new_pos==adv_pos):
            my_pos=new_pos
            i+=1
        max_trials+=1
    
    availableWalls=[]
    for i in range(4):
        if not chess_board[my_pos[0]][my_pos[1]][i]:
            availableWalls+=[i]
    
    return [my_pos, random.choice(availableWalls)]

def bestMoveDFS(my_pos, chess_board, adv_pos, max_step, n):
    size=len(chess_board)
    to_check=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    rev_map = [2, 3, 0, 1]
    best_move=[0, 0, 'u']
    best_eval=-1
    
    moves=[]
    
    i=0
    while i<n:
        #generate a new DFS move
        new_move=DFSpath(my_pos, chess_board, adv_pos, max_step)
        if [new_move] not in moves:
            #create a new board
            moves+=[new_move]
            pos=new_move[0]
            o=new_move[1]
            new_cb=copy.deepcopy(chess_board)
            alter_pos=[pos[0]+to_check[o][0], pos[1]+to_check[o][1]]
            new_cb[pos[0]][pos[1]][o]=True
            new_cb[alter_pos[0]][alter_pos[1]][rev_map[o]]=True
            
            #generate an evaluation
            new_eval=cheap_Eval(pos, new_cb, adv_pos, max_step, [size/2-0.5, size/2-0.5])
            if new_eval>best_eval:
                best_eval=new_eval
                best_move=new_move
            i+=1
    return best_move




#CHECK FOR AN ENDGAME
#this function checks whether this is an endgame situation
def checkEndgame(chess_board, pos_1, pos_2):
    fullSplit=True
    
    #generates the list of all reachable slots from each of the two players' positions
    #Here fullSight is called with [-1,-1] as last argument, so it acts as if the other
    #player was not there (so their slot is still visited by the DFS)
    list_reach1=fullSight(pos_1, chess_board, [-1,-1], len(chess_board[0])**2)
    territory1=len(list_reach1)+1
    
    i=0
    
    #checks whether the player 2 is on a slot reachable by player 1
    #If not, it means the board is split in 2 with one player on each side
    #fullSplit evaluates to true
    while fullSplit and i<territory1-1:
        if list_reach1[i][0]==pos_2:
            fullSplit=False
        i+=1
    
    if fullSplit:
        list_reach2=fullSight(pos_2, chess_board, [-1,-1], len(chess_board[0])**2)
        territory2=len(list_reach2)+1

    #if the board is split, return the gain for player 1
    if fullSplit:
        if territory1>territory2:
            return [True, 1]
        elif territory1==territory2:
            return [True, 0, i]
        else:
            return [True, -1]
    else:
        return [False, 0]

#Returns whether there are bottlenecks on the board. We define bottlenecks as
#walls that, if they were closed, would cause the end of the game
def isEndClose(chess_board, pos_1, pos_2):
    bottlenecks=[]
    
    #for each possible wall, try whether it creates an end of game situation
    for i in range(len(chess_board[0])):
        for j in range(len(chess_board[0])):
            
            #checks for vertical walls
            if not chess_board[i][j][1]:
                sim1=copy.deepcopy(chess_board)
                sim1[i][j][1]=True
                sim1[i][j+1][3]=True
                check=checkEndgame(sim1, pos_1, pos_2)
                if check[0]:
                    bottlenecks+=[[[i, j],1,check[1]]]
            
            #checks for horizontal walls
            if not chess_board[i][j][2]:
                sim2=copy.deepcopy(chess_board)
                sim2[i][j][2]=True
                sim2[i+1][j][0]=True
                check=checkEndgame(sim2, pos_1, pos_2)
                if check[0]:
                    bottlenecks+=[[[i, j],2,check[1]]]
    
    #return the bottlenecks, if any
    if len(bottlenecks)==0:
        return [False, bottlenecks]
    return [True, bottlenecks]





#DEFINE THE INITIAL MOVESET OR FIND WHETHER A MOVE LEADS TO VICTORY
def oneAway(my_new_pos, chess_board, adv_pos, max_step):
    to_check=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    rev_map = [2, 3, 0, 1]
    
    scan=fullSight(adv_pos, chess_board, my_new_pos, max_step)
    for new_goal in scan:
        new_pos=new_goal[0]
        for o in [0,1,2,3]:
            if not chess_board[new_pos[0]][new_pos[1]][o]:
                new_cb=copy.deepcopy(chess_board)
                alter_pos=[new_pos[0]+to_check[o][0], new_pos[1]+to_check[o][1]]
                new_cb[new_pos[0]][new_pos[1]][o]=True
                new_cb[alter_pos[0]][alter_pos[1]][rev_map[o]]=True
                
                #check the endgame
                inCase=checkEndgame(new_cb, new_pos, my_new_pos)
                if inCase[0] and inCase[1]==1:
                    return True
    return False
            

def bestNFromEval(my_pos, chess_board, adv_pos, max_step, n, evalfctn, point):
    to_check=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    rev_map = [2, 3, 0, 1]
    
    if len(chess_board)<10:
        n=n-2
    else:
        n=n-1
        
    scan=fullSight(my_pos, chess_board, adv_pos, max_step)
    
    walls=['u','r','d','l']
    
    i=0
    best_eval=[-2]*n
    best_move=[[[-1, -1],'a']]*n
    mediocre_moves=[]
    
    #iterate through all possible moves
    while i<len(scan):
        new_goal=scan[i]
        new_pos=new_goal[0]
        
        MoveBestEval=-1
        MoveBestWall=[[-1, -1], 'a']
        #iterate through all possible walls
        for o in [0,1,2,3]:
            
            #create the corresponding chessboard
            if not chess_board[new_pos[0]][new_pos[1]][o]:
                new_cb=copy.deepcopy(chess_board)
                alter_pos=[new_pos[0]+to_check[o][0], new_pos[1]+to_check[o][1]]
                new_cb[new_pos[0]][new_pos[1]][o]=True
                new_cb[alter_pos[0]][alter_pos[1]][rev_map[o]]=True
                
                #generate an evaluation and rank it
                inCase=checkEndgame(new_cb, new_pos, adv_pos)
                if inCase[0]:
                    new_eval=inCase[1]
                elif sum(new_cb[new_pos[0]][new_pos[1]])>=3 and (abs(new_pos[0]-adv_pos[0])+abs(new_pos[1]-adv_pos[1]))<=max_step+1:
                    new_eval=-0.999
                else:
                    new_eval=evalfctn(new_pos, new_cb, adv_pos, max_step, point)
                
                if new_eval>MoveBestEval:
                    MoveBestEval=new_eval
                    MoveBestWall=[new_pos, walls[o]]
        
        j=len(best_eval)-1
        while j>-1:
            if best_eval[j]<MoveBestEval:
                temp=best_eval[j]
                tempgoal=best_move[j]
                best_eval[j]=MoveBestEval
                best_move[j]=MoveBestWall
                if j+1<len(best_eval):
                    best_eval[j+1]=temp
                    best_move[j+1]=tempgoal
                elif tempgoal[1]!='a' and temp>-0.90:
                    mediocre_moves+=[tempgoal]
            elif MoveBestEval<best_eval[j] and j==len(best_eval)-1 and MoveBestEval>-0.90:
                mediocre_moves+=[MoveBestWall]
            j=j-1
                            
        i+=1
    
    l=0
    
    #return the best N-2 moves, or less if there are less available moves, plus 2 mediocre ones (if there are)
    while l<len(best_move):
        if best_move[l][1]=='a':
            break
        l+=1
    if len(mediocre_moves)==0:
        return best_move[:l]
    else:
        if len(chess_board)<10:
            return best_move[:l]+[random.choice(mediocre_moves)]+[random.choice(mediocre_moves)]
        else:
            return best_move[:l]+[random.choice(mediocre_moves)]

def noGift(list_of_moves, chess_board, adv_pos, max_step):
    to_check=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    rev_map = [2, 3, 0, 1]
    walls=['u','r','d','l']
    
    i=len(list_of_moves)-1
    while i>=0:
        move=list_of_moves[i]
        new_cb=copy.deepcopy(chess_board)
        alter_pos=[move[0][0]+to_check[walls.index(move[1])][0], move[0][1]+to_check[walls.index(move[1])][1]]
        new_cb[move[0][0]][move[0][1]][walls.index(move[1])]=True
        new_cb[alter_pos[0]][alter_pos[1]][rev_map[walls.index(move[1])]]=True
        if oneAway(move[0], new_cb, adv_pos, max_step):
            list_of_moves=remove(list_of_moves, move)
        i=i-1
    return list_of_moves



#RUN A GAME
def runGame(pos_1, pos_2, chess_board, max_step, depth, thoughtfulness):
    size=len(chess_board)
    isOverTime=0
    choiceTime=0
    
    sim_chess_board=copy.deepcopy(chess_board)
    toupdate=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    reverse=[2,3,0,1]
    rounds=0
    
    isOverStart=time.time()
    isOver=checkEndgame(chess_board, pos_1, pos_2)
    isOverEnd=time.time()
    isOverTime+=isOverEnd-isOverStart
    
    
    while not isOver[0] and rounds<depth:
        choiceTimeStart=time.time()
        move1=bestMoveDFS(pos_1, chess_board, pos_2, max_step, thoughtfulness)
        choiceTimeEnd=time.time()
        choiceTime+=choiceTimeEnd-choiceTimeStart
        
        pos_1=move1[0]
        #print('player1 moves to'+str(move1))
        sim_chess_board[move1[0][0]][move1[0][1]][move1[1]]=True
        sim_chess_board[move1[0][0]+toupdate[move1[1]][0]][move1[0][1]+toupdate[move1[1]][1]][reverse.index(move1[1])]=True
        
        isOverStart=time.time()
        isOver=[(all(sim_chess_board[pos_2[0]][pos_2[1]]) or all(sim_chess_board[pos_1[0]][pos_1[1]])),-1]
        isOverEnd=time.time()
        isOverTime+=isOverEnd-isOverStart
        
        if isOver:
            break
        
        choiceTimeStart=time.time()
        move2=bestMoveDFS(pos_2, chess_board, pos_1, max_step, thoughtfulness)
        choiceTimeEnd=time.time()
        choiceTime+=choiceTimeEnd-choiceTimeStart
        
        pos_2=move2[0]
        #print('player2 moves to'+str(move2))
        sim_chess_board[move2[0][0]][move2[0][1]][move2[1]]=True
        sim_chess_board[move2[0][0]+toupdate[move2[1]][0]][move2[0][1]+toupdate[move2[1]][1]][reverse.index(move2[1])]=True
        
        isOverStart=time.time()
        if rounds%3==0:
            isOver=checkEndgame(sim_chess_board, pos_1, pos_2)
        else:
            isOver=[(all(sim_chess_board[pos_2[0]][pos_2[1]]) or all(sim_chess_board[pos_1[0]][pos_1[1]])),-1]
        isOverEnd=time.time()
        isOverTime+=isOverEnd-isOverStart
        rounds+=1
    
    isOver=checkEndgame(sim_chess_board, pos_1, pos_2)
    if isOver[0]:
        return isOver
    else:
        return [0, Eval(pos_1, sim_chess_board, pos_2, max_step, [size/2-0.5, size/2-0.5])]
    
def BestFromTree(chess_board, my_pos, adv_pos, max_step, list_moves, depth, thoughtfulness, time_left):
    start=time.time()
    
    c=1.41

    walls=['u','r','d','l']
    toupdate=[[-1, 0],[0, 1],[1, 0],[0, -1]]
    reverse=['d','l','u','r']
        
    num_branches=len(list_moves)
    root=Node(0, 0, my_pos, adv_pos, chess_board, max_step)
    MCT=Tree(root)
    score=[0]*num_branches
    exploration_data=[1]*num_branches
    
    runs=0
    first_round=0
    
    for move in list_moves:
        sim_chess_board=copy.deepcopy(chess_board)
        sim_chess_board[move[0][0]][move[0][1]][walls.index(move[1])]=True
        sim_chess_board[move[0][0]+toupdate[walls.index(move[1])][0]][move[0][1]+toupdate[walls.index(move[1])][1]][reverse.index(move[1])]=True
        new_node=Node(0, 0, move[0], adv_pos, sim_chess_board, max_step)
        new_node.AddGame(runGame(move[0], adv_pos, sim_chess_board, max_step, depth, thoughtfulness)[1])
        root.addNode(new_node)
        runs+=1
    
    while time.time()<(start+time_left):
        #print('scores are '+str(score)+' and we choose move '+str(list_moves[np.argmax(score)]))
        move=list_moves[argmax(score)]
        exploration_data[argmax(score)]+=1
        sim_chess_board=copy.deepcopy(chess_board)
        sim_chess_board[move[0][0]][move[0][1]][walls.index(move[1])]=True
        sim_chess_board[move[0][0]+toupdate[walls.index(move[1])][0]][move[0][1]+toupdate[walls.index(move[1])][1]][reverse.index(move[1])]=True   
        root.children[argmax(score)].AddGame(runGame(move[0], adv_pos, sim_chess_board, max_step, depth, thoughtfulness)[1])
        explored, won = root.children[argmax(score)].getInfo()
        
        for k in range(len(score)):
            explored, won = root.children[k].getInfo()
            score[k]=(won/explored)+c*((math.log(runs)/explored)**(1/2))

        runs+=1
    
    for l in range(len(score)):
        explored, won=root.children[l].getInfo()
        score[l]=won/explored
    
    new_time=time.time()
    return list_moves[argmax(score)]




@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """


    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent T-3"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        start=time.time()
        depth=5
        size=len(chess_board)
        n=14-size
        thoughtfulness=3
        
        #converts the positions into lists, for convenience
        my_pos=[my_pos[0],my_pos[1]]
        adv_pos=[adv_pos[0],adv_pos[1]]
        
        risk=isEndClose(chess_board, my_pos, adv_pos)
        tooFar=False
                
        if risk[0]:
            tooFar=True
            for bottleneck in risk[1]:
                if abs(bottleneck[0][0]-my_pos[0])+abs(bottleneck[0][1]-my_pos[1])<max_step:
                    tooFar=False
                    
        if tooFar:
            list_of_moves=bestNFromEval(my_pos, chess_board, adv_pos, max_step, n, Alt_Eval, bottleneck[0])
        else:
            list_of_moves=bestNFromEval(my_pos, chess_board, adv_pos, max_step, n, Eval, [size/2-0.5, size/2-0.5])
                
        list_of_moves=noGift(list_of_moves, chess_board, adv_pos, max_step)
                
        time_left=1.75-(time.time()-start)
        
        best_move=BestFromTree(chess_board, my_pos, adv_pos, max_step, list_of_moves, depth, thoughtfulness, time_left)
        
        return best_move[0], self.dir_map[best_move[1]]