
# # Basic SPR functions
# 
# The following are the basic functions associated to SPR analysis

import io, os, sys, types
from os import listdir
from os.path import isfile, join
import re # This is to easily manipulate the strings in the file name
#from nbformat import read
import pandas as pd
import numpy as np


def loadGamesForPlayer(name):
    
    dataPath = '/Users/juliocorrearios/Dropbox/MCSX/Semester4/Capstone/ScissorsPaperRockAnalysisCodeSolutions/SPR/'
    index = 0
    files = [file for file in listdir(dataPath) if isfile(join(dataPath, file))]
    files.sort()
    games = []
    allGamesData = []
    

    for file in files:
        # Parse the file name for the player names:
        # a. Pull off the timestamp
        players = [str(x) for x in filter(None, re.split('[,\_,\.]',file))]
        player1 = players[1]
        player2 = players[2]
        print('player1: {}, player2: {}\n'.format(player1, player2))

        if player1 == name or name == '*':
            # Player1 is our player, or we getting all games
            playerCol = 0
            opponentCol = 1
            thisPlayer = player1
            opponent = player2
        
        elif player2 == name:
            # Player2 is our player
            playerCol = 0
            opponentCol = 1
            thisPlayer = player2
            opponent = player1
        else:
            continue # Move to next file

    
        # Load this game in:
        
        path = dataPath +file
        df = pd.read_csv(path,sep="\t",header=0)
        
        df.to_csv('data.csv', header=False, index=True)
        
        df = pd.read_csv('data.csv')
        game = df.values
        
        gameData = pd.DataFrame(game, columns=[player1,player2])
    
        allGamesData.append(gameData)
        
        # Grab their moves:
        # 0 = scissors
        # 1 = paper
        # 2 = rock
        playerMoves = np.array(gameData[thisPlayer]).reshape((1,len(gameData[thisPlayer])))
        print(opponent, player2)
        opponentMoves = np.array(gameData[opponent]).reshape((1,len(gameData[opponent])))
        # The player wins if their move is one
        # less than opponents, or (opponent - player) mod 3 == 1.
        # If (opponent - player) mod 3 == 2, then opponent wins.
        # Otherwise if (player == opponent) then it's a tie.
        # Can express this concisely as the following to make
        #  I win == 1
        #  You win == -1
        #  Tie == 0
        results = ((opponentMoves - playerMoves + 1) % 3) - 1
        # Now store all of this in the cell array:
        allGameData = np.column_stack((playerMoves.T, opponentMoves.T, results.T))
        
        #print(allGameData)
        
        # User doesn't want the data returned, just printed:
        print("Game {} for {} ({} iterations):\n".format(index, name, allGameData.shape[0]))
        
        for i in range(allGameData.shape[0]):
            # iteration is the data for this one iteration
            # in the game
            print('{}:\t{},\t{}:\t{},\tresult: {}\n'.format(player1, translateMove(allGameData[i,0]), player2, translateMove(allGameData[i,1]), translateResult(allGameData[i,2])))
          

        if name == '*':
            # If we're grabbing data for all players, then take the 
            #  player2's perspective as well:
            index = index + 1
            allGameData = np.column_stack((opponentMoves.T, playerMoves.T, -results.T))
        
            # User doesn't want the data returned, just printed:
            print('Game {} for {} ({} iterations):\n').format(index, name, allGameData.shape[0])

            for i in range(allGameData.shape[0]):
                # iteration is the data for this one iteration
                #  in the game
                print('{}:\t{},\t{}:\t{},\tresult: {}\n'.format(player2, translateMove(allGameData[i,0]), player1, translateMove(allGameData[i,1]), translateResult(allGameData[i,2])))

        index += 1
 
        
        allGameData = pd.DataFrame(allGameData,columns=[player1,player2,"Result"])
        games.append(allGameData)


    if (index == 1): raiseError('No games found for user {}'.format(name))
    
    return  allGamesData, games


def listPlayers(): 
    dataPath = '/Users/juliocorrearios/Dropbox/MCSX/Semester4/Capstone/ScissorsPaperRockAnalysisCodeSolutions/SPR/'
    plist = []
    index = 1
    files = [file for file in listdir(dataPath) if isfile(join(dataPath, file))]
    files.sort()
    
    for file in files:
        print(file)
        # Parse the file name for the player names:
        # a. Pull off the timestamp
        players = [str(x) for x in filter(None, re.split('[,\_,\.]',file))]
        
        player1 = players[1]
        player2 = players[2]
        
        #print(result)
        print('player1: {}, player2: {}\n'.format(player1, player2))
        plist.append(player1)
        plist.append(player2)
        index = index + 2

    # Finally remove any duplicate names:
    plist = list(set(plist))
    plist.sort()
    playerslist = plist
    
    print('Player names:\n')
    for name in plist:
        print(name)
    
    return playerslist


def translateMove(move):
    if move == 0:
        stringRepresentation = 'scis'
    elif move == 1:
        stringRepresentation = 'papr'
    elif move == 2:
        stringRepresentation = 'rock'
    else:
        print('Error: Move {} not recognised'.format(move))

    return stringRepresentation



def translateResult(gameResult):
    if gameResult == -1:
        stringRepresentation = 'los'
    elif gameResult == 0:
        stringRepresentation = 'tie'
    elif gameResult == 1:
        stringRepresentation = 'win'
    else:
        print('Error: Result {} not recognised'.format(gameResult))

    return stringRepresentation


