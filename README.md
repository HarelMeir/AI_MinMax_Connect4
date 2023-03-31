# AI_MinMax_Connect4

This is an implementation of the Connect4 game using the Minimax algorithm with alpha-beta pruning. Connect4 is a two-player strategy game in which the players take turns dropping colored discs into a vertical grid. The objective of the game is to connect four of one's own discs of the same color next to each other vertically, horizontally, or diagonally before your opponent does.

## Requirements
Python 3.x  
numpy
## Installation
### To install the required dependencies, run:


    pip install numpy
    Usage
      

### To start playing the game, simply run the connect4.py file  

    python connect4.py
      
You will be prompted to choose the size of the board and the difficulty level. The difficulty level corresponds to the maximum depth of the search tree for the Minimax algorithm. A higher difficulty level will result in a smarter AI player, but also in longer computation times.

### Once the game is started, you can use the following commands:

r <column>: drop a disc in the specified column (0-indexed)
q: quit the game

