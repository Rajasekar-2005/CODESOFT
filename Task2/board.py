"""
Board class for Tic-Tac-Toe game
Handles board state and operations
"""

class Board:
    def __init__(self):
        """Initialize empty 3x3 board"""
        self.board = [' ' for _ in range(9)]
        self.winning_combinations = [
            [0, 1, 2],  # Top row
            [3, 4, 5],  # Middle row
            [6, 7, 8],  # Bottom row
            [0, 3, 6],  # Left column
            [1, 4, 7],  # Middle column
            [2, 5, 8],  # Right column
            [0, 4, 8],  # Diagonal \
            [2, 4, 6],  # Diagonal /
        ]
    
    def display(self):
        """Display the current board state"""
        print("\n")
        print("     |     |     ")
        print(f"  {self.board[0]}  |  {self.board[1]}  |  {self.board[2]}  ")
        print("     |     |     ")
        print("-----------------")
        print("     |     |     ")
        print(f"  {self.board[3]}  |  {self.board[4]}  |  {self.board[5]}  ")
        print("     |     |     ")
        print("-----------------")
        print("     |     |     ")
        print(f"  {self.board[6]}  |  {self.board[7]}  |  {self.board[8]}  ")
        print("     |     |     ")
        print("\n")
    
    def display_positions(self):
        """Display board with position numbers"""
        print("\nPosition numbers:")
        print("     |     |     ")
        print("  1  |  2  |  3  ")
        print("     |     |     ")
        print("-----------------")
        print("     |     |     ")
        print("  4  |  5  |  6  ")
        print("     |     |     ")
        print("-----------------")
        print("     |     |     ")
        print("  7  |  8  |  9  ")
        print("     |     |     ")
        print("\n")
    
    def is_valid_move(self, position):
        """Check if a move is valid"""
        return 0 <= position < 9 and self.board[position] == ' '
    
    def make_move(self, position, player):
        """Make a move on the board"""
        if self.is_valid_move(position):
            self.board[position] = player
            return True
        return False
    
    def undo_move(self, position):
        """Undo a move (used in Minimax)"""
        self.board[position] = ' '
    
    def get_available_moves(self):
        """Get list of available positions"""
        return [i for i in range(9) if self.board[i] == ' ']
    
    def check_winner(self, player):
        """Check if a player has won"""
        for combo in self.winning_combinations:
            if all(self.board[pos] == player for pos in combo):
                return True
        return False
    
    def is_board_full(self):
        """Check if board is full"""
        return ' ' not in self.board
    
    def is_game_over(self):
        """Check if game is over"""
        return self.check_winner('X') or self.check_winner('O') or self.is_board_full()
    
    def get_game_state(self):
        """Get current game state"""
        if self.check_winner('X'):
            return 'X'
        elif self.check_winner('O'):
            return 'O'
        elif self.is_board_full():
            return 'Draw'
        else:
            return 'Ongoing'
    
    def reset(self):
        """Reset the board"""
        self.board = [' ' for _ in range(9)]
    
    def get_board_copy(self):
        """Return a copy of the board"""
        return self.board.copy()