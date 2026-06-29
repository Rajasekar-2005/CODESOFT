"""
Player classes for Tic-Tac-Toe
"""

class Player:
    def __init__(self, symbol, name):
        """Initialize player with symbol (X or O) and name"""
        self.symbol = symbol
        self.name = name
    
    def get_move(self, board):
        """Get move from player - to be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement get_move()")


class HumanPlayer(Player):
    def __init__(self, symbol, name="Human"):
        """Initialize human player"""
        super().__init__(symbol, name)
    
    def get_move(self, board):
        """Get move from human player"""
        while True:
            try:
                move = input(f"\n{self.name} ({self.symbol}), enter your move (1-9): ")
                move = int(move) - 1  # Convert to 0-indexed
                
                if board.is_valid_move(move):
                    return move
                else:
                    print("Invalid move! That position is already taken or out of range.")
            except ValueError:
                print("Invalid input! Please enter a number between 1 and 9.")
            except KeyboardInterrupt:
                print("\nGame interrupted!")
                return None