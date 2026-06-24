"""
Main game logic for Tic-Tac-Toe
"""

from board import Board
from player import HumanPlayer
from ai import AIPlayer

class TicTacToe:
    def __init__(self):
        """Initialize the game"""
        self.board = Board()
        self.current_player = None
        self.player1 = None
        self.player2 = None
        self.game_mode = None
        self.stats = {
            'X': 0,
            'O': 0,
            'Draw': 0
        }
    
    def setup_game(self):
        """Setup game mode and players"""
        print("\n" + "="*50)
        print("  TIC-TAC-TOE AI GAME")
        print("="*50)
        
        print("\nGame Modes:")
        print("1. Human vs AI")
        print("2. Human vs Human")
        print("3. AI vs AI")
        
        while True:
            try:
                choice = input("\nSelect game mode (1-3): ")
                
                if choice == '1':
                    self.setup_human_vs_ai()
                    break
                elif choice == '2':
                    self.setup_human_vs_human()
                    break
                elif choice == '3':
                    self.setup_ai_vs_ai()
                    break
                else:
                    print("Invalid choice! Please select 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\nGame setup cancelled!")
                return False
        
        return True
    
    def setup_human_vs_ai(self):
        """Setup Human vs AI game"""
        print("\nHuman vs AI Mode")
        
        # Choose symbol
        while True:
            symbol = input("Do you want to be X or O? (X goes first): ").upper()
            if symbol in ['X', 'O']:
                break
            print("Invalid choice! Please enter X or O.")
        
        # Choose difficulty
        print("\nAI Difficulty Levels:")
        print("1. Easy (Random moves)")
        print("2. Medium (Mixed strategy)")
        print("3. Hard (Unbeatable)")
        
        while True:
            diff = input("Select difficulty (1-3): ")
            if diff == '1':
                difficulty = 'easy'
                break
            elif diff == '2':
                difficulty = 'medium'
                break
            elif diff == '3':
                difficulty = 'hard'
                break
            else:
                print("Invalid choice! Please select 1, 2, or 3.")
        
        # Get player name
        name = input("Enter your name: ") or "Human"
        
        # Create players
        if symbol == 'X':
            self.player1 = HumanPlayer('X', name)
            self.player2 = AIPlayer('O', "AI", difficulty)
        else:
            self.player1 = AIPlayer('X', "AI", difficulty)
            self.player2 = HumanPlayer('O', name)
        
        self.current_player = self.player1
        self.game_mode = "Human vs AI"
    
    def setup_human_vs_human(self):
        """Setup Human vs Human game"""
        print("\nHuman vs Human Mode")
        
        player1_name = input("Enter Player 1 name (X): ") or "Player 1"
        player2_name = input("Enter Player 2 name (O): ") or "Player 2"
        
        self.player1 = HumanPlayer('X', player1_name)
        self.player2 = HumanPlayer('O', player2_name)
        self.current_player = self.player1
        self.game_mode = "Human vs Human"
    
    def setup_ai_vs_ai(self):
        """Setup AI vs AI game"""
        print("\nAI vs AI Mode (Watch them battle!)")
        
        self.player1 = AIPlayer('X', "AI-1", 'hard')
        self.player2 = AIPlayer('O', "AI-2", 'hard')
        self.current_player = self.player1
        self.game_mode = "AI vs AI"
    
    def switch_player(self):
        """Switch to the other player"""
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
    
    def play_turn(self):
        """Play one turn"""
        # Get move from current player
        move = self.current_player.get_move(self.board)
        
        if move is None:
            return False  # Game interrupted
        
        # Make the move
        self.board.make_move(move, self.current_player.symbol)
        
        return True
    
    def play_game(self):
        """Main game loop"""
        self.board.reset()
        self.board.display_positions()
        
        print(f"\n{self.player1.name} ({self.player1.symbol}) vs {self.player2.name} ({self.player2.symbol})")
        print(f"Game Mode: {self.game_mode}")
        
        while not self.board.is_game_over():
            self.board.display()
            
            # Play turn
            if not self.play_turn():
                return None  # Game interrupted
            
            # Check for winner
            if self.board.check_winner(self.current_player.symbol):
                self.board.display()
                print(f"\n🎉 {self.current_player.name} ({self.current_player.symbol}) wins! 🎉")
                self.stats[self.current_player.symbol] += 1
                return self.current_player.symbol
            
            # Check for draw
            if self.board.is_board_full():
                self.board.display()
                print("\n🤝 It's a draw! 🤝")
                self.stats['Draw'] += 1
                return 'Draw'
            
            # Switch player
            self.switch_player()
            
            # Small delay for AI vs AI
            if self.game_mode == "AI vs AI":
                import time
                time.sleep(1)
        
        return None
    
    def display_stats(self):
        """Display game statistics"""
        print("\n" + "="*50)
        print("  GAME STATISTICS")
        print("="*50)
        print(f"X Wins: {self.stats['X']}")
        print(f"O Wins: {self.stats['O']}")
        print(f"Draws: {self.stats['Draw']}")
        print(f"Total Games: {sum(self.stats.values())}")
        print("="*50)
    
    def play_again(self):
        """Ask if players want to play again"""
        while True:
            choice = input("\nPlay again? (y/n): ").lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Invalid input! Please enter 'y' or 'n'.")
    
    def start(self):
        """Start the game"""
        if not self.setup_game():
            return
        
        while True:
            result = self.play_game()
            
            if result is None:
                break  # Game was interrupted
            
            self.display_stats()
            
            if not self.play_again():
                break
            
            # Reset board for new game
            self.board.reset()
        
        print("\n" + "="*50)
        print("  Thanks for playing!")
        print("="*50)
        self.display_stats()