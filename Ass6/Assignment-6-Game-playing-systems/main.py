from game.board import Board
from game.player import Player, RandomPlayer, MCTSPlayer

def main():
    board = Board(4)

    # player_x = Player('X')
    # player_x = RandomPlayer('X')
    player_x = MCTSPlayer('X', 100, 1.4)

    player_o = MCTSPlayer('O', 1000, 1.4)
    #player_o = RandomPlayer('O')
    
    # game loop
    current_player = player_x
    while not board.is_full():
        current_player.make_move(board)
        current_player = player_o if current_player == player_x else player_x
        board.display()
        winner = board.check_winner()
        if winner:
            print(winner, 'wins!')
            break

if __name__ == '__main__':
    main()
