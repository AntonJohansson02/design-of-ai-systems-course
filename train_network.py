import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import os
from game.board import Board
from game.nn import ValueNetwork, ConvValueNetwork
from game.mcts import MCTS
from game.mctsnn import MCTSNN  # Import MCTSNN class
from copy import deepcopy

def generate_game_data(num_games, board_size=3, mcts_simulations=1000):
    """Generate game data by self-play using MCTS"""
    game_data = []
    
    for game in range(num_games):
        print(f"Playing game {game+1}/{num_games}")
        board = Board(board_size)
        mcts = MCTS(mcts_simulations, 1.4)
        player = 'X'
        game_states = []
        
        # Play a game
        while not board.check_winner() and not board.is_full():
            # Record current board state
            game_states.append((deepcopy(board), player))
            
            # Get best move from MCTS
            move = mcts.search(board, player)
            if move:
                board.make_move(move[0], move[1], player)
            
            # Switch player
            player = 'O' if player == 'X' else 'X'
        
        # Get winner
        winner = board.check_winner()
        
        # Assign rewards based on outcome
        for board_state, curr_player in game_states:
            # Create input tensor
            input_tensor = []
            opponent = 'O' if curr_player == 'X' else 'X'
            
            for row in board_state.grid:
                for cell in row:
                    if cell == curr_player:
                        input_tensor.append(1)
                    elif cell == opponent:
                        input_tensor.append(-1)
                    else:
                        input_tensor.append(0)
            
            # Assign reward
            if winner == curr_player:
                reward = 1.0  # Win
            elif winner is None:
                reward = 0.0  # Draw
            else:
                reward = -1.0  # Loss
                
            game_data.append((input_tensor, reward))
    
    return game_data

def generate_game_data_nn(num_games, model_path, board_size=3, mcts_simulations=1000):
    """Generate game data by self-play using MCTSNN with neural network guidance"""
    game_data = []
    
    # Load the value network model
    value_network = ValueNetwork(board_size=board_size)
    value_network.load_state_dict(torch.load(model_path))
    value_network.eval()
    
    for game in range(num_games):
        print(f"Playing game {game+1}/{num_games} with MCTSNN")
        board = Board(board_size)
        mctsnn = MCTSNN(mcts_simulations, 1.4, value_network, board_size)
        player = 'X'
        game_states = []
        
        # Play a game
        while not board.check_winner() and not board.is_full():
            # Record current board state
            game_states.append((deepcopy(board), player))
            
            # Get best move from MCTSNN
            move = mctsnn.search(board, player)
            if move:
                board.make_move(move[0], move[1], player)
            
            # Switch player
            player = 'O' if player == 'X' else 'X'
        
        # Get winner
        winner = board.check_winner()
        
        # Assign rewards based on outcome
        for board_state, curr_player in game_states:
            # Create input tensor
            input_tensor = []
            opponent = 'O' if curr_player == 'X' else 'X'
            
            for row in board_state.grid:
                for cell in row:
                    if cell == curr_player:
                        input_tensor.append(1)
                    elif cell == opponent:
                        input_tensor.append(-1)
                    else:
                        input_tensor.append(0)
            
            # Assign reward
            if winner == curr_player:
                reward = 1.0  # Win
            elif winner is None:
                reward = 0.0  # Draw
            else:
                reward = -1.0  # Loss
                
            game_data.append((input_tensor, reward))
    
    return game_data

def save_game_data(game_data, board_size):
    """Save game data to a file with board size in the filename"""
    filename = f"game_data_board_{board_size}x{board_size}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(game_data, f)
    print(f"Game data saved to {filename}")
    return filename

def save_game_data_nn(game_data, board_size):
    """Save NN-guided game data to a file with board size in the filename"""
    filename = f"game_data_nn_board_{board_size}x{board_size}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(game_data, f)
    print(f"NN-guided game data saved to {filename}")
    return filename

def load_game_data(board_size):
    """Load game data from a file with board size in the filename"""
    filename = f"game_data_board_{board_size}x{board_size}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            game_data = pickle.load(f)
        print(f"Loaded {len(game_data)} training examples from {filename}")
        return game_data
    else:
        print(f"No saved game data found for {board_size}x{board_size} board")
        return None

def load_game_data_nn(board_size):
    """Load NN-guided game data from a file with board size in the filename"""
    filename = f"game_data_nn_board_{board_size}x{board_size}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            game_data = pickle.load(f)
        print(f"Loaded {len(game_data)} NN-guided training examples from {filename}")
        return game_data
    else:
        print(f"No saved NN-guided game data found for {board_size}x{board_size} board")
        return None

def train_network(game_data, model, epochs=100, batch_size=32, learning_rate=0.001, board_size=3):
    """Train the value network on the game data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert game data to tensors
    states = []
    values = []
    
    for state, value in game_data:
        states.append(torch.FloatTensor(state))
        values.append(torch.FloatTensor([value]))
    
    # Create dataset
    dataset = list(zip(states, values))
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0
        batches = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if not batch:
                continue
                
            batch_states, batch_values = zip(*batch)
            
            batch_states = torch.stack(batch_states).to(device)
            batch_values = torch.stack(batch_values).to(device)
            
            # Forward pass
            predictions = model(batch_states)
            loss = criterion(predictions, batch_values)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_loss = total_loss/batches if batches > 0 else 0
        
        # Save checkpoint if loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_filename = f"value_network_{board_size}x{board_size}.pt"
            torch.save(model.state_dict(), model_filename)
            print(f"Checkpoint saved at epoch {epoch+1} with loss {best_loss:.6f}")
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save the final model
    model_filename = f"value_network_{board_size}x{board_size}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Training complete. Model saved as '{model_filename}'")
    return model_filename

if __name__ == "__main__":
    # Parameters
    board_size = 4  
    num_games = 100 # Number of self-play games to generate training data
    mcts_simulations = 10000 # Number of MCTS simulations per move
    epochs = 200
    
    # Check if we already have data for this board size
    game_data = load_game_data(board_size)
    
    if game_data is None:
        # Generate training data through self-play
        game_data = generate_game_data(num_games, board_size, mcts_simulations)
        print(f"Generated {len(game_data)} training examples")
        
        # Save the generated data
        save_game_data(game_data, board_size)
    
    # Create and train the network
    value_network = ValueNetwork(board_size=board_size)
    model_file = train_network(game_data, value_network, epochs=epochs, board_size=board_size)
    
    print(f"\nTo use this model in your game, update your main.py:")
    print(f"player_x = MCTSNNPlayer('X', 100, 1.4, model_path='{model_file}', board_size={board_size})")