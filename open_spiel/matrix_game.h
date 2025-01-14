// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_OPEN_SPIEL_MATRIX_GAME_H_
#define THIRD_PARTY_OPEN_SPIEL_MATRIX_GAME_H_

#include <algorithm>
#include <numeric>

#include "open_spiel/normal_form_game.h"
#include "open_spiel/spiel.h"

// A matrix game is an example of a 2-player normal-form game.

namespace open_spiel {
namespace matrix_game {

constexpr int kRowPlayer = 0;
constexpr int kColPlayer = 1;

// Return a flattened version of these vector of rows. This simply scans each
// row in turn, appending each elements onto the end of a 1D vector. The rows
// must have the same size.
std::vector<double> FlattenMatrix(
    const std::vector<std::vector<double>>& matrix_rows);

class MatrixGame : public NormalFormGame {
 public:
  MatrixGame(GameType game_type, GameParameters game_parameters,
             std::vector<std::string> row_action_names,
             std::vector<std::string> col_action_names,
             std::vector<double> row_utilities,
             std::vector<double> col_utilities)
      : NormalFormGame(game_type, game_parameters),
        row_action_names_(row_action_names),
        col_action_names_(col_action_names),
        row_utilities_(row_utilities),
        col_utilities_(col_utilities) {}

  MatrixGame(GameType game_type, GameParameters game_parameters,
             std::vector<std::string> row_action_names,
             std::vector<std::string> col_action_names,
             const std::vector<std::vector<double>> row_utilities,
             const std::vector<std::vector<double>> col_utilities)
      : NormalFormGame(game_type, game_parameters),
        row_action_names_(row_action_names),
        col_action_names_(col_action_names),
        row_utilities_(FlattenMatrix(row_utilities)),
        col_utilities_(FlattenMatrix(col_utilities)) {}

  // Implemwentation of Game interface
  int NumDistinctActions() const override {
    return std::max(NumRows(), NumCols());
  }

  std::unique_ptr<State> NewInitialState() const override;

  int NumPlayers() const override { return 2; }

  double MinUtility() const override {
    return std::min(
        *std::min_element(begin(row_utilities_), end(row_utilities_)),
        *std::min_element(begin(col_utilities_), end(col_utilities_)));
  }

  double MaxUtility() const override {
    return std::max(
        *std::max_element(begin(row_utilities_), end(row_utilities_)),
        *std::max_element(begin(col_utilities_), end(col_utilities_)));
  }

  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new MatrixGame(*this));
  }

  // Methods for MatrixState to call.
  int NumRows() const { return row_action_names_.size(); }
  int NumCols() const { return col_action_names_.size(); }
  double RowUtility(int row, int col) const {
    return row_utilities_[Index(row, col)];
  }
  double ColUtility(int row, int col) const {
    return col_utilities_[Index(row, col)];
  }
  double PlayerUtility(Player player, int row, int col) {
    SPIEL_CHECK_TRUE(player == Player{0} || player == Player{1});
    return (player == Player{0} ? row_utilities_[Index(row, col)]
                                : col_utilities_[Index(row, col)]);
  }
  const std::string& RowActionName(int row) const {
    return row_action_names_[row];
  }
  const std::string& ColActionName(int col) const {
    return col_action_names_[col];
  }

 private:
  int Index(int row, int col) const { return row * NumCols() + col; }
  std::vector<std::string> row_action_names_;
  std::vector<std::string> col_action_names_;
  std::vector<double> row_utilities_;
  std::vector<double> col_utilities_;
};

class MatrixState : public NFGState {
 public:
  explicit MatrixState(const MatrixGame& game);
  MatrixState(const MatrixState&) = default;

  virtual std::vector<Action> LegalActions(Player player) const {
    if (player == kSimultaneousPlayerId) {
      return LegalFlatJointActions();
    } else {
      std::vector<Action> moves(player == kRowPlayer ? game_.NumRows()
                                                     : game_.NumCols());
      std::iota(moves.begin(), moves.end(), 0);  // fill with values 0...n-1
      return moves;
    }
  }

  std::string ToString() const override;

  virtual std::string ActionToString(Player player, Action action_id) const {
    if (player == kSimultaneousPlayerId)
      return FlatJointActionToString(action_id);
    else if (player == kRowPlayer)
      return game_.RowActionName(action_id);
    else
      return game_.ColActionName(action_id);
  }

  virtual bool IsTerminal() const { return !joint_move_.empty(); }

  virtual std::vector<double> Returns() const {
    if (IsTerminal()) {
      return {game_.RowUtility(joint_move_[0], joint_move_[1]),
              game_.ColUtility(joint_move_[0], joint_move_[1])};
    } else {
      return {0, 0};
    }
  }

  virtual std::unique_ptr<State> Clone() const {
    return std::unique_ptr<State>(new MatrixState(*this));
  }

 protected:
  virtual void DoApplyActions(const std::vector<Action>& moves) {
    SPIEL_CHECK_EQ(moves.size(), 2);
    SPIEL_CHECK_GE(moves[kRowPlayer], 0);
    SPIEL_CHECK_LT(moves[kRowPlayer], game_.NumRows());
    SPIEL_CHECK_GE(moves[kColPlayer], 0);
    SPIEL_CHECK_LT(moves[kColPlayer], game_.NumCols());
    joint_move_ = moves;
  }

 private:
  const MatrixGame& game_;
  std::vector<Action> joint_move_{};  // joint move that was chosen
};

// Create a matrix game with the specified utilities and row/column names.
// Utilities must be in row-major form.
std::unique_ptr<MatrixGame> CreateMatrixGame(
    const std::string& short_name, const std::string& long_name,
    const std::vector<std::string>& row_names,
    const std::vector<std::string>& col_names,
    const std::vector<std::vector<double>>& row_player_utils,
    const std::vector<std::vector<double>>& col_player_utils);

// Create a matrix game with the specified utilities, with default names
// ("short_name", "Long Name", row0, row1.., col0, col1, ...).
// Utilities must be in row-major form.
std::unique_ptr<MatrixGame> CreateMatrixGame(
    const std::vector<std::vector<double>>& row_player_utils,
    const std::vector<std::vector<double>>& col_player_utils);

}  // namespace matrix_game
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_MATRIX_GAME_H_
