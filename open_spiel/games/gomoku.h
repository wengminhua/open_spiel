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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_GOMOKU_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_GOMOKU_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Standard game of GOMOKU:
// https://baike.baidu.com/item/%E4%BA%94%E5%AD%90%E6%A3%8B/130266
//
// Parameters: none

namespace open_spiel {
namespace gomoku {

// Constants.
constexpr int kNumPlayers = 2;
constexpr int kNumRows = 15;
constexpr int kNumCols = 15;
constexpr int kNumPoints = kNumRows * kNumCols;
constexpr int kPointStates = 1 + kNumPlayers;  // empty, 'black', and 'white'.

// https://math.stackexchange.com/questions/485752/tictactoe-state-space-choose-calculation/485852
// Total number of states is only for testing.
// constexpr int kNumberStates = 5478;

// State of a point.
enum class PointState {
  kEmpty,
  kBlack,
  kWhite,
};

// State of an in-play game.
class GomokuState : public State {
 public:
  GomokuState(int num_distinct_actions);

  GomokuState(const GomokuState&) = default;
  GomokuState& operator=(const GomokuState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(Player player) const override;
  std::string Observation(Player player) const override;
  void ObservationAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  PointState BoardAt(int point) const { return board_[point]; }
  PointState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }

 protected:
  std::array<PointState, kNumPoints> board_;
  void DoApplyAction(Action move) override;

 private:
  bool HasFiveInner(int r_start, int c_start, PointState s) const; // Is there a connected five points in a 5x5 rect?
  bool HasFive(Player player) const;  // Does this player have a connected five points?
  bool IsFull() const;                // Is the board full?
  Player current_player_ = 0;         // Player zero goes first
};

// Game object.
class GomokuGame : public Game {
 public:
  explicit GomokuGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumPoints; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new GomokuState(NumDistinctActions()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new GomokuGame(*this));
  }
  std::vector<int> ObservationNormalizedVectorShape() const override {
    return {kPointStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const { return kNumPoints; }
};

PointState PlayerToState(Player player);
std::string StateToString(PointState state);

inline std::ostream& operator<<(std::ostream& stream, const PointState& state) {
  return stream << StateToString(state);
}

}  // namespace gomoku
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_GOMOKU_H_
