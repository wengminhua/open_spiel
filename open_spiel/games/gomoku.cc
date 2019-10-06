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

#include "open_spiel/games/gomoku.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace gomoku {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"gomoku",
    /*long_name=*/"Gomoku",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new GomokuGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

PointState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return PointState::kBlack;
    case 1:
      return PointState::kWhite;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return PointState::kEmpty;
  }
}

std::string StateToString(PointState state) {
  switch (state) {
    case PointState::kEmpty:
      return ".";
    case PointState::kWhite:
      return "o";
    case PointState::kBlack:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

void GomokuState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], PointState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  current_player_ = 1 - current_player_;
}

std::vector<Action> GomokuState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty point.
  std::vector<Action> moves;
  for (int point = 0; point < kNumPoints; ++point) {
    if (board_[point] == PointState::kEmpty) {
      moves.push_back(point);
    }
  }
  return moves;
}

std::string GomokuState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id % kNumCols, ",", action_id / kNumCols, ")");
}

bool GomokuState::HasFiveInner(int r_start, int c_start, PointState s) const {
  for (int i = 0; i < 5; ++i) {
    bool r_sum = BoardAt(r_start + i, c_start) == s && 
                 BoardAt(r_start + i, c_start + 1) == s &&
                 BoardAt(r_start + i, c_start + 2) == s && 
                 BoardAt(r_start + i, c_start + 3) == s && 
                 BoardAt(r_start + i, c_start + 4) == s;
    if (r_sum) {
      return true;
    }
    bool c_sum = BoardAt(r_start,     c_start + i) == s && 
                BoardAt(r_start + 1, c_start + i) == s &&
                BoardAt(r_start + 2, c_start + i) == s && 
                BoardAt(r_start + 3, c_start + i) == s &&
                BoardAt(r_start + 4, c_start + i) == s;
    if (c_sum) {
      return true;
    }    
  }
  bool x_sum = BoardAt(r_start    , c_start) == s && 
              BoardAt(r_start + 1, c_start + 1) == s &&  
              BoardAt(r_start + 2, c_start + 2) == s && 
              BoardAt(r_start + 3, c_start + 3) == s && 
              BoardAt(r_start + 4, c_start + 4) == s;
  if (x_sum) {
    return true;
  }
  x_sum = BoardAt(r_start + 4, c_start) == s && 
          BoardAt(r_start + 3, c_start + 1) == s && 
          BoardAt(r_start + 2, c_start + 2) == s && 
          BoardAt(r_start + 1, c_start + 3) == s && 
          BoardAt(r_start    , c_start + 4) == s;
  if (x_sum) {
    return true;
  }
  return false;
}

bool GomokuState::HasFive(Player player) const {
  PointState s = PlayerToState(player);
  for (int r = 0; r <= kNumRows - 5; ++r) {
    for (int c = 0; c <= kNumCols - 5; ++c) {
      bool has_five = HasFiveInner(r, c, s);
      if (has_five) {
        return true;
      }
    }
  }
  return false;
}

bool GomokuState::IsFull() const {
  for (int point = 0; point < kNumPoints; ++point) {
    if (board_[point] == PointState::kEmpty) return false;
  }
  return true;
}

GomokuState::GomokuState(int num_distinct_actions)
    : State(num_distinct_actions, kNumPlayers) {
  std::fill(begin(board_), end(board_), PointState::kEmpty);
}

std::string GomokuState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool GomokuState::IsTerminal() const {
  return HasFive(0) || HasFive(1) || IsFull();
}

std::vector<double> GomokuState::Returns() const {
  if (HasFive(0)) {
    return {1.0, -1.0};
  } else if (HasFive(1)) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string GomokuState::InformationState(Player player) const {
  return HistoryString();
}

std::string GomokuState::Observation(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void GomokuState::ObservationAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(kNumPoints * kPointStates);
  std::fill(values->begin(), values->end(), 0.);
  for (int point = 0; point < kNumPoints; ++point) {
    (*values)[kNumPoints * static_cast<int>(board_[point]) + point] = 1.0;
  }
}

void GomokuState::UndoAction(Player player, Action move) {
  board_[move] = PointState::kEmpty;
  current_player_ = player;
  history_.pop_back();
}

std::unique_ptr<State> GomokuState::Clone() const {
  return std::unique_ptr<State>(new GomokuState(*this));
}

GomokuGame::GomokuGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace gomoku
}  // namespace open_spiel
