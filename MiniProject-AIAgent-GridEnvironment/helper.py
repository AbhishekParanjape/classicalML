from typing import List, Tuple
from grid_universe.gym_env import Observation, Action
from grid_universe.state import State
from grid_universe.levels.grid import Level
from grid_universe.actions import Action
from grid_universe.step import step
from grid_universe.components.properties.pathfinding import PathfindingType
from grid_universe.systems.pathfinding import get_astar_next_position, get_straight_line_next_position
from grid_universe.systems.status import use_status_effect_if_present
from grid_universe.components.properties import Pathfinding
from grid_universe.components.properties.pathfinding import PathfindingType
from grid_universe.components.properties.position import Position
from grid_universe.components.properties.usage_limit import UsageLimit
from grid_universe.components.properties.pmap import PMap
from grid_universe.components.properties.entity_id import EntityID
from grid_universe.components.properties.priority_queue import PriorityQueue
from itertools import count
from dataclasses import replace
from grid_universe.components.properties.is_blocked_at import is_blocked_at
from grid_universe.components.properties.is_in_bounds import is_in_bounds
from grid_universe.gym_env import Observation, Action
from grid_universe.levels.grid import Level
from grid_universe.objectives import all_unlocked_objective_fn, all_pushable_at_exit_objective_fn, exit_objective_fn, default_objective_fn

import random

from grid_universe.gym_env import GridUniverseEnv
from grid_universe.state import State


# Unified imports for Grid Universe tutorial (run this cell first)
from typing import List, Tuple

# Core API
from grid_universe.state import State
from grid_universe.levels.convert import to_state, from_state
from grid_universe.actions import Action
from grid_universe.step import step

# Pathfinding
from grid_universe.systems.pathfinding import get_astar_next_position

# Factories
from grid_universe.levels.factories import (
    create_floor, create_agent, create_box, create_coin, create_exit, create_wall,
    create_key, create_door, create_portal, create_core, create_hazard, create_monster,
    create_phasing_effect, create_speed_effect, create_immunity_effect,
)

# Movement and objectives
from grid_universe.moves import default_move_fn
from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn, all_pushable_at_exit_objective_fn, all_unlocked_objective_fn,
)

# Components and enums
from grid_universe.components.properties import Moving
from grid_universe.components.properties.moving import MovingAxis
from grid_universe.components.properties.appearance import AppearanceName
from grid_universe.components.properties.pathfinding import PathfindingType

# Rendering and display
from grid_universe.renderer.texture import TextureRenderer
from IPython.display import display


def get_agent_position_from_level_repr(level) -> Tuple[int, int]:
    # try to get your (the agent's) position from the Level representation by traversing the grid
    for x in range(level.width):
        for y in range(level.height):
            objects = level.objects_at((y, x))
            for obj in objects:
                if obj.agent is not None:
                    return (y, x)

def get_agent_position_from_state_repr(state) -> Tuple[int, int]:
    # Now, try to get your (the agent's) position from the State representation by reading the position of the agent
    # Hint: `agent_id` has already been defined
    s = state.position[next(iter(state.agent.keys()) )]
    return (s.x, s.y)

# Now, do the same thing for exit
def get_exit_position_from_level_repr(level) -> Tuple[int, int]:
    for x in range(level.width):
        for y in range(level.height):
            objects = level.objects_at((y, x))
            for obj in objects:
                if obj.exit is not None:
                    return (y, x)            
                
def get_exit_position_from_state_repr(state) -> Tuple[int, int]:
    s = state.position[ next(iter(state.exit.keys())) ]
    return (s.x, s.y)

def move(current_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> Action:
    curr_x, curr_y = current_pos
    next_x, next_y = next_pos

    if next_x == curr_x and next_y == curr_y - 1:
        return Action.UP
    elif next_x == curr_x and next_y == curr_y + 1:
        return Action.DOWN
    elif next_x == curr_x - 1 and next_y == curr_y:
        return Action.LEFT
    elif next_x == curr_x + 1 and next_y == curr_y:
        return Action.RIGHT
    else:
        return Action.WAIT

class metastate:
    def __init__(self, state: State):
        
        self.x = state.position[next(iter(state.agent.keys()))].x
        self.y = state.position[next(iter(state.agent.keys()))].y
        self.door = 
        
def get_next_position(
    state: State, entity_id: EntityID, target_id: EntityID
) -> Position:
    """Compute next step toward target using A* (Manhattan metric).

    Ignores collidable/pushable differences and treats only blocking tiles as
    obstacles. Returns current position if already at goal or no path.
    """
    start = state.position[entity_id]
    goal = state.position[target_id]

    if start == goal:
        return start

    def in_bounds(pos: Position) -> bool:
        return is_in_bounds(state, pos)

    def is_blocked(pos: Position) -> bool:
        return is_blocked_at(state, pos, check_collidable=False)

    def heuristic(a: Position, b: Position) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def get_valid_next_positions(position: Position) -> List[Position]:
        neighbor_positions = [
            Position(position.x + dx, position.y + dy) for dx, dy in neighbors
        ]
        return [
            pos for pos in neighbor_positions if in_bounds(pos) and not is_blocked(pos)
        ]

    frontier: PriorityQueue[Tuple[int, int, Position]] = PriorityQueue()
    prev_pos: Dict[Position, Position] = {}
    cost_so_far: Dict[Position, int] = {start: 0}

    tiebreaker = count()  # Unique sequence count
    frontier.put((0, next(tiebreaker), start))

    while not frontier.empty():
        _, __, current = frontier.get()
        if current == goal:
            break
        for next_pos in get_valid_next_positions(current):
            new_cost = cost_so_far[current] + 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                frontier.put((priority, next(tiebreaker), next_pos))
                prev_pos[next_pos] = current

    # Reconstruct path
    if goal not in prev_pos:
        return start  # No path found

    # Walk backwards to get the path
    path: List[Position] = []
    current = goal
    while current != start:
        path.append(current)
        current = prev_pos[current]
    path.reverse()

    if not path:
        return start
    return path[0]


def entity_pathfinding(
    state: State, usage_limit: PMap[EntityID, UsageLimit], entity_id: EntityID
) -> State:
    """Apply pathfinding for a single entity (straight-line or A*)."""
    if entity_id not in state.position or entity_id not in state.pathfinding:
        return state

    pathfinding_type = state.pathfinding[entity_id].type
    pathfinding_target = state.pathfinding[entity_id].target

    if pathfinding_target is None:
        return state

    if pathfinding_target in state.status:
        usage_limit, effect_id = use_status_effect_if_present(
            state.status[pathfinding_target].effect_ids,
            state.phasing,
            state.time_limit,
            usage_limit,
        )
        if effect_id is not None:
            return state

    if pathfinding_type == PathfindingType.STRAIGHT_LINE:
        next_pos = get_straight_line_next_position(state, entity_id, pathfinding_target)
    elif pathfinding_type == PathfindingType.PATH:
        next_pos = get_astar_next_position(state, entity_id, pathfinding_target)
    else:
        raise NotImplementedError

    if is_blocked_at(state, next_pos, check_collidable=False) or not is_in_bounds(
        state, next_pos
    ):
        return state

    return replace(state, position=state.position.set(entity_id, next_pos))