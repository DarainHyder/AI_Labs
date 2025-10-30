% ==========================================================
% WAMPA WORLD KNOWLEDGE BASE (SYMBOLIC AI)
% Agent: Deductive Logic Agent (KBA)
% ==========================================================
% safe_move([2,2]).
% --- 1. CONFIGURATION FACTS ---
grid_size(4).

% Predicate to check if a location [X, Y] is within the grid boundaries
in_grid([X, Y]) :-
    grid_size(S),
    X >= 1, X =< S,
    Y >= 1, Y =< S.

% Predicate to check for adjacency (neighbors up, down, left, right)
adjacent([X1, Y], [X2, Y]) :-
    in_grid([X1, Y]),
    (X2 is X1 + 1; X2 is X1 - 1),
    in_grid([X2, Y]).

adjacent([X, Y1], [X, Y2]) :-
    in_grid([X, Y1]),
    (Y2 is Y1 + 1; Y2 is Y1 - 1),
    in_grid([X, Y2]).

% --- 2. DYNAMIC FACTS (Agent's Observations) ---
visited([1, 1]).
visited([1, 2]).

percept(breeze, [1, 2]).
% No stench in [1, 2] is implicitly handled by the absence of percept(stench, [1, 2]).

% --- 3. DEDUCTIVE RULES (The Core Logic) ---
possible_pit(L) :-
    in_grid(L),
    adjacent(L, A),
    visited(A),
    percept(breeze, A).

possible_wampa(L) :-
    in_grid(L),
    adjacent(L, A),
    visited(A),
    percept(stench, A).

known_not_pit(L) :-
    in_grid(L),
    adjacent(L, A),
    visited(A),
    \+ percept(breeze, A).

known_not_wampa(L) :-
    in_grid(L),
    adjacent(L, A),
    visited(A),
    \+ percept(stench, A).

safe_move(L) :-
    in_grid(L),
    \+ visited(L),
    known_not_pit(L),
    known_not_wampa(L).


