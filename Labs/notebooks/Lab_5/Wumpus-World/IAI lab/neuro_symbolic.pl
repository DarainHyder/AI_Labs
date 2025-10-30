% neuro_symbolic.pl
% Prolog reasoning side for demo integration

:- dynamic python_output/3.  % python_output(Cell, PerceptName, Probability)
                            % e.g. python_output([1,2], breeze_prob, 0.88).

% --- Grid and adjacency (same as your previous code) ---
grid_size(4).

in_grid([X, Y]) :-
    grid_size(S),
    X >= 1, X =< S,
    Y >= 1, Y =< S.

adjacent([X, Y], [X2, Y]) :-
    X2 is X + 1,
    in_grid([X2, Y]).
adjacent([X, Y], [X2, Y]) :-
    X2 is X - 1,
    in_grid([X2, Y]).
adjacent([X, Y], [X, Y2]) :-
    Y2 is Y + 1,
    in_grid([X, Y2]).
adjacent([X, Y], [X, Y2]) :-
    Y2 is Y - 1,
    in_grid([X, Y2]).

% --- Agent observations (visited cells) ---
:- dynamic visited/1.
visited([1,1]).
visited([1,2]).

% --- Helpers: combine probabilities (1 - Π(1 - p_i)) ---
prod([], 1.0).
prod([H|T], P) :-
    prod(T, PT),
    P is H * PT.

combine_probs(List, Combined) :-
    % Combined risk = 1 - product(1 - p_i)
    maplist(negate_one_minus, List, OneMinus),
    prod(OneMinus, Prod),
    Combined is 1.0 - Prod.

negate_one_minus(P, R) :-
    R is (1.0 - P).

% --- Risk computations: gather percept probs from adjacent visited cells ---
risk_of_pit(L, Risk) :-
    findall(PB, (
        adjacent(L, A),
        visited(A),
        python_output(A, breeze_prob, PB)
    ), PBs),
    ( PBs = [] -> Risk = 0.0 ; combine_probs(PBs, Risk) ).

risk_of_wampa(L, Risk) :-
    findall(PS, (
        adjacent(L, A),
        visited(A),
        python_output(A, stench_prob, PS)
    ), PSs),
    ( PSs = [] -> Risk = 0.0 ; combine_probs(PSs, Risk) ).

% --- Decision: safe if combined risk < threshold ---
safe_move(L) :-
    in_grid(L),
    \+ visited(L),
    risk_of_pit(L, RP),
    risk_of_wampa(L, RW),
    TotalRisk is RP + RW,
    TotalRisk < 0.10.  % safe threshold (tunable)

% --- Utility to list candidate safe moves ---
list_safe_moves(List) :-
    grid_size(S),
    findall([X,Y],
        ( between(1,S,X),
          between(1,S,Y),
          safe_move([X,Y])
        ),
        List).

