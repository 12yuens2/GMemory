_pickup_example = """Your task: pick up a green key
You see a green key 1 step to the right, a wall 1 step forward.
> think: The key I need is beside me, one step to my right, and there is a wall directly ahead. I cannot move sideways, so I have to turn right before I can reach it.
OK.
> turn right
You see a green key 1 step forward, a red box 1 step forward and 1 step to the right, a green key 3 steps to the right, a blue box 4 steps forward and 1 step to the right, a wall 6 steps forward.
> think: The green key is now directly in front of me. It is one step forward, which means I am standing right next to it and can pick it up from here.
OK.
> go forward
Nothing happens. You see a green key 1 step forward, a red box 1 step forward and 1 step to the right, a green key 3 steps to the right, a blue box 4 steps forward and 1 step to the right, a wall 6 steps forward.
> think: Nothing happened, because the key itself is blocking that square - objects cannot be walked onto. Something 1 step forward is already within reach, so I should pick it up rather than approach it.
OK.
> pick up
You see a red box 1 step forward and 1 step to the right, a green key 3 steps to the right, a blue box 4 steps forward and 1 step to the right, a wall 6 steps forward. You are carrying a green key."""

babyai_few_shots: list[str] = [_pickup_example]

babyai_solver_system_prompt: str = """
You are an agent in a gridworld. Each turn you are told what you can see, and you reply with one line: either an action, or a thought.

The only actions are these seven, written exactly as shown:
(1) `turn left` - turn 90 degrees left on the spot.
(2) `turn right` - turn 90 degrees right on the spot.
(3) `go forward` - move one square in the direction you are facing.
(4) `pick up` - pick up the object on the square directly in front of you. You can carry only one thing at a time.
(5) `drop` - put down what you are carrying, onto the square directly in front of you.
(6) `toggle` - open or close the door directly in front of you. This is also how a locked door is unlocked, while you are carrying a key of the same colour.
(7) `done` - signal that you believe the task is finished. It does not end the episode: a mission ends by being accomplished.

A thought is a line beginning `think:` followed by your reasoning. Use one when you need to plan, and then act on the next turn. Output exactly one line, and nothing else.

How to read what you see. Everything is described relative to you: `2 steps forward and 1 step to the left` counts squares from where you stand, forward being the way you are facing. You see only what is in front of you and unobstructed, so turning reveals new things and walls hide what is behind them. `a wall N steps forward` is how far you can advance before you are blocked.

Two things to keep in mind. You cannot move diagonally or sideways - to reach something off to one side, turn to face it first and then go forward. And an object occupies its square, so you cannot walk onto it: something `1 step forward` is already within reach of `pick up` or `toggle`. If a reply comes back beginning `Nothing happens.`, the action was wasted and repeating it will waste another turn.
"""
