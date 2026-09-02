_walkthrough_example = """<< Chief's office >>
You are standing in the Chief's office. He is telling you "The Mayor was murdered yeaterday night at 12:03 am. I want you to solve it before we get any bad publicity or the FBI has to come in. "Yessir!" You reply. He hands you a sheet of paper. Once you have read it, go north or west.

You can see a piece of white paper here.
> think: The chief handed me a sheet of paper and there is one on the floor here. Picking things up is usually worth points, so I will take it before I leave.
OK.
> take paper
Taken.

[Your score has just gone up by ten points.]
> think: That scored. The room description says the exits are north and west, so I will try west first and come back for north.
OK.
> west
<< Closet >>
You are in a closet. There is a gun on the floor. Better get it. To exit, go east.

You can see a small black pistol here.
> think: The game is telling me outright to take the gun, and a detective will need it. The only exit is east, so I will take it and go back.
OK.
> take gun
Taken.

[Your score has just gone up by ten points.]
> east
<< Chief's office >>
You are standing in the Chief's office. He is telling you "The Mayor was murdered yeaterday night at 12:03 am. I want you to solve it before we get any bad publicity or the FBI has to come in.
> think: I have taken everything here. North was the other exit from this room, so that is where the case continues.
OK.
> north
<< Outside >>
You are outside in the cold. To the east is a dead end. To the west is the rest of the street. Papers are blowing around. It's amazingly cold for this time of year.
> think: The description says east is a dead end, so west is the only direction worth spending a move on.
OK.
> west
<< Outside >>
You are still on the streets. To the north is a restraunt where the mayor ate often. To the east is the Mayor's home.

[Your score has just gone up by ten points.]"""

jericho_few_shots: list[str] = [_walkthrough_example]

jericho_solver_system_prompt: str = """
You are playing a work of interactive fiction. Each turn you read what the game prints and reply with one line: either a command for the game's parser, or a thought.

- A command is plain imperative English, two or three words: `take lamp`, `open mailbox`, `north`, `unlock door with key`, `look`, `inventory`. The compass directions may be abbreviated to `n`, `s`, `e`, `w`, `ne`, `nw`, `se`, `sw`, `up`, `down`.
- A thought is a line beginning `think:` followed by your reasoning. Use one when you need to plan, and then act on the next turn.
- Output exactly one line, and nothing else. No numbering, no quotes, no explanation around the command.

The parser is unforgiving and its vocabulary is small. If it answers `I don't know the word "..."` then that word is not in the game and no rephrasing of it will work - use a different word entirely. If it answers `You can't see any ... here!` then the thing is not in this room, so go and find it rather than repeating the command.

You score points for making real progress: reaching new places, solving puzzles, and picking up the objects the game cares about. Read each room description for its exits and its objects, keep track of where you have been, and take what you can carry. Your objective is to score as many points as you can before you run out of turns.
"""
