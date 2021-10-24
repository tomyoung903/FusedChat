
"appended.json" contains "TOD + ODD" sessions while "prepended.json" contains "ODD + TOD sessions".

"appended.json" and "prepended.json" are both dictionaries. Each item is a dialogue. The key is the original dialogue id used in MultiWOZ. The value is another dictionary containing the following items:
*   **creator**: Which creator wrote the dialogue. The creator id is in the range of (0, 112).
*   **turns**: A list of dialogue turns.
*   **types**: A list of types of **turns** (**types** is the same length as **turns**). 
    A type can be one of the following 4:
    *   **original** - This dialogue turn is the same as the original turn in MultiWOZ.
    *   **rewritten** - This dialogue turn corresponds to an original turn in MultiWOZ. However, the turn has been rewritten in FusedChat. If the turn is in an "TOD + ODD" dialogue session (appending), it was rewritten to promote more natural dialogue flow. If the turn is in an "ODD + TOD" dialogue session (prepending), it was rewritten to model inter-mode dependency.
    *   **prepended** - Only exists in "prepended.json". This dialogue turn is a new ODD turn added by FusedChat.
    *   **appended** - Only exists in "appended.json". This dialogue turn is a new ODD turn added by FusedChat.
In the case of "appended.json", we also note down
    *   **num_rewritten**: How many dialogue turns have been rewritten. It could be 0. (In the case of "prepended.json", it is always 1, therefore we do not note it down)
    *   **num_disarded**: How many turns have been discarded from the end of the dialogue. (We heuristically discard dialogue turns that feature conversation ending patterns such as dialogue-act:thank-you and dialogue-act:goodbye)