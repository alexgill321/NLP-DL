from typing import List, Set
import os

class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx # Unique index of the token
        self.word = word # Token string
        self.pos  = pos # Part of speech tag 

class DependencyEdge:
    def __init__(self, source: Token, target: Token, label:str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label  = label  # dependency label
        pass


class ParseState:
    def __init__(self, stack: List[Token], parse_buffer: List[Token], dependencies: List[DependencyEdge]):
        self.stack = stack # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.parse_buffer = parse_buffer  # A buffer of token indices
        self.dependencies = dependencies

    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )


def shift(state: ParseState) -> None:

    next = state.parse_buffer.pop()
    state.stack.append(next)


def left_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.
    top = state.stack.pop()
    top2 = state.stack.pop()
    state.add_dependency(top, top2, label)
    state.stack.append(top)


def right_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.
    top = state.stack.pop()
    top2 = state.stack.pop()
    state.add_dependency(top2, top, label)
    state.stack.append(top2)



def is_final_state(state: ParseState, cwindow: int) -> bool:
    if len(state.parse_buffer) == 0 and len(state.stack) == cwindow - 1:
        return True

def generate_from_data(token_list, label_list, label_tags, c=2):
    """
    Generate training data for dependency parsing model.
    
    :param parsed_data: Parsed data containing words, POS tags, and labels.
    :param c: Number of elements to consider from the top of stack and start of buffer.
    :return: List of tuples containing input features (w, p) and output label.
    """
    data = []

    for tokens, labels in zip(token_list, label_list):
        # Initialize parse state
        stack = []
        parse_buffer = tokens.copy()
        dependencies = []
        state = ParseState(stack, parse_buffer, dependencies)

        # Apply actions according to labels and generate training data
        for label in labels:
            # Extract top c elements from stack and buffer and their corresponding POS tags
            w = [t.word for t in stack[-c:]] + [t.word for t in parse_buffer[:c]]
            p = [t.pos for t in stack[-c:]] + [t.pos for t in parse_buffer[:c]]
            
            # Pad w and p if necessary
            w = (['[PAD]'] * (2*c - len(w))) + w
            p = (['NULL'] * (2*c - len(p))) + p
            
            label_val = label_tags[label]
            data.append((w, p, label_val))
            
            # Apply action to parse state
            if label == "SHIFT":
                shift(state)
            elif label.startswith("REDUCE_L"):
                left_arc(state, label.split("_")[1])  # Extracting dependency type from label
            elif label.startswith("REDUCE_R"):
                right_arc(state, label.split("_")[1])  # Extracting dependency type from label

    return data
