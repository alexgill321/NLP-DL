from typing import List, Set
import os

class Token:
    def __init__(self, idx: int, word: str, pos: str, emb: List[float] = None):
        self.idx = idx # Unique index of the token
        self.word = word # Token string
        self.pos  = pos # Part of speech tag 
        self.emb  = emb # Embedding of the word (if available)

class DependencyEdge:
    def __init__(self, source: Token, target: Token, label:str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label  = label  # dependency label


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


def shift(state: ParseState, c=2) -> None:
    if len(state.parse_buffer) == c:
        pass
    else:
        next = state.parse_buffer.pop(0)
        state.stack.append(next)


def left_arc(state: ParseState, label: str, c=2) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.
    top = state.stack.pop()
    top2 = state.stack.pop()
    state.add_dependency(top, top2, label)
    state.stack.append(top)


def right_arc(state: ParseState, label: str, c=2) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.
    top = state.stack.pop()
    top2 = state.stack.pop()
    state.add_dependency(top2, top, label)
    state.stack.append(top2)
    
def is_legal(action: str, state: ParseState, c=2) -> bool:
    if action == "SHIFT":
        if len(state.parse_buffer) == c:
            return False
        else:
            return True
    elif action.startswith("REDUCE_L"):
        if len(state.stack) <= c+1:
            return False
        else:
            return True
    elif action.startswith("REDUCE_R"):
        if len(state.stack) <= c+1:
            return False
        else:
            return True
    else:
        return False


def is_final_state(state: ParseState, cwindow: int) -> bool:
    if len(state.parse_buffer) == cwindow and len(state.stack) == cwindow + 1:
        return True
    else:
        return False
    
def find_children(state:ParseState, cwindow: int) -> tuple:
    """
    Find the leftmost and rightmost child of a given token.

    Parameters:
    - token: The token for which to find the children.
    - dependencies: The list of DependencyEdge objects that define the dependencies between tokens.

    Returns:
    - A tuple (leftmost_child, rightmost_child), where each element is a Token object or None if no such child exists.
    """
    children = []
    labels = []
    for source in state.stack[-cwindow:]:
        leftmost_child = None
        rightmost_child = None
        for dep in state.dependencies:
            if dep.source == source:
                # Checking for leftmost child
                if leftmost_child is None or dep.target.idx < leftmost_child.target.idx:
                    leftmost_child = dep
                
                # Checking for rightmost child
                if rightmost_child is None or dep.target.idx > rightmost_child.target.idx:
                    rightmost_child = dep
        if leftmost_child is None:
            children.append('[PAD]')
        else:
            children.append(leftmost_child.target.word)
        if rightmost_child is None:
            children.append('[PAD]')
        else:
            children.append(rightmost_child.target.word)
        if leftmost_child is None:
            labels.append('NULL')
        else:
            labels.append(leftmost_child.label)
        if rightmost_child is None:
            labels.append('NULL')
        else:
            labels.append(rightmost_child.label)
    return children, labels

def generate_from_data(data, label_tags, pos_tags, c=2):
    """
    Generate training data for dependency parsing model.
    
    :param parsed_data: Parsed data containing words, POS tags, and labels.
    :param c: Number of elements to consider from the top of stack and start of buffer.
    :return: List of tuples containing input features (w, p) and output label.
    """
    words = []
    pos = []
    y = []

    for tokens, labels in data:
        # Initialize parse state
        stack = [Token(idx=-i-1, word="[PAD]", pos="NULL") for i in range(c)]
        parse_buffer = tokens.copy()
        ix = len(parse_buffer)
        parse_buffer.extend([Token(idx=ix+i+1, word="[PAD]", pos="NULL") for i in range(c)])
        dependencies = []
        state = ParseState(stack, parse_buffer, dependencies)

        # Apply actions according to labels and generate training data
        for label in labels:
            # Extract top c elements from stack and buffer and their corresponding POS tags
            w_stack = [t.word for t in state.stack[-c:]]
            w_stack.reverse()
            p_stack = [t.pos for t in state.stack[-c:]]
            p_stack.reverse()
            w_buff = [t.word for t in state.parse_buffer[:c]]
            p_buff = [t.pos for t in state.parse_buffer[:c]]
            
            w = w_stack + w_buff
            
            p = p_stack + p_buff            
            
            label_val = label_tags[label]
            pos_val = [pos_tags[p[i]] for i in range(len(p))]
            words.append(w)
            pos.append(pos_val)
            y.append(label_val)
            
            # Apply action to parse state
            if label == "SHIFT":
                shift(state)
            elif label.startswith("REDUCE_L"):
                left_arc(state, label)  # Extracting dependency type from label
            elif label.startswith("REDUCE_R"):
                right_arc(state, label)  # Extracting dependency type from label

    return words, pos, y

def generate_from_data_with_dep(data, label_tags, pos_tags, c=2):
    """
    Generate training data for dependency parsing model.
    
    :param parsed_data: Parsed data containing words, POS tags, and labels.
    :param c: Number of elements to consider from the top of stack and start of buffer.
    :return: List of tuples containing input features (w, p) and output label.
    """
    words = []
    pos = []
    y = []
    dep = []

    for tokens, labels in data:
        # Initialize parse state
        stack = [Token(idx=-i-1, word="[PAD]", pos="NULL") for i in range(c)]
        parse_buffer = tokens.copy()
        ix = len(parse_buffer)
        parse_buffer.extend([Token(idx=ix+i+1, word="[PAD]", pos="NULL") for i in range(c)])
        dependencies = []
        state = ParseState(stack, parse_buffer, dependencies)

        # Apply actions according to labels and generate training data
        for label in labels:
            # Extract top c elements from stack and buffer and their corresponding POS tags
            w_stack = [t.word for t in state.stack[-c:]]
            w_stack.reverse()
            p_stack = [t.pos for t in state.stack[-c:]]
            p_stack.reverse()
            w_buff = [t.word for t in state.parse_buffer[:c]]
            p_buff = [t.pos for t in state.parse_buffer[:c]]

            w_dep, dep_lab = find_children(state, c)
            
            w = w_stack + w_buff + w_dep
            
            p = p_stack + p_buff

            l = dep_lab            
            
            label_val = label_tags[label]
            l_val = [label_tags[l] for l in l]
            pos_val = [pos_tags[p[i]] for i in range(len(p))]
            words.append(w)
            pos.append(pos_val)
            dep.append(l_val)
            y.append(label_val)
            
            # Apply action to parse state
            if label == "SHIFT":
                shift(state)
            elif label.startswith("REDUCE_L"):
                left_arc(state, label)  # Extracting dependency type from label
            elif label.startswith("REDUCE_R"):
                right_arc(state, label)  # Extracting dependency type from label

    return words, pos, dep, y
