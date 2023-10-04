from state import Token

def parse_file(file_path):
    # Store the parsed data
    x = []
    y = []
    
    # Open and read the file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            # Split the line into three parts: words, pos, and labels
            words_part, pos_part, labels_part = line.strip().split(" ||| ")
            
            # Split each part into individual elements
            words = words_part.split()
            pos = pos_part.split()
            labels = labels_part.split()
            
            # Ensure that words and pos have the same length
            if len(words) == len(pos):
                # create tokens
                tokens = [Token(idx, word, pos) for idx, (word, pos) in enumerate(zip(words, pos))]
                x.append(tokens)
                y.append(labels)
            else:
                print("Mismatched lengths in line:", line)
                break
    
    return x, y

def get_tag_dict(file_path):
    tag_dict = {}
    with open(file_path, 'r') as file:
        for idx, tag in enumerate(file.readlines()):
            tag_dict[tag.strip()] = idx
    return tag_dict