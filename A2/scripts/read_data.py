from state import Token
import torchtext

def parse_file(file_path):
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words_part, pos_part, labels_part = line.strip().split(" ||| ")
            words = words_part.split()
            pos = pos_part.split()
            labels = labels_part.split()
            
            if len(words) == len(pos):
                tokens = [Token(idx, word, pos) for idx, (word, pos) in enumerate(zip(words, pos))]
                data.append((tokens, labels))
            else:
                print("Mismatched lengths in line:", line)
                break
    
    return data

def emb_parse(file_path, name='840B', d_emb=300, c=2):
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words_part, pos_part, labels_part = line.strip().split(" ||| ")
            words = words_part.split()
            pos = pos_part.split()
            labels = labels_part.split()
            if len(words) == len(pos):
                w_emb = torchtext.vocab.GloVe(name=name, dim=d_emb)
                emb_list = w_emb.get_vecs_by_tokens(words, lower_case_backup=True)
                tokens = [Token(idx, word, pos, emb) for idx, (word, pos, emb) in enumerate(zip(words, pos, emb_list))]
                data.append((tokens, labels))
            else:
                print("Mismatched lengths in line:", line)
                break
    return data

def get_tag_dict(file_path):
    tag_dict = {}
    with open(file_path, 'r') as file:
        for idx, tag in enumerate(file.readlines()):
            tag_dict[tag.strip()] = idx
    return tag_dict

def get_tag_dict_rev(file_path):
    tag_dict = {}
    with open(file_path, 'r') as file:
        for idx, tag in enumerate(file.readlines()):
            tag_dict[idx] = tag.strip()
    return tag_dict