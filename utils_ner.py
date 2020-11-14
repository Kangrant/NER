def get_entity_bios(seq,id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]

        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1] #chunk [type,3,3]
            chunks.append(chunk)
            chunk = (-1, -1, -1)

        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]  #chunk [type,0,-1]

        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]: #如果后缀相同，则记录下来
                chunk[2] = indx
            if indx == len(seq) - 1: #如果是最后一个标签了，就直接append
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bieos(seq,id2label):
    """Gets entities from sequence.
        note: BIOS
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).

        Example:
            # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            # >>> get_entity_bios(seq)
            [['PER', 0,1], ['LOC', 3, 3]]
        """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]

        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag  # chunk [S-a,3,3]
            chunks.append(chunk)
            chunk = (-1, -1, -1)

        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[-1]  # chunk [type,0,-1]

        elif tag.startswith('I-')  and chunk[1] != -1:
            _type = tag.split('-')[-1]
            if _type == chunk[0]:  # 如果后缀相同，则记录下来
                #chunk[2] = indx
                continue
            else:
                chunk = [-1,-1,-1]

        elif tag.startswith('E-'):
            _type = tag.split('-')[-1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1 and chunk[2] != -1:  # 如果是最后一个标签了，就直接append
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup='bieos'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios','bieos']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    elif markup=='bios':
        return get_entity_bios(seq,id2label)
    else:
        return get_entity_bieos(seq,id2label)

if __name__ == '__main__':

    seq = ['B-a','E-a','O','B-fac-exp','I-fac-exp','E-fac-exp','O','S-a','S-e']
    id2label = []
    res = get_entity_bieos(seq,id2label)
    print(res)


