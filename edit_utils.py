def get_span(orig, new, editType):
    orig_list = orig.split(" ")
    new_list = new.split(" ")
    
    flag = False # this indicate whether the actual edit follow the specified editType
    if editType == "deletion":
        assert len(orig_list) > len(new_list), f"the edit type is deletion, but new is not shorter than original:\n new: {new}\n orig: {orig}"
        diff = len(orig_list) - len(new_list)
        for i, (o, n) in enumerate(zip(orig_list, new_list)):
            if o != n: # assume the index of the first different word is the starting index of the orig_span
            
                orig_span = [i, i + diff - 1] # assume that the indices are starting and ending index of the deleted part
                new_span = [i-1, i] # but for the new span, the starting and ending index is the two words that surround the deleted part
                flag = True
                break


    elif editType == "insertion": 
        assert len(orig_list) < len(new_list), f"the edit type is insertion, but the new is not longer than the original:\n new: {new}\n orig: {orig}"
        diff = len(new_list) - len(orig_list)
        for i, (o, n) in enumerate(zip(orig_list, new_list)):
            if o != n: # insertion is just the opposite of deletion
                new_span = [i, i + diff - 1] # NOTE if only inserted one word, s and e will be the same
                orig_span = [i-1, i]
                flag = True
                break

    elif editType == "substitution":
        new_span = []
        orig_span = []
        for i, (o, n) in enumerate(zip(orig_list, new_list)):
            if o != n:
                new_span = [i]
                orig_span = [i]
                break
        assert len(new_span) == 1 and len(orig_span) == 1, f"new_span: {new_span}, orig_span: {orig_span}"
        for j, (o, n) in enumerate(zip(orig_list[::-1], new_list[::-1])):
            if o != n:
                new_span.append(len(new_list) - j -1)
                orig_span.append(len(orig_list) - j - 1)
                flag = True
                break
    else:
        raise RuntimeError(f"editType unknown: {editType}")

    if not flag:
        raise RuntimeError(f"wrong editing with the specified edit type:\n original: {orig}\n new: {new}\n, editType: {editType}")

    return orig_span, new_span    