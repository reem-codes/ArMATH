import re
from copy import deepcopy


class Et:
    """ An expression tree node"""
    def __init__(self, value):
        """Constructor to create a node"""
        self.value = value
        self.left = None
        self.right = None


def from_infix_to_prefix(expression):
    """transform math expression from infix order to prefix order"""
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res


def compute_prefix_expression(pre_fix):
    """ Given prefix expression, compute answer """
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def convert_prefix_to_infix(l):
    stack = []
    l = list(l)
    l.reverse()
    try:
        for p in l:
            if p in ['-',  '/',  '*',  '+',  '^']:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append('(%s %s %s)' % (op1, p, op2))
            else:
                stack.append(p)
        return stack.pop()
    except:
        l.reverse()
        return str(l)