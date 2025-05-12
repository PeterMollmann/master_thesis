import sympy
import copy


class sympyConversion():
    def __init__(self, tree, best_ind_consts=None):
        self.tree = tree
        self.best_ind_consts = best_ind_consts

    def convert_inverse_prim(self, prim, args):
        prim = copy.copy(prim)
        converter = {
            'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
            'protectedDiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
            'div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
            'mul': lambda *args_: "Mul({},{})".format(*args_),
            'add': lambda *args_: "Add({},{})".format(*args_),
            'square': lambda *args_: "Pow({}, 2)".format(*args_),
            'sqrt': lambda *args_: "Pow({}, 0.5)".format(*args_)
        }
        prim_formatter = converter.get(prim.name, prim.format)

        return prim_formatter(*args)

    def stringify_for_sympy(self):
        string = ""
        stack = []
        for node in self.tree:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = self.convert_inverse_prim(prim, args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)
        return string

    def insert_str_consts(self, str_expr):
        """ Returns string expression with constants inserted if constants are pressent
        """
        idx = ([idx for idx, char in enumerate(str_expr)
               if char == 'a' and str_expr[idx-1] != 't' and str_expr[idx+1] != 'n'])
        list_expr = list(str_expr)
        for i, j in enumerate(idx):
            list_expr[j] = str(self.best_ind_consts[i])
            new_str_expr = ''.join(list_expr)
        return new_str_expr

    def simplify_expr(self):
        """ Simplify string expression
        """
        str_to_simplify = self.stringify_for_sympy()
        if len(self.best_ind_consts) > 0:
            new_str = self.insert_str_consts(str_to_simplify)
            # return sympy.expand(sympy.simplify(new_str))
            return sympy.simplify(new_str)
        else:
            # return sympy.expand(sympy.simplify(str_to_simplify))
            return sympy.simplify(str_to_simplify)

    def unNorm_expr(self, norm_str: str):

        # norm_str: string that is used as normalisation factor.

        str_to_simplify = self.stringify_for_sympy()

        if len(self.best_ind_consts) > 0:
            new_str = sympy.sympify(self.insert_str_consts(str_to_simplify))
            return sympy.expand(sympy.simplify(new_str / sympy.sympify(norm_str)))
        else:
            return sympy.expand(sympy.simplify(str_to_simplify / sympy.sympify(norm_str)))
