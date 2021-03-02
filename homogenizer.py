import os
from typing import List


def splitByNseq(s: str, n: List[int]) -> List[str]:
    """Разбивает строку s на список строк, длины которых указаны в списке n

    Args:
        s (str): входная строка
        n (List[int]): список длин фрагментов строки, если длина входной строки больше чем сумма n, то последний элемент n многократно повторяется

    Returns:
        List[str]: списко фрагментов строки s

    Пример s='123456789', n=[2,3] -> ['12', '345', '678']
    """
    while sum(n)+n[-1] <= len(s):
        n.append(n[-1])
    pos = [sum(n[:i+1]) for i in range(len(n))]
    # if pos[-1] != len(s):
    #     pos.append(len(s))
    pos.insert(0, 0)
    rez = []
    for i in range(len(pos)-1):
        rez.append(s[pos[i]:pos[i+1]].strip())
    return rez


class node(object):
    def __init__(self, x: float = 0.0,
                 y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'

    def __repr__(self) -> str:
        return self.__str__()


class element(object):
    def __init__(self, part: int = 1,
                 nodes: list = []):
        self.part = part
        self.nodes = nodes

    def __str__(self) -> str:
        return f'part = {self.part}, nodes: {self.nodes}'

    def __repr__(self) -> str:
        return self.__str__()


class FE_model(object):
    def __init__(self, mesh_path: str) -> None:
        self.__mesh_path = mesh_path
        self.nodes = {}
        self.elements = {}

    def node_lines(self):
        with open(self.__mesh_path, 'r') as f:
            node_mode = False
            for l in f:
                if l.startswith("$"):
                    continue
                if l.startswith("*"):
                    node_mode = l.upper().strip() == '*NODE'
                    continue
                if node_mode:
                    yield l

    def element_lines(self):
        with open(self.__mesh_path, 'r') as f:
            element_mode = False
            for l in f:
                if l.startswith("$"):
                    continue
                if l.startswith("*"):
                    element_mode = l.upper().strip() == '*ELEMENT_SOLID'
                    continue
                if element_mode:
                    yield l

    def read_mesh(self):
        for l in self.node_lines():
            n, *crds = splitByNseq(l, [8, 16])
            crds += (3-len(crds))*['0.0']
            self.nodes[int(n)] = node(*[float(c) for c in crds])
        for l in self.element_lines():
            num, part, *nds = splitByNseq(l, [8])
            self.elements[int(num)] = element(part, [int(n) for n in nds])


class homogenizer(object):
    __slots__: ()

    def __init__(self, working_dir: str = os.curdir):
        self.__solver = 'logos_sa.exe'
        self.__working_dir = working_dir
        self.__fe_model = None

    @property
    def working_dir(self) -> str:
        return self.__working_dir

    @property
    def nodes(self):
        return None if self.__fe_model is None else self.__fe_model.nodes

    @property
    def elements(self):
        return None if self.__fe_model is None else self.__fe_model.elements

    def read_model(self, model_path) -> None:
        self.__fe_model = FE_model(model_path)
        self.__fe_model.read_mesh()

    def upply_bc(self, test_id: int) -> None:
        pass

    def solve_task(self, test_id: int) -> None:
        pass

    def do_result_homogenization(self, d3plot_path: str) -> None:
        pass


if __name__ == "__main__":
    h = homogenizer()
    print(h.nodes)
    h.read_model('./test.k')
    print(h.nodes)
    print(h.elements)
    # print(splitByNseq('123456789', [2, 3]))
