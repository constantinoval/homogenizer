import os
from typing import List
from class_lib import (CurveCardFormer, PrescribedMotionCardFormer,
                       SpcCardFormer, logos_helper, vol4points)
from math import sqrt
from shutil import rmtree, copyfile


def unistraintensor(val, idx):
    """
        idx=1,2,3,4,5,6
        strains:
                1 - xx
                2 - yy
                3 - zz
                4 - xy
                5 - xz
                6 - yz
    """
    # if idx>3: val/=2
    idxs = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
    rez = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    i = idxs[idx-1]
    rez[i[0]-1][i[1]-1] = val
    rez[i[1]-1][i[0]-1] = val
    return rez


def splitByNseq(s: str, n: List[int]) -> List[str]:
    """Разбивает строку s на список строк, длины которых указаны в списке n

    Args:
        s (str): входная строка
        n (List[int]): список длин фрагментов строки, если длина входной
                       строки больше чем сумма n, то последний элемент n
                       многократно повторяется

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

    def distance_to_node(self, other_node) -> float:
        return sqrt(
            (self.x-other_node.x)**2 +
            (self.y-other_node.y)**2 +
            (self.z-other_node.z)**2
        )

    @property
    def coordinates(self):
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'

    def __repr__(self) -> str:
        return self.__str__()


class element(object):
    def __init__(self, part: int = 1,
                 nodes: list = []):
        self.part = part
        self.nodes = nodes

    @property
    def nodes_count(self):
        return len(set(self.nodes))

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
            if ',' in l:
                n, *crds = l.split(',')
            else:
                n, *crds = splitByNseq(l, [8, 16])
            crds += (3-len(crds))*['0.0']
            self.nodes[int(n)] = node(*[float(c) for c in crds])
        for l in self.element_lines():
            if ',' in l:
                num, part, *nds = l.split(',')
            else:
                num, part, *nds = splitByNseq(l, [8])
            self.elements[int(num)] = element(part, [int(n) for n in nds])

    @property
    def bbox(self) -> List[float]:
        xmin = ymin = zmin = 1e6
        xmax = ymax = zmax = -1e6
        for n in self.nodes.values():
            xmin = min(xmin, n.x)
            ymin = min(ymin, n.y)
            zmin = min(zmin, n.z)
            xmax = max(xmax, n.x)
            ymax = max(ymax, n.y)
            zmax = max(zmax, n.z)
        return (xmin, ymin, zmin, xmax, ymax, zmax)

    def nodes_by_condition(self, cond: str) -> List[int]:
        rez = []
        for nnum, n in self.nodes.items():
            x = n.x
            y = n.y
            z = n.z
            check_cond = eval(cond)
            if check_cond is True:
                rez.append(nnum)
        return rez

    def calculate_volume(self, element_number):
        el = self.elements[element_number]
        nds = el.nodes
        V = 0
        nc = el.nodes_count
        if nc == 4:
            idxs = [[0, 1, 2, 3]]
        if nc == 6:
            idxs = [[0, 1, 3, 4],
                    [4, 1, 3, 2],
                    [4, 3, 2, 7]]
        if nc == 8:
            idxs = [[0, 4, 5, 7],
                    [2, 6, 5, 7],
                    [0, 2, 3, 7],
                    [0, 2, 5, 7],
                    [0, 1, 2, 5]]
        for idx in idxs:
            points = [self.nodes[nds[i]].coordinates for i in idx]
            V += vol4points(points)
        return V

    def calculate_volumes(self):
        print('Расчитываются объемы конечных элементов.')
        self.volums = {}
        for e in self.elements:
            self.volums[e] = self.calculate_volume(e)


class homogenizer(object):
    __slots__: ()

    def __init__(self,
                 input_file: str, logos_helper: logos_helper,
                 working_dir: str = os.curdir,
                 geom_tol: float = 1e-6,
                 strain_value=0.01):
        # путь к решателю
        self.__solver = logos_helper
        # величина деформации представительной ячейки
        self.__strain_value = strain_value
        # папка, где будут создаваться и решаться задачи на
        # элементарном объеме
        self.__working_dir = working_dir
        self.__input_file = os.path.abspath(input_file)
        self.__fe_model = None
        self.__common_data = None
        if not os.path.exists(input_file):
            print(f'Файл {input_file} не найден.')
            self.__input_dir = None
            self.__input_file = None
        else:
            print(f'Задача решается на базе модели {self.__input_file}.')
            self.__input_dir = os.path.dirname(self.__input_file)
            for l in open(self.__input_file, 'r').readlines():
                if l.startswith('MESH:'):
                    mesh_file = os.path.join(self.__input_dir, l[5:].strip())
                    self.read_model(mesh_file)
                if l.startswith('COMMON_DATA:'):
                    self.__common_data = os.path.join(
                        self.__input_dir, l[12:].strip())
                    if not os.path.exists(self.__common_data):
                        print(f'Файл {self.__common_data} не найден.')
                        self.__common_data = None
                    else:
                        print(f'Файл общих данных - {self.__common_data}')
        self.__geom_tol = geom_tol
        self.boundary_nodes = []
        self.created_tasks = {}
        self.solved_tasks = {}

    @property
    def working_dir(self) -> str:
        return self.__working_dir

    @property
    def nodes(self):
        return None if self.__fe_model is None else self.__fe_model.nodes

    @property
    def elements(self):
        return None if self.__fe_model is None else self.__fe_model.elements

    @property
    def bbox(self):
        return None if self.__fe_model is None else self.__fe_model.bbox

    @property
    def fe_model(self):
        return self.__fe_model

    def read_model(self, model_path) -> None:
        if os.path.exists(model_path):
            self.__fe_model = FE_model(model_path)
            self.__fe_model.read_mesh()
            self.__mesh_file = model_path
            self.__fe_model.calculate_volumes()
            print(f'Сетка прочитана из файла {model_path}.')
        else:
            print(f'Файл сетки {model_path} не найден.')
            self.__fe_model = None
            self.__mesh_file = None

    def find_boundary_nodes(self) -> None:
        self.boundary_nodes = set([])
        bbox = self.__fe_model.bbox
        for i, dof in enumerate(['x', 'y', 'z']):
            plane = self.__fe_model.nodes_by_condition(
                f'abs({dof}-{bbox[i]})<{self.__geom_tol}')
            self.boundary_nodes.update(plane)
            plane = self.__fe_model.nodes_by_condition(
                f'abs({dof}-{bbox[i+3]})<{self.__geom_tol}')
            self.boundary_nodes.update(plane)
        self.boundary_nodes = list(self.boundary_nodes)

    def form_bc_cards(self, unit_strain_id: int, lc_id=1, null_curve=2) -> List[str]:
        if not self.boundary_nodes:
            self.find_boundary_nodes()
        bc_cards = []
        n0 = self.__fe_model.nodes[self.boundary_nodes[0]]
        spc = SpcCardFormer(
            nid=self.boundary_nodes[0],
            dofs='xyz'
        )
        bc_cards.append(spc.header)
        bc_cards.append(str(spc))
        bc_cards.append(PrescribedMotionCardFormer.header)
        eps = unistraintensor(self.__strain_value, unit_strain_id)
        for nn in self.boundary_nodes[1:]:
            n1 = self.__fe_model.nodes[nn]
            for i in range(3):
                u = eps[i][0]*(n1.x-n0.x) + \
                    eps[i][1]*(n1.y-n0.y) + \
                    eps[i][2]*(n1.z-n0.z)
                if abs(u) < self.__geom_tol:
                    lc = null_curve
                    u = 1
                else:
                    lc = lc_id
                bc_cards.append(str(PrescribedMotionCardFormer(
                    nid=nn,
                    dof_type='u',
                    dof={0: 'x', 1: 'y', 2: 'z'}[i],
                    lc_id=lc,
                    sf=u
                )))
        return bc_cards

    def create_task(self, unit_strain_id):
        if self.__common_data is None:
            return
        if self.__mesh_file is None:
            return
        if self.__input_file is None:
            return
        task_dir = os.path.join(self.working_dir, f'case_{unit_strain_id}')
        if os.path.exists(task_dir):
            rmtree(task_dir)
        os.mkdir(task_dir)
        with open(os.path.join(task_dir, f'case_{unit_strain_id}.yaml'), 'w') as f:
            f.write(f'TITLE: case_{unit_strain_id}\n')
            f.write(f'MESH: case_{unit_strain_id}.k\n')
            f.write(f'COMMON_DATA: case_{unit_strain_id}.cd')
        copyfile(self.__mesh_file, os.path.join(
            task_dir, f'case_{unit_strain_id}.k'))
        input_file = open(self.__common_data, 'r')
        output_file = open(os.path.join(
            task_dir, f'case_{unit_strain_id}.cd'), 'w')
        for l in input_file:
            if 'CALC_STRATEGY:' in l:
                output_file.write(CurveCardFormer.header)
                output_file.write(str(CurveCardFormer(id=1, name='curve1')))
                output_file.write(
                    str(CurveCardFormer(id=2, y=[0, 0], name='curve2')))
                output_file.write(
                    ''.join(self.form_bc_cards(unit_strain_id, lc_id=1, null_curve=2)))
            output_file.write(l)
        input_file.close()
        output_file.close()
        self.created_tasks[f'case_{unit_strain_id}'] = {'name': f'case_{unit_strain_id}',
                                                        'task_dir': os.path.abspath(task_dir),
                                                        'unit_strain_id': unit_strain_id}

    def solve_tasks(self) -> None:
        for task in self.created_tasks.values():
            self.__solver.run_task(os.path.join(
                task['task_dir'], task['name']+'.yaml'))

    def do_result_homogenization(self, d3plot_path: str) -> None:
        pass


if __name__ == "__main__":
    lg = logos_helper(r"D:\programs\LOGOS\LOGOS\LOGOS-SA\Bin\Logos_SA.exe")
    h = homogenizer(
        r'D:\oneDrive\work\srv\2021\code\homogenizer\clear\test_task.yaml',
        logos_helper=lg)
    # h.create_task(1)
    # h.create_task(2)
    # print(h.created_tasks)
    # h.solve_tasks()
