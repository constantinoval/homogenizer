import os
from typing import List
from math import sqrt
from shutil import rmtree, copyfile
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import lsreader as ls
import uuid
from subprocess import Popen, DEVNULL


def vol4points(points) -> float:
    """calculate volume of tetra using
       coordinates of it's 4 points

    Args:
        points (tuple of point coordinates in 3D)

    Returns:
        float: volume
    """
    m = [[a1-a0 for a1, a0 in zip(points[i+1], points[0])]
         for i in range(3)]
    d = m[0][0]*m[1][1]*m[2][2] + \
        m[0][1]*m[1][2]*m[2][0] + \
        m[1][0]*m[2][1]*m[0][2] - \
        m[2][0]*m[1][1]*m[0][2] - \
        m[1][0]*m[0][1]*m[2][2] - \
        m[2][1]*m[1][2]*m[0][0]
    return abs(d)/6.


def invM(C):
    """
        Функция обращения матрицы 6x6
    """
    C = deepcopy(C)
    S = []
    for i in range(6):
        S.append([0]*i+[1.0]+[0]*(5-i))
    for i in range(6):
        tmp = C[i][i]
        for j in range(6):
            C[i][j] /= tmp
            S[i][j] /= tmp
        for k in range(i+1, 6):
            tmp = C[k][i]
            for j in range(6):
                C[k][j] -= C[i][j]*tmp
                S[k][j] -= S[i][j]*tmp

    for i in range(5, -1, -1):
        for k in range(i-1, -1, -1):
            tmp = C[k][i]
            for j in range(6):
                C[k][j] -= C[i][j]*tmp
                S[k][j] -= S[i][j]*tmp
    return S


class CurveCardFormer:
    header = "DEFINE_CURVE:\n"
    card_format = """   -
    NAME: {name:s}
    SIDR: 0
    LCID: {id:d}
    SFA: {sfx:e}
    SFO: {sfy:e}
    OFFA: 0
    OFFO: 0
    DATTYP: 0
    A1:
{a_lines:s}    O1:
{o_lines:s}    UNIT_A: ""
    UNIT_O: ""
"""
    line_format = '      - {x:e}\n'

    def __init__(self, id=1, name='curve', x=[0, 1], y=[0, 1], sfx=1, sfy=1):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.sfx = sfx
        self.sfy = sfy
        self.form_table_lines()

    def form_table_lines(self):
        self.a_lines = ''
        self.o_lines = ''
        for xx in self.x:
            self.a_lines += self.line_format.format(x=xx)
        for yy in self.y:
            self.o_lines += self.line_format.format(x=yy)

    def __repr__(self):
        return self.card_format.format(
            id=self.id,
            name=self.name,
            sfx=self.sfx,
            sfy=self.sfy,
            a_lines=self.a_lines,
            o_lines=self.o_lines
        )

    def __str__(self):
        return self.__repr__()


class SpcCardFormer:
    header = "BOUNDARY_SPC_NODE:\n"
    card_format = """   -
    NID: {nid:d}
    CID: 0
    DOFX: {dofx:d}
    DOFY: {dofy:d}
    DOFZ: {dofz:d}
    DOFRX: 0
    DOFRY: 0
    DOFRZ: 0
    SK: 1
"""

    def __init__(self, nid=1, dofs='xyz') -> None:
        self.nid = nid
        self.dofx = 1 if 'x' in dofs else 0
        self.dofy = 1 if 'y' in dofs else 0
        self.dofz = 1 if 'z' in dofs else 0

    def __repr__(self):
        return self.card_format.format(
            nid=self.nid,
            dofx=self.dofx,
            dofy=self.dofy,
            dofz=self.dofz
        )

    def _str__(self):
        return self.__repr__()


class PrescribedMotionCardFormer:
    # dof_type = [v=0, a=1, u=2]
    # direction = [x=0, y=1, z=2]
    header = "BOUNDARY_PRESCRIBED_MOTION_NODE:\n"
    card_format = """   -
    NID: {nid:d}
    DOF: {dof:d}
    VAD: {dof_type:d}
    LCID: {lc_id:d}
    SF: {sf:e}
    VID: 0
    VX: {x:e}
    VY: {y:e}
    VZ: {z:e}
    SK: 1
"""

    def __init__(self, nid=1, dof_type='u', dof='x',
                 lc_id=1, sf=1, components=[0, 0, 0]) -> None:
        self.nid = nid
        self.dof = {'x': 1, 'y': 2, 'z': 3, 'vec': -4}[dof]
        self.dof_type = {'v': 0, 'a': 1, 'u': 2}[dof_type]
        self.sf = sf
        self.lc_id = lc_id
        self.components = components

    def __repr__(self):
        return self.card_format.format(
            nid=self.nid,
            dof=self.dof,
            dof_type=self.dof_type,
            lc_id=self.lc_id,
            sf=self.sf,
            x=self.components[0],
            y=self.components[1],
            z=self.components[2]
        )

    def _str__(self):
        return self.__repr__()


class logos_helper:
    logos_params = """task {task_name}
num      000
obschet  0
stepout  1000
outfmt   1
geom     3
sa_paral 4
nthreads 1
input    {input_file}
"""

    def __init__(self, solver_path):
        self.__solver_path = solver_path

    def run_task(self, task_path):
        task_path = os.path.abspath(task_path)
        if not os.path.exists(task_path):
            print(f'Файл задачи {task_path} не найден')
            return
        task_dir = os.path.dirname(task_path)
        task_file = os.path.basename(task_path)
        bat_name = str(uuid.uuid4())+'.bat'
        with open(bat_name, 'w') as bat:
            bat.write(f'cd {task_dir}\n')
            bat.write(f'{self.__solver_path}\n')
        with open(os.path.join(task_dir, 'logos_sa.params'), 'w') as logos_sa:
            logos_sa.write(self.logos_params.format(
                task_name=task_file,
                input_file=task_path
            ))
        print(f'Запуск задачи {task_path}')
        p = Popen(bat_name, stdout=DEVNULL)
        p.communicate()
        os.remove(bat_name)
        print(f'Задача {task_path} решена')

    def get_solid_results(self, data_path):
        if not os.path.exists(data_path):
            print(
                f'Не возможно обработать результаты. Файл {data_path} не найден')
            return {}
        data_path = os.path.normpath(data_path)
        dr = ls.D3plotReader(data_path)
        # num_solids = dr.get_data(ls.DataType.D3P_NUM_SOLID, ist=0)
        num_states = dr.get_data(ls.DataType.D3P_NUM_STATES)
        has_stresses = dr.get_data(ls.DataType.D3P_HAS_SOLID_STRESS)
        has_strains = dr.get_data(ls.DataType.D3P_HAS_STRAIN)
        solid_ids = dr.get_data(ls.DataType.D3P_SOLID_IDS)
        if has_strains:
            strains = dr.get_data(
                ls.DataType.D3P_SOLID_STRAIN, ist=num_states-1)
        if has_stresses:
            stresses = dr.get_data(
                ls.DataType.D3P_SOLID_STRESS, ist=num_states-1)
        rez = {}
        for n, ids in enumerate(solid_ids):
            rez[ids] = {}
            if has_stresses:
                s = [stresses[n].x(),
                     stresses[n].y(),
                     stresses[n].z(),
                     stresses[n].xy(),
                     stresses[n].zx(),
                     stresses[n].yz()]
                rez[ids]['s'] = s
            if has_strains:
                e = [strains[n].x(),
                     strains[n].y(),
                     strains[n].z(),
                     strains[n].xy(),
                     strains[n].zx(),
                     strains[n].yz()]
                rez[ids]['e'] = e
        return rez


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
        print('Расчитываются объемы конечных элементов')
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
            print(f'Файл {input_file} не найден')
            self.__input_dir = None
            self.__input_file = None
        else:
            print(f'Задача решается на базе модели {self.__input_file}')
            self.__input_dir = os.path.dirname(self.__input_file)
            for l in open(self.__input_file, 'r').readlines():
                if l.startswith('MESH:'):
                    mesh_file = os.path.join(self.__input_dir, l[5:].strip())
                    self.read_model(mesh_file)
                if l.startswith('COMMON_DATA:'):
                    self.__common_data = os.path.join(
                        self.__input_dir, l[12:].strip())
                    if not os.path.exists(self.__common_data):
                        print(f'Файл {self.__common_data} не найден')
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
            print(f'Сетка прочитана из файла {model_path}')
        else:
            print(f'Файл сетки {model_path} не найден')
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

    def create_task(self, unit_strain_id, clear_old=True):
        if self.__common_data is None:
            return
        if self.__mesh_file is None:
            return
        if self.__input_file is None:
            return
        task_dir = os.path.join(self.working_dir, f'case_{unit_strain_id}')
        if clear_old:
            if os.path.exists(task_dir):
                rmtree(task_dir)
        if not os.path.exists(task_dir):
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
        self.solved_tasks = {}
        for task in self.created_tasks.values():
            self.__solver.run_task(os.path.join(
                task['task_dir'], task['name']+'.yaml'))
            self.solved_tasks[task['name']] = {
                'task_dir': task['task_dir'], 'unit_strain_id': task['unit_strain_id']}

    def solve_tasks_parallel(self, num_workers=2):
        self.solved_tasks = {}
        tasks_pool = []
        for task in self.created_tasks.values():
            self.solved_tasks[task['name']] = {
                'task_dir': task['task_dir'], 'unit_strain_id': task['unit_strain_id']}
            tasks_pool.append(os.path.join(
                task['task_dir'], task['name']+'.yaml'))

        executor = ProcessPoolExecutor(max_workers=num_workers)
        for e in executor.map(self.__solver.run_task, tasks_pool):
            pass

    def process_results_one_case(self, task_path: str):
        d3plot_path = glob(f'{task_path}'+os.path.sep +
                           '*.RESULTS/D3PLOT/d3plot')
        if len(d3plot_path):
            d3plot_path = d3plot_path[0]
        else:
            print(f'Решение по пути {task_path} найти не удалось')
        rez = self.__solver.get_solid_results(os.path.abspath(d3plot_path))
        V = 0
        mean_stress = [0, 0, 0, 0, 0, 0]
        mean_strain = [0, 0, 0, 0, 0, 0]
        for eid, vol in self.fe_model.volums.items():
            if 's' in rez[eid]:
                for i in range(6):
                    mean_stress[i] += rez[eid]['s'][i]*vol
            if 'e' in rez[eid]:
                for i in range(6):
                    mean_strain[i] += rez[eid]['e'][i]*vol
            V += vol
        for i in range(6):
            mean_stress[i] /= V
            mean_strain[i] /= V
        return mean_stress, mean_strain

    def process_results(self):
        self.mean_stresses = []
        self.mean_strains = []
        for task in self.solved_tasks.values():
            ms, me = self.process_results_one_case(task['task_dir'])
            self.mean_stresses.append(ms)
            self.mean_strains.append(me)

    def calculate_effective_moduli(self):
        if self.mean_stresses is None:
            print('Не посчитаны средние напряжения')
            return
        self.D = deepcopy(self.mean_stresses)
        for i in range(6):
            for j in range(6):
                self.D[i][j] /= self.__strain_value
        self.S = invM(self.D)

    def calculate_engeenering_constants(self):
        self.eng_moduli = {}
        self.eng_moduli['Ex'] = 1/self.S[0][0]
        self.eng_moduli['Ey'] = 1/self.S[1][1]
        self.eng_moduli['Ez'] = 1/self.S[2][2]

        self.eng_moduli['nuxy'] = -self.S[0][1]/self.S[0][0]
        self.eng_moduli['nuxz'] = -self.S[0][2]/self.S[0][0]
        self.eng_moduli['nuyz'] = -self.S[1][2]/self.S[1][1]

        self.eng_moduli['nuyx'] = -self.S[1][0]/self.S[1][1]
        self.eng_moduli['nuzx'] = -self.S[2][0]/self.S[2][2]
        self.eng_moduli['nuzy'] = -self.S[2][1]/self.S[2][2]

        self.eng_moduli['Gxy'] = 0.5/self.S[3][3]
        self.eng_moduli['Gxz'] = 0.5/self.S[4][4]
        self.eng_moduli['Gyz'] = 0.5/self.S[5][5]

        # self.eng_moduli['nuyx'] = self.eng_moduli['nuxy'] * \
        #     self.eng_moduli['Ey']/self.eng_moduli['Ex']
        # self.eng_moduli['nuzx'] = self.eng_moduli['nuxz'] * \
        #     self.eng_moduli['Ez']/self.eng_moduli['Ex']
        # self.eng_moduli['nuzy'] = self.eng_moduli['nuyz'] * \
        #     self.eng_moduli['Ez']/self.eng_moduli['Ey']

    def do_homogenization(self, max_workers=2) -> None:
        for i in range(6):
            self.create_task(i+1, clear_old=True)
        self.solve_tasks_parallel(max_workers)
        self.process_results()
        self.calculate_effective_moduli()
        self.calculate_engeenering_constants()


if __name__ == "__main__":
    lg = logos_helper(r"D:\programs\LOGOS\LOGOS\LOGOS-SA\Bin\Logos_SA.exe")
    h = homogenizer(
        r'D:\oneDrive\work\srv\2021\code\homogenizer\clear\test_task.yaml',
        logos_helper=lg)
    h.do_homogenization(6)
    print(h.eng_moduli)
