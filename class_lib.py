from os.path import join, abspath, exists, dirname, basename
from os import system, remove


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
        task_path = abspath(task_path)
        if not exists(task_path):
            print(f'Файл задачи {task_path} не найден.')
            return
        task_dir = dirname(task_path)
        task_file = basename(task_path)
        with open('tmp.bat', 'w') as bat:
            bat.write(f'cd {task_dir}\n')
            bat.write(f'{self.__solver_path}')
        with open(join(task_dir, 'logos_sa.params'), 'w') as logos_sa:
            logos_sa.write(self.logos_params.format(
                task_name=task_file,
                input_file=task_path
            ))
        print(f'Запуск задачи {task_path}')
        system('tmp.bat')
        remove('tmp.bat')


if __name__ == '__main__':
    curve = CurveCardFormer()
    print(curve)
    fix = SpcCardFormer(nid=2, dofs='xy')
    print(fix)
    motion = PrescribedMotionCardFormer()
    print(motion)
