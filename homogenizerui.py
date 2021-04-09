import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfile, askdirectory, askopenfilename
import os
from time import sleep
import json
from homogenizer import homogenizer, logos_helper


class HomogenizerUI:
    def __init__(self, master=None):
        self.toplevel1 = tk.Tk() if master is None else tk.Toplevel(master)
        self.toplevel1.title('Homogenizer')
        self.toplevel1.resizable(False, False)
        self.frame1 = tk.Frame(self.toplevel1)
        self.frame1.pack(fill='both', pady=10)
        tk.Grid.columnconfigure(self.frame1, 1, weight=1)

        self.label1 = ttk.Label(
            self.frame1, text="Путь к файлу logos_sa.exe",
            font='{Times New Roman} 10 {bold}')
        self.label1.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.LogosPathVar = tk.StringVar()
        self.eLogosPath = ttk.Entry(self.frame1, width=50,
                                    state='readonly',
                                    textvariable=self.LogosPathVar)
        self.eLogosPath.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5)
        self.eLogosPath.bind('<Double-Button-1>', self.ChooseLogosPath)

        self.label2 = ttk.Label(
            self.frame1, text="Путь к файлу проекта  (yaml)",
            font='{Times New Roman} 10 {bold}')
        self.label2.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.YamlPathVar = tk.StringVar()
        self.YamlPath = ttk.Entry(self.frame1, state='readonly',
                                  textvariable=self.YamlPathVar)
        self.YamlPath.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5)
        self.YamlPath.bind('<Double-Button-1>', self.ChooseYamlPath)

        self.label3 = ttk.Label(
            self.frame1, text="Рабочий каталог",
            font='{Times New Roman} 10 {bold}')
        self.label3.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        self.WorkingDirVar = tk.StringVar()
        self.WDPath = ttk.Entry(self.frame1, state='readonly',
                                textvariable=self.WorkingDirVar)
        self.WDPath.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5)
        self.WDPath.bind('<Double-Button-1>', self.ChooseWorkingDir)

        self.StrainFrame = tk.Frame(self.frame1)
        self.StrainFrame.grid(row=3, column=0, sticky=tk.W+tk.E)
        self.label4 = ttk.Label(
            self.StrainFrame, text="Деформация")
        self.label4.pack(side='left', padx=5, pady=5)

        self.StrainVar = tk.StringVar()
        self.eStrain = ttk.Entry(self.StrainFrame,
                                 textvariable=self.StrainVar)
        self.eStrain.pack(side='left', fill='y', padx=5, pady=5)
        self.StrainVar.set('0.01')

        self.frame3 = tk.Frame(self.frame1)
        self.frame3.grid(row=3, column=1, sticky=tk.W+tk.E)
        self.ParallelVar = tk.IntVar(value=1)
        self.cbParallel = ttk.Checkbutton(self.frame3,
                                          text='Параллельные вычисления',
                                          onvalue=1,
                                          offvalue=0,
                                          variable=self.ParallelVar)
        self.ParallelVar.set(0)
        self.cbParallel.pack(side='left', padx=5, pady=5)
        self.MaxWorkersVar = tk.StringVar()
        self.sbMaxWorkers = ttk.Spinbox(self.frame3,
                                        width=3,
                                        from_=1,
                                        to=6, increment=1,
                                        textvariable=self.MaxWorkersVar)
        self.MaxWorkersVar.set('2')
        self.sbMaxWorkers.pack(side='left')
        self.button1 = ttk.Button(self.frame3, text='Рассчитать')
        self.button1.pack(side='right', padx=20)
        self.button1.bind('<Button-1>', self.Calculate)

        self.frame2 = tk.Frame(self.toplevel1)
        self.frame2.pack(fill='both', padx=10, pady=10)
        self.text1 = tk.Text(self.frame2)
        self.text1.pack(side='left', fill='both')
        self.scrol1 = tk.Scrollbar(self.frame2, orient='vertical',
                                   command=self.text1.yview)
        self.scrol1.pack(side='left', fill='y')
        self.text1.config(yscrollcommand=self.scrol1.set)
        self.load_settings()
        self.mainwindow = self.toplevel1
        self.mainwindow.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def ChooseLogosPath(self, event=None):
        rez = askopenfilename(filetypes=[("logos_sa", "logos_sa.exe")])
        if rez and os.path.exists(rez):
            self.LogosPathVar.set(rez)

    def ChooseYamlPath(self, event=None):
        rez = askopenfilename(filetypes=[("yaml file", "*.yaml")])
        if rez and os.path.exists(rez):
            self.YamlPathVar.set(rez)

    def ChooseWorkingDir(self, event=None):
        rez = askdirectory()
        if rez and os.path.exists(rez):
            self.WorkingDirVar.set(rez)

    def Calculate(self, event=None):
        lh = logos_helper(self.LogosPathVar.get())
        h = homogenizer(self.YamlPathVar.get(),
                        logos_helper=lh,
                        working_dir=self.WorkingDirVar.get(),
                        strain_value=float(self.StrainVar.get()))
        h.do_homogenization(int(self.MaxWorkersVar.get()))
        self.text1.insert('end', 'Матрица жесткости D:\n')
        self.text1.tag_add('D', 1.0, '1.end')
        self.text1.tag_config('D', font=('Times New Roman', 12, 'bold'))
        for row in h.D:
            s = ''
            for dd in row:
                s += '{:12.3e}'.format(dd)
            s += '\n'
            self.text1.insert('end', s)

        self.text1.insert('end', '\nМатрица податливости S:\n')
        self.text1.tag_add('S', 9.0, '9.end')
        self.text1.tag_config('S', font=('Times New Roman', 12, 'bold'))
        for row in h.S:
            s = ''
            for dd in row:
                s += '{:12.3e}'.format(dd)
            s += '\n'
            self.text1.insert('end', s)

        self.text1.insert('end', '\nИнженерные константы:\n')
        self.text1.tag_add('Eng', 17.0, '17.end')
        self.text1.tag_config('Eng', font=('Times New Roman', 12, 'bold'))
        self.text1.insert('end',
                          '  Ex={:10.3e}   Ey={:10.3e}   Ez={:10.3e}\n'.format(h.eng_moduli['Ex'],
                                                                               h.eng_moduli['Ey'],
                                                                               h.eng_moduli['Ez']))
        self.text1.insert('end',
                          ' Gxy={:10.3e}  Gxz={:10.3e}  Gyz={:10.3e}\n'.format(h.eng_moduli['Gxy'],
                                                                               h.eng_moduli['Gxz'],
                                                                               h.eng_moduli['Gyz']))
        self.text1.insert('end',
                          'nuxy={:10.3e} nuxz={:10.3e} nuyz={:10.3e}\n'.format(h.eng_moduli['nuxy'],
                                                                               h.eng_moduli['nuxz'],
                                                                               h.eng_moduli['nuyz']))

    def run(self):
        self.mainwindow.mainloop()

    def load_settings(self):
        if os.path.exists('config.json'):
            try:
                cfg = json.load(open('config.json', 'r'))
            except:
                return
            if ('LogosPath' in cfg) and os.path.exists(cfg['LogosPath']):
                self.LogosPathVar.set(cfg['LogosPath'])
            if ('YamlPath' in cfg) and os.path.exists(cfg['YamlPath']):
                self.YamlPathVar.set(cfg['YamlPath'])
            if ('WorkingDir' in cfg) and os.path.exists(cfg['WorkingDir']):
                self.WorkingDirVar.set(cfg['WorkingDir'])
            if ('StrainValue' in cfg):
                self.StrainVar.set(cfg['StrainValue'])
            if 'Parallel' in cfg:
                self.ParallelVar.set(cfg['Parallel'])
            if 'MaxWorkers' in cfg:
                self.MaxWorkersVar.set(cfg['MaxWorkers'])

    def on_window_close(self):
        cfg = {}
        cfg['LogosPath'] = self.LogosPathVar.get()
        cfg['YamlPath'] = self.YamlPathVar.get()
        cfg['WorkingDir'] = self.WorkingDirVar.get()
        cfg['StrainValue'] = self.StrainVar.get()
        cfg['Parallel'] = self.ParallelVar.get()
        cfg['MaxWorkers'] = self.MaxWorkersVar.get()
        json.dump(cfg, open('config.json', 'w'))
        self.mainwindow.quit()


if __name__ == '__main__':
    app = HomogenizerUI()
    app.run()
