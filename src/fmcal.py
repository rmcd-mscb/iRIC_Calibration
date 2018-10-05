import configparser
import numpy as np
import os
import h5py
import vtk
import subprocess
from itertools import count
from shutil import copyfile
import sys

class  fm_cal:
    def __init__(self, file=''):
        if file != '' :
            config = configparser.ConfigParser()
            config.read(file)
            self.meas_wse = np.genfromtxt(config.get('Params', 'meas_WSE_File'),
                                          delimiter=',', skip_header=1)
            self.nummeas = self.meas_wse.shape[0]
            self.cdmin = config.getfloat('Params', 'cdmin')
            self.cdmax = config.getfloat('Params', 'cdmax')
            self.cdinc = config.getfloat('Params', 'cdinc')
            self.cdtype = config.getint('Params', 'cdtype')
            self.cds = np.arange(self.cdmin, self.cdmax, self.cdinc)
            self.numcds = self.cds.shape[0]
            self.xoffset = config.getfloat('Params', 'xoffset')
            self.yoffset = config.getfloat('Params', 'yoffset')
            self.Q = config.getfloat('Params', 'Q')
            self.H_DS = config.getfloat('Params', 'H_DS')
            self.H_US = config.getfloat('Params', 'H_US')
            self.iniType = config.getint('Params', 'iniType')
            self.OneDCD = config.getfloat('Params', 'OneDCD')

            self.working_dir = config.get('Params', 'working_dir')
            self.lib_path = config.get('Params', 'lib_path')
            self.solver_path = config.get('Params', 'solver_path')
            self.base_file = config.get('Params', 'base_file')
        # sol_file = D:\USACE\MeanderCalibration\2011\Meander_Base_2011_5m_bridge - Copy\Case1_Solution1.cgn
        # new_sol_file = D:\USACE\MeanderCalibration\2011\m20110718_457pt3cms\Case1_Solution1.cgn
            self.rmse_file = config.get('Params', 'rmse_file')
            self.meas_vs_sim_file = config.get('Params', 'meas_vs_sim_file')

            self.stat_init = False
            self.stat_numcalpts = count()
            self.g = self.gen_filenames("FM_Calib_Flow_", ".cgns")

            self.rmse_data = np.zeros(self.numcds)
            self.cd_val = np.zeros(self.numcds)
            self.meas_and_sim_wse = np.zeros(shape=(self.nummeas, self.numcds + 1))

    def create_ini_file(self, file=''):
        f = open(file, "w+")
        f.write("[Params]\n")
        f.write("#enter full or relative path to measured water-surface elevation file (csv)\n")
        f.write("meas_WSE_File = path\\to\\file\n")
        f.write("#enter the min, max Cd (drag coefficient) and increment\n")
        f.write("cdmin = 0.004\n")
        f.write("cdmax = 0.006\n")
        f.write("cdinc = 0.0001\n")
        f.write("#cdtype == 0 (constant cd) cdtype == 1 (variable cd)\n")
        f.write("#  Variable cd allows one region where cd is fixed and one that is adjusted\n")
        f.write("#  Copy roughness polygon of region to be adjusted into sand-depth and make its value == 1\n")
        f.write("cdtype = 0\n")
        f.write("xoffset = 0.0\n")
        f.write("yoffset = 0.0\n")
        f.write("Q = 100.0\n")
        f.write("H_DS = 447.1\n")
        f.write("H_US = 449\n")
        f.write("iniType = 2\n")
        f.write("OneDCD = 0.015\n")
        f.write("working_dir =..\\test\\cal_const_cd\n")
        f.write("#lib_path =;C:\\Users\\rmcd\\iRIC_dev\\guis\\prepost\n")
        f.write("solver_path =;C:\\Users\\rmcd\\iRIC_dev\\solvers\\Fastmech_v1\n")
        f.write("base_file =..\\test\\test_const_cd\\Case1.cgn\n")
        f.write("rmse_file = test_rmse.csv\n")
        f.write("meas_vs_sim_file = test_m_vs_s.csv\n")
        f.close()

    def initialize(self):

        """
        This funtion takes no arguements.  It adds the fastmech
        solver specified in .ini file and temporarily puts it in your path
        :return:
        """
        os.chdir(self.working_dir)
        self.add_fastmech_solver_to_path()
        self.add_fastmech_libs_to_path()
        self.stat_init = True



    def update(self):
        if not self.stat_init:
            print('call initialize() before run()')
        else:
            index = next(self.stat_numcalpts)
            tcd = self.cds[index]
            hdf5_file_name = next(self.g)
            copyfile(self.base_file, hdf5_file_name)
            self.fastmech_change_cd(hdf5_file_name, tcd)
            self.fastmech_BCs(hdf5_file_name)
            for path in self.execute(["Fastmech.exe", hdf5_file_name]):
                print(path, end="")

            SGrid = vtk.vtkStructuredGrid()
            self.create_vtk_structured_grid(SGrid, hdf5_file_name)
            cellLocator2D = vtk.vtkCellLocator()
            cellLocator2D.SetDataSet(SGrid)
            # cellLocator2D.SetNumberOfCellsPerBucket(10);
            cellLocator2D.BuildLocator()

            WSE_2D = SGrid.GetPointData().GetScalars('WSE')
            IBC_2D = SGrid.GetPointData().GetScalars('IBC')
            Velocity_2D = SGrid.GetPointData().GetScalars('Velocity')
            simwse = np.zeros(self.meas_wse.shape[0])
            measwse = np.zeros(self.meas_wse.shape[0])
            for counter, line in enumerate(self.meas_wse):
                point2D = [line[0] - self.xoffset, line[1] - self.yoffset, 0.0]
                pt1 = [line[0] - self.xoffset, line[1] - self.yoffset, 10.0]
                pt2 = [line[0] - self.xoffset, line[1] - self.yoffset, -10]
                idlist1 = vtk.vtkIdList()
                cellLocator2D.FindCellsAlongLine(pt1, pt2, 0.0, idlist1)
                cellid = idlist1.GetId(0)
                # cellid = cellLocator2D.FindCell(point2D)
                # print (isCellWet(SGrid, point2D, cellid, IBC_2D))
                tmpwse = self.getCellValue(SGrid, point2D, cellid, WSE_2D)
                if tcd == self.cdmin:
                    self.meas_and_sim_wse[counter, 0] = line[2]
                #     print counter
                simwse[counter] = tmpwse
                measwse[counter] = line[2]
                print(cellid, line[2], tmpwse)
            self.meas_and_sim_wse[:, index + 1] = simwse
            self.rmse_data[index] = self.rmse(simwse, measwse)
            self.cd_val[index] = tcd
            print(self.rmse_data[index])
            print(self.cd_val)
            print(self.rmse_data)
            trmse = np.column_stack((self.cd_val.flatten(), self.rmse_data.flatten()))
            print(trmse)
            np.savetxt(self.rmse_file, trmse, delimiter=',')
            np.savetxt(self.meas_vs_sim_file, self.meas_and_sim_wse, delimiter=',')
            return trmse
            # next(self.stat_numcalpts)

    def add_fastmech_solver_to_path(self):

        print(os.environ['PATH'])
        os.environ['PATH'] += self.solver_path
        print("\n")
        print('new path')
        print(os.environ['PATH'])

    def add_fastmech_libs_to_path(self):
        print(os.environ['PATH'])
        os.environ['PATH'] += self.lib_path
        print("\n")
        print('new path')
        print(os.environ['PATH'])

    def gen_filenames(self, prefix, suffix, places=3):
        """Generate sequential filenames with the format <prefix><index><suffix>

           The index field is padded with leading zeroes to the specified number of places

           http://stackoverflow.com/questions/5068461/how-do-you-increment-file-name-in-python
        """
        pattern = "{}{{:0{}d}}{}".format(prefix, places, suffix)
        for i in count(1):
            yield pattern.format(i)

    def fastmech_change_cd(self, hdf_file, newCd):
        # hdf5_file_name = r'F:\Kootenai Project\USACE\Braided\Case11_tmp.cgn'
        # r+ adds read/write permisions to file
        file = h5py.File(hdf_file, 'r+')
        group = file['/iRIC/CalculationConditions/FM_HydAttCD/Value']
        dset = group[u' data']
        # print dset[0]
        dset[0] = newCd
        # print dset[0]
        file.close()

    def fastmech_BCs(self, hdf_file):
        file = h5py.File(hdf_file, 'r+')
        group = file['/iRIC/CalculationConditions/FM_HydAttQ/Value']
        dset = group[u' data']
        dset[0] = self.Q
        group2 = file['/iRIC/CalculationConditions/FM_HydAttWS2/Value']
        dset2 = group2[u' data']
        dset2[0] = self.H_US
        group3 = file['/iRIC/CalculationConditions/FM_HydAttWS/Value']
        dset3 = group3[u' data']
        dset3[0] = self.H_DS
        group4 = file['/iRIC/CalculationConditions/FM_HydAttWSType/Value']
        dset4 = group4[u' data']
        dset4[0] = self.iniType
        # group5 = file['/iRIC/CalculationConditions/FM_HydAttWS1DStage/Value']
        # dset5 = group5[u' data']
        # dset5[0] = OneDStage
        # group6 = file['/iRIC/CalculationConditions/FM_HydAttWS1DDisch/Value']
        # dset6 = group6[u' data']
        # dset6[0] = OneDQ
        group7 = file['/iRIC/CalculationConditions/FM_HydAttWS1DCD/Value']
        dset7 = group7[u' data']
        dset7[0] = self.OneDCD
        file.close()

    def execute(self, cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    def create_vtk_structured_grid(self, sgrid, hdf5_file_name):
        # type: (object) -> object
        file = h5py.File(hdf5_file_name, 'r')
        xcoord_grp = file['/iRIC/iRICZone/GridCoordinates/CoordinateX']
        print(xcoord_grp.keys())
        ycoord_grp = file['/iRIC/iRICZone/GridCoordinates/CoordinateY']
        print(ycoord_grp.keys())
        wse_grp = file['iRIC/iRICZone/FlowSolution1/WaterSurfaceElevation']
        print(wse_grp.keys())
        topo_grp = file['iRIC/iRICZone/FlowSolution1/Elevation']
        print(topo_grp.keys())
        ibc_grp = file['iRIC/iRICZone/FlowSolution1/IBC']
        velx_grp = file['iRIC/iRICZone/FlowSolution1/VelocityX']
        vely_grp = file['iRIC/iRICZone/FlowSolution1/VelocityY']

        xcoord_data = xcoord_grp[u' data']
        ycoord_data = ycoord_grp[u' data']
        wse_data = wse_grp[u' data']
        topo_data = topo_grp[u' data']
        ibc_data = ibc_grp[u' data']
        velx_data = velx_grp[u' data']
        vely_data = vely_grp[u' data']


        # SGrid = vtk.vtkStructuredGrid()
        ny, nx, = xcoord_data.shape
        print(ny, nx)
        sgrid.SetDimensions(nx, ny, 1)
        points = vtk.vtkPoints()
        wseVal = vtk.vtkFloatArray()
        wseVal.SetNumberOfComponents(1)
        ibcVal = vtk.vtkIntArray()
        ibcVal.SetNumberOfComponents(1)
        velVal = vtk.vtkFloatArray()
        velVal.SetNumberOfComponents(1)
        for j in range(ny):
            for i in range(nx):
                points.InsertNextPoint(xcoord_data[j, i] - self.xoffset,
                                       ycoord_data[j, i] - self.yoffset, 0.0)
                wseVal.InsertNextValue(wse_data[j, i])
                ibcVal.InsertNextValue(ibc_data[j, i])
                velVal.InsertNextValue(np.sqrt(np.power(velx_data[j, i],2)
                                               + np.power(vely_data[j,i],2)))
            sgrid.SetPoints(points)

            sgrid.GetPointData().AddArray(wseVal)
            sgrid.GetPointData().AddArray(ibcVal)
            sgrid.GetPointData().AddArray(velVal)
        wseVal.SetName("WSE")
        ibcVal.SetName("IBC")
        velVal.SetName("Velocity")

    def getCellValue(self, vtkSGrid2D, newPoint2D, cellID, valarray):
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        clspoint = [0., 0., 0.]
        tmpid = vtk.mutable(0)
        vtkid2 = vtk.mutable(0)
        vtkcell2D = vtk.vtkQuad()
        vtkcell2D = vtkSGrid2D.GetCell(cellID)
        tmpres = vtkcell2D.EvaluatePosition(newPoint2D, clspoint, tmpid, pcoords, vtkid2, weights)
        print(newPoint2D, clspoint, tmpid, pcoords, vtkid2, weights)
        idlist1 = vtk.vtkIdList()
        numpts = vtkcell2D.GetNumberOfPoints()
        idlist1 = vtkcell2D.GetPointIds()
        tmpVal = 0.0
        for x in range(0, numpts):
            tmpVal = tmpVal + weights[x] * valarray.GetTuple(idlist1.GetId(x))[0]
        return tmpVal

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def fastmech_change_var_cd(self, hdf_file, newCd_0, newCd_1):
        # hdf5_file_name = r'F:\Kootenai Project\USACE\Braided\Case11_tmp.cgn'
        # r+ adds read/write permisions to file
        file = h5py.File(hdf_file, 'r+')
        group = file['/iRIC/iRICZone/GridConditions/sanddepth/Value']
        dset = group[u' data']
        group2 = file['/iRIC/iRICZone/GridConditions/roughness/Value']
        dset2 = group2[u' data']
        for index, val in enumerate(dset):
            if val == 1.0:
                dset2[index] = newCd_0
            # else:
            # dset2[index] = newCd_1 #keep values in original project, change only values with 0
        # print dset[0]
        # print dset[0]
        file.close()
