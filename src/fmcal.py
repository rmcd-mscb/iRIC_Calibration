import configparser
import numpy as np
import os
import h5py
import vtk
import subprocess
import pandas as pd
from itertools import count
from shutil import copyfile


class fm_cal:
    def __init__(self, file=''):
        if file != '':
            config = configparser.ConfigParser()
            config.read(file)
            self.meas_wse = np.genfromtxt(config.get('Params', 'meas_WSE_File'),
                                          delimiter=',', skip_header=1)
            self.nummeas = self.meas_wse.shape[0]
            self.cdtype = 1  # config.getint('Params', 'cdtype')
            self.mcdmin = {}
            for key in config['mcdmin']:
                self.mcdmin[key] = config.getfloat('mcdmin', key)
            self.mcdmax = {}
            for key in config['mcdmax']:
                self.mcdmax[key] = config.getfloat('mcdmax', key)
            self.mcdinc = {}
            for key in config['mcdinc']:
                self.mcdinc[key] = config.getfloat('mcdinc', key)
            self.mnumcds = len(self.mcdinc)

            rng = range(0, int(self.mnumcds / 2) + 1)
            self.dfcols = ['index'] + ['cd' + str(i) for i in rng] + ['Discharge']
            self.resdf = pd.DataFrame(columns=self.dfcols)

            self.xoffset = 0  # config.getfloat('Params', 'xoffset')
            self.yoffset = 0  # config.getfloat('Params', 'yoffset')
            self.Q = config.getfloat('Params', 'Q')
            self.H_DS = config.getfloat('Params', 'H_DS')
            self.H_US = config.getfloat('Params', 'H_US')
            self.iniType = config.getint('Params', 'iniType')
            self.OneDCD = config.getfloat('Params', 'OneDCD')

            self.working_dir = config.get('Params', 'working_dir')
            self.solver_path = config.get('Params', 'solver_path')
            self.base_file = config.get('Params', 'base_file')
            self.rmse_file = config.get('Params', 'rmse_file')

            self.stat_init = False
            self.stat_numcalpts = count()
            self.g = self.gen_filenames("FM_Calib_Flow_", ".cgns")

    def create_ini_file(self, file=''):
        if file == '': file = 'config.ini'
        cfgfile = open(file, 'w')
        Config = configparser.ConfigParser(allow_no_value=True)
        Config.add_section('mcdmin')
        Config.set('mcdmin', '# One or more pairs of Id = min. roughness value.', None)
        Config.set('mcdmin', '0', '0.004')
        Config.add_section('mcdmax')
        Config.set('mcdmax', '# One or more pairs of Id = max. roughness value.', None)
        Config.set('mcdmax', '0', '0.010')
        Config.add_section('mcdinc')
        Config.set('mcdinc', '# One or more pairs of Id = increment values.', None)
        Config.set('mcdinc', '0', '0.00025')
        Config.add_section('Params')
        Config.set('Params', '# Relative or complete path\\file.csv to measured water-surface elevation file.', None)
        Config.set('Params', 'meas_WSE_File', r'..\test\GR_wse.csv')
        #        Config.set('Params','cdtype', '1')
        #        Config.set('Params', 'xoffset', '0')
        #        Config.set('Params', 'yoffset', '0')
        Config.set('Params', '# Discharge Value', None)
        Config.set('Params', 'Q', '241.0')
        Config.set('Params', '# Downstream Stage Value', None)
        Config.set('Params', 'H_DS', '447.1')
        Config.set('Params', '# Set initial water-surface elevation boundary condition', None)
        Config.set('Params', '# iniType == 1 use upstream elevation (H_US)', None)
        Config.set('Params', '# iniType == 2 use step-backwater (OneDCD = 1-D Cd for step-backwater)', None)
        Config.set('Params', 'iniType', '2')
        Config.set('Params', 'H_US', '449')
        Config.set('Params', 'OneDCD', '.015')
        Config.set('Params', '# Complete path to FaSTMECH solver', None)
        Config.set('Params', 'solver_path', r';C:\Users\rmcd\iRICt\solvers\fastmech')
        Config.set('Params', '# Relative (to python code) or complete path to working directory where files/results are stored', None)
        Config.set('Params', 'working_dir', r'..\test\cal_const_cd')
        Config.set('Params', '# Relative (to working directory) or complete path to base iRIC project (folder)', None)
        Config.set('Params', 'base_file', r'..\test_const_cd\Case1.cgn')
        Config.set('Params', '# File used to output results - save to working directory')
        Config.set('Params', 'rmse_file', 'test_rmse.csv')
        Config.write(cfgfile)
        cfgfile.close()

    def initialize(self):

        """
        This funtion takes no arguements.  It adds the fastmech
        solver specified in .ini file and temporarily puts it in your path
        :return:
        """
        os.chdir(self.working_dir)
        self.add_fastmech_solver_to_path()
        # self.add_fastmech_libs_to_path()
        self.stat_init = True

    def update_var(self, cnt, tcd, q=0):
        if not self.stat_init:
            print('Error: call initialize() before update()')
        else:
            # index = next(self.stat_numcalpts)
            # tcd = self.cds[index]
            hdf5_file_name = next(self.g)
            copyfile(self.base_file, hdf5_file_name)
            self.fastmech_change_var_cd2(hdf5_file_name, tcd)
            if q != 0:
                self.Q = q
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
                tmpwse = self.getCellValue(SGrid, point2D, cellid, WSE_2D)
                simwse[counter] = tmpwse
                measwse[counter] = line[2]
                print(cellid, line[2], tmpwse)
            self.resdf.loc[cnt, self.dfcols[0]] = cnt
            for key in tcd:
                self.resdf.loc[cnt, self.dfcols[int(key + 1)]] = tcd[key]
            self.resdf.loc[cnt, 'rmse'] = self.rmse(simwse, measwse)
            self.resdf.loc[cnt, 'Discharge'] = self.Q

            # trmse = np.column_stack((self.cd0_var_vals.flatten(), self.cd1_var_vals.flatten(), self.rmse_var_data.flatten()))
            # print(trmse)
            #
            # np.savetxt(self.rmse_file, trmse, delimiter=',')
            self.resdf.to_csv(self.rmse_file)
            # np.savetxt(self.meas_vs_sim_file, self.meas_and_sim_wse_var, delimiter=',')
            return self.resdf

    def update_const(self, index, tcd):
        if not self.stat_init:
            print('Error: call initialize() before update()')
        else:
            # index = next(self.stat_numcalpts)
            # tcd = self.cds[index]
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
                # print(cellid, line[2], tmpwse)
            self.meas_and_sim_wse[:, index + 1] = simwse
            self.rmse_data[index] = self.rmse(simwse, measwse)
            self.cd_val[index] = tcd
            # print(self.rmse_data[index])
            # print(self.cd_val)
            # print(self.rmse_data)
            trmse = np.column_stack((self.cd_val.flatten(), self.rmse_data.flatten()))
            # print(trmse)
            np.savetxt(self.rmse_file, trmse, delimiter=',')
            np.savetxt(self.meas_vs_sim_file, self.meas_and_sim_wse, delimiter=',')
            return trmse
            # next(self.stat_numcalpts)

    def add_fastmech_solver_to_path(self):

        # print(os.environ['PATH'])
        os.environ['PATH'] += self.solver_path
        # print("\n")
        # print('new path')
        # print(os.environ['PATH'])

    def add_fastmech_libs_to_path(self):
        # print(os.environ['PATH'])
        os.environ['PATH'] += self.lib_path
        # print("\n")
        # print('new path')
        # print(os.environ['PATH'])

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
        group5 = file['/iRIC/CalculationConditions/FM_HydAttCDType/Value']
        dset5 = group5[u' data']
        dset5[0] = self.cdtype
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
        # print(xcoord_grp.keys())
        ycoord_grp = file['/iRIC/iRICZone/GridCoordinates/CoordinateY']
        # print(ycoord_grp.keys())
        wse_grp = file['iRIC/iRICZone/FlowSolution1/WaterSurfaceElevation']
        # print(wse_grp.keys())
        topo_grp = file['iRIC/iRICZone/FlowSolution1/Elevation']
        # print(topo_grp.keys())
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
        # print(ny, nx)
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
                velVal.InsertNextValue(np.sqrt(np.power(velx_data[j, i], 2)
                                               + np.power(vely_data[j, i], 2)))
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
        # print(newPoint2D, clspoint, tmpid, pcoords, vtkid2, weights)
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
                dset2[index] = newCd_1
            else:
                dset2[index] = newCd_0
        # print dset[0]
        # print dset[0]
        file.close()

    def fastmech_change_var_cd2(self, hdf_file, tcd):
        # hdf5_file_name = r'F:\Kootenai Project\USACE\Braided\Case11_tmp.cgn'
        # r+ adds read/write permisions to file
        file = h5py.File(hdf_file, 'r+')
        # group = file['/iRIC/iRICZone/GridConditions/sanddepth/Value']
        # dset = group[u' data']
        group2 = file['/iRIC/iRICZone/GridConditions/roughness/Value']
        dset2 = group2[u' data']
        for index, val in enumerate(dset2):
            if val in tcd:
                dset2[index] = tcd[val]
            else:
                print('invalid key %d', val)
            # if val == 1.0:
            #     dset2[index] = newCd_1
            # else:
            #     dset2[index] = newCd_0
        # print dset[0]
        # print dset[0]
