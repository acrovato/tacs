from omflut import Builder
import openmdao.api as om
import numpy as np

class TacsMesh(om.IndepVarComp):
    """Initial structural mesh
    """
    def initialize(self):
        self.options.declare('fea_assembler', desc='pyTACS interface', recordable=False)
        self.options.declare('x_mask', desc='array to recover true degrees of freedom (node coordinates)')

    def setup(self):
        asb = self.options['fea_assembler']
        mask = self.options['x_mask']
        # Store all the node coordinates and add as output
        self.add_output('x_struct0', val=asb.getOrigNodes()[mask], desc='initial structural node coordinates')

class TacsSolver(om.ExplicitComponent):
    """OpenMDAO structural wrapper
    
    Attributes
    ----------
    n_modes : int
        number of eigenvalue frequencies / eigenvector modes
    abs : tacs.pyTACS object
        pyTACS interface
    pbl : tacs.problem.modal object
        Modal problem definition
    x_mask : np.array
        Array to recover true degrees of freedom (node coordinates)
    q_mask : np.array
        Array to recover true degrees of freedom (mode shapes)
    """
    def initialize(self):
        self.options.declare('fea_assembler', desc='pyTACS interface', recordable=False)
        self.options.declare('fea_problem', desc='modal problem', recordable=False)
        self.options.declare('write_solution', desc='flag to write solution')
        self.options.declare('x_mask', desc='array to recover true degrees of freedom (node coordinates)')
        self.options.declare('q_mask', desc='array to recover true degrees of freedom (mode shapes)')
        self.options.declare('n_nodes', desc='number of retained nodes')
        self.options.declare('retained_modes', desc='list of retained modes')

    def setup(self):
        self.asb = self.options['fea_assembler']
        self.pbl = self.options['fea_problem']
        self.x_mask = self.options['x_mask']
        self.q_mask = self.options['q_mask']
        self.retained_modes = self.options['retained_modes']
        self.n_modes = len(self.retained_modes)
        self.add_input('dv_struct', shape_by_conn=True, desc='structural design variables')
        self.add_input('x_struct0', shape_by_conn=True, desc='structural coordinates')
        self.add_output('q_struct', val=np.zeros((3 * self.options['n_nodes'], self.n_modes)), desc='modal displacements')
        self.add_output('M', val=np.identity(self.n_modes), desc='mass matrix')
        self.add_output('K', val=np.zeros((self.n_modes, self.n_modes)), desc='stiffness matrix')
        # Partials
        # self.declare_partials(of=['q_struct', 'M', 'K'], wrt=['x_struct0'], method='exact')
        self.declare_partials(of=['K'], wrt=['dv_struct'], method='exact')

    def compute(self, inputs, outputs):
        # Update nodes and design variables
        x_s = self.asb.getOrigNodes()
        x_s[self.x_mask] = inputs['x_struct0']
        self.pbl.setNodes(x_s)
        self.pbl.setDesignVars(inputs['dv_struct'])
        # Perform modal analysis
        self.pbl.solve()
        if self.options['write_solution']:
            self.pbl.writeSolution()
        # Mask and set modes
        for i in range(self.n_modes):
            _, q_s = self.pbl.getVariables(self.retained_modes[i])
            outputs['q_struct'][:, i] = self._extract_data(q_s[self.q_mask])
        # Get eigenvalues and set modal stiffness matrix
        funcs = {}
        self.pbl.evalFunctions(funcs)
        outputs['K'] = np.diag(list(funcs.values()))[np.ix_(self.retained_modes, self.retained_modes)]

    def compute_partials(self, inputs, partials):
        # Approximate derivative of stiffness matrix diagonal entries by derivative of eigenvalues
        func_sens = {}
        self.pbl.evalFunctionsSens(func_sens, evalVars='dv')
        for i in range(self.n_modes):
            d_lambda = func_sens[f'{self.pbl.name}_eigsm.{self.retained_modes[i]}']
            partials['K', 'dv_struct'][np.ravel_multi_index(([i], [i]), (self.n_modes, self.n_modes)), :] = d_lambda['struct']

    def _extract_data(self, q):
        """Extract modal displacements along z and rotations about x and y
        """
        # Constants
        vars_node = 6 # number of variables per node
        v_idx = [2, 3, 4] # indices of dz, rx and ry
        n_var = len(v_idx) # number of variables to extract
        n_nod = len(q) // vars_node # number of nodes
        # Extract
        q_red = np.zeros(n_nod * n_var)
        for i in range(n_nod):
            for j in range(n_var):
                q_red[i * n_var + j] = q[i * vars_node + v_idx[j]]
        return q_red

class TacsBuilder(Builder):
    """TACS builder for OpenMDAO

    Attributes
    ----------
    fea_assembler : tacs.pyTACS object
        pyTACS interface
    fea_problem : tacs.problem.modal object
        Modal problem definition
    mask : np.array
        Array to recover true degrees of freedom
    write_solution : bool
        Flag indicating whether to write solution
    """
    def __init__(self, mesh_file, num_modes, sigma=1., element_callback=None, exclude_modes=None, components_tag=None, pytacs_options=None, write_solution=True):
        """Instantiate and initialize TACS components

        Parameters
        ----------
        mesh_file : str or pyNastran.bdf.bdf.BDF
            The BDF file or a pyNastran BDF object to load
        num_modes : int
            Number of modes
        sigma : float
            Guess for the lowest eigenvalue (default: 1.0)
        element_callback : collections.abc.Callable, optional
            User-defined callback function for setting up TACS elements and element DVs (default: None)
        exclude_modes : list[int]
            List of indices of modes to exclude from analysis (default: None)
        components_tag : list[str]
            List of tag names for which nodes will be included (default: None)
        pytacs_options : dict, optional
            Options dictionary passed to pyTACS assembler (default: None)
        write_solution : bool, optional
            Flag to determine whether to write out TACS solutions to f5 file each design iteration (default: True)
        """
        from tacs.pytacs import pyTACS
        from mpi4py import MPI
        # Initialize assembler
        self.fea_assembler = pyTACS(mesh_file, options=pytacs_options, comm=MPI.COMM_WORLD)
        self.fea_assembler.initialize(element_callback)
        # Set up the problem
        self.fea_problem = self.fea_assembler.createModalProblem('modal', sigma, num_modes)
        self.retained_modes = list(range(num_modes))
        if exclude_modes is not None: self.retained_modes = list(np.delete(self.retained_modes, exclude_modes))
        # Select nodes from components
        nodes_id = self._select_nodes(components_tag)
        self.n_nodes = len(nodes_id)
        # Create masks
        self.x_mask = self._create_mask(nodes_id, 3)
        self.q_mask = self._create_mask(nodes_id, 6)
        # Save other parameters
        self.write_solution = write_solution

    def get_mesh(self):
        """Return OpenMDAO component to get the initial mesh coordinates
        """
        return TacsMesh(fea_assembler=self.fea_assembler, x_mask=self.x_mask)

    def get_solver(self, scenario_name=''):
        """Return OpenMDAO component containing the solver
        """
        return TacsSolver(fea_assembler=self.fea_assembler, fea_problem=self.fea_problem, write_solution=self.write_solution, x_mask=self.x_mask, q_mask=self.q_mask, n_nodes=self.n_nodes, retained_modes=self.retained_modes)

    def get_number_of_nodes(self):
        """Return the number of (true) nodes
        """
        return self.n_nodes

    def get_number_of_dv(self):
        """Return the number of degrees of freedom
        """
        return self.fea_assembler.getTotalNumDesignVars()

    def _select_nodes(self, components):
        """Select nodes from given components
        
        Parameters
        ----------
        components : list[str]
            List of tag names for which nodes will be included
        """
        if components is None:
            nodes_id = np.arange(self.fea_assembler.getNumOwnedNodes())
        else:
            nodes_id = self.fea_assembler.getLocalNodeIDsForComps(self.fea_assembler.selectCompIDs(components))
        mults_id = self.fea_assembler.getLocalMultiplierNodeIDs()
        return list(set(nodes_id) - set(mults_id))

    def _create_mask(self, nodes_id, vars_node):
        """Create a mask to recover array indices associated to retained true degrees of freedom
        
        Parameters
        ----------
        nodes_id : list[int]
            index of retained nodes
        vars_node : int
            Number of variables per node
        """
        mask = np.full((self.fea_assembler.getNumOwnedNodes(), vars_node), False)
        mask[nodes_id, :] = True
        return mask.flatten()
