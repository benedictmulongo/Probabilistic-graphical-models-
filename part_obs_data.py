import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


def print_tables(px, py, pz):
    """ Prints probability tables in a nice way. 
    Args:
        px, py, pz (np.array): Parameters, true or learnt.
    """
    assert px.shape == (2,), f"px = {px} should have shape (2,)"
    assert py.shape == (2,), f"py = {py} should have shape (2,)"
    assert pz.shape == (2, 2, 2), f"pz = {pz} should have shape (2, 2, 2)"
    print(f"p(x) = {px}")
    print(f"p(y) = {py}")
    print(f"p(z|x=0, y=0) = {pz[0, 0]}")
    print(f"p(z|x=0, y=1) = {pz[0, 1]}")
    print(f"p(z|x=1, y=0) = {pz[1, 0]}")
    print(f"p(z|x=1, y=1) = {pz[1, 1]}")


def print_marginals(px, py, pz):
    """ Prints marginal probabilities of z given x or y, i.e. one variable is summed out.
    Args:
        px, py, pz (np.array): Parameters, true or learnt.
    """
    assert px.shape == (2,), f"px = {px} should have shape (2,)"
    assert py.shape == (2,), f"py = {py} should have shape (2,)"
    assert pz.shape == (2, 2, 2), f"pz = {pz} should have shape (2, 2, 2)"
    print(f"p(z|x=0) = {pz[0, 0] * py[0] + pz[0, 1] * py[1]}")
    print(f"p(z|x=1) = {pz[1, 0] * py[0] + pz[1, 1] * py[1]}")
    print(f"p(z|y=0) = {pz[0, 0] * px[0] + pz[1, 0] * px[1]}")
    print(f"p(z|y=1) = {pz[0, 1] * px[0] + pz[1, 1] * px[1]}")

def initialize_parameters(random=False):
    """ Initializes parameters for the EM algorithm
    Args:
        random (bool): If True, the parameters are set to random values (in range [0, 1] that sum to 1).
            If False, all probabilities are 0.5 (binary variables).
    Returns:
        qx, qy, qx (np.array): Initial parameters.
    """
    if random:
        qx = np.random.rand(2)
        qy = np.random.rand(2)
        qz = np.random.rand(2, 2, 2) 
    else:
        qx = np.ones(2)
        qy = np.ones(2)
        qz = np.ones((2,2,2))
    qx = qx / np.sum(qx)
    qy = qy / np.sum(qy)
    qz = qz / np.sum(qz, axis=2, keepdims=1)
    return qx, qy, qz
    
def generate_data(px, py, pz, n, *, partially_observed=False, never_coobserved=False):
    """ Generates data given table-CPDs of a V-stucture X -> Z <- Y
    It can generate complete or partially observed data
    Args:
        px, py, px (np.array): Parameters to generate data with.
        n (int): Number of data points.
        partially_observed (bool): If True, half of x and y will be missing (set to None)
        never_coobserved (bool): If True, y is missing if and only if x is observed, 
            so that no data points contains both x and y. 
            If False, y is missing independently of whether x is missing.
            Has no effect if partially_observed is False.
    Return:
        x, y, z (np.array): Generated data where a None in x or y is interpreted as missing data. 
    """
    assert px.shape == (2,), f"px = {px} should have shape (2,)"
    assert py.shape == (2,), f"py = {py} should have shape (2,)"
    assert pz.shape == (2, 2, 2), f"pz = {pz} should have shape (2, 2, 2)"
    x = np.argmax(np.random.multinomial(1, px, n), axis=1)
    y = np.argmax(np.random.multinomial(1, py, n), axis=1)
    z = np.argmax([np.random.multinomial(1, p) for p in pz[x, y]], axis=1)
    if partially_observed:
        x = x.astype(object)
        y = y.astype(object)
        x[np.unique(np.random.choice(n, int(n/2)))] = None
        if never_coobserved:
            y[np.not_equal(x, None)] = None
        else:
            y[np.unique(np.random.choice(n, int(n/2)))] = None
    return x, y, z
    

def e_step(Ox, Oy, Oz, xs, ys, zs):
    n = len(xs)
    print("Length = ",n)
    Mx = np.zeros(2)
    My = np.zeros(2)
    Mz = np.zeros((2, 2, 2))  # Remember index order: Mz[x, y, z]
    cpd = {}
    i = 1
    for x, y, z in zip(xs, ys, zs):
        """
        To do:  p(X|x, y, z) and add to Mx
                p(Y|x, y, z) and add to My
                p(Z, X, Y|x, y, z) and add to Mz.
            Remember to normalize p(.), i.e. each should sum to 1. 
                For example, if x, y and z are not None, we should have p(X=x) = 1, p(Y=y) = 1, p(Z=z, X=x, Y=y) = 1. 
        Naive solution (~45 lines of code): 
            x and y are None? ...
            x is None: ... (NB: p(X|Y=y, Z=z) = p(X, Y=y, Z=z) / p(Y=y, Z=z) != p(X))
            y is None: ...
            x and y are known: p(...) = 1
        Pythonic solution (<10 lines of code):
            Q = np.zeros((2, 2)) # Q(x, y) = p(Z=z, X=x, Y=y)
            # Q <- p(...)
            Mx += ...
            My += ...
            Mz[:, :, z] += ...
        """
        print("Index = ", i)
        print("Ox = ", Ox)
        print("Oy = ", Oy)
        print("Oz = ", Oz)
        prob_x = Ox[x]
        prob_y = Oy[y]
        prob_z = Oz[z][x][y]
        
        prob_xyz = prob_x * prob_y * prob_z
        
        c = str(x) + str(y) + str(z) 
        
        if c in cpd.keys() :
            cpd[c] +=  1/n
        else :
            cpd[c] =  1/n
            
        Mx[x] += 1
        My[y] += 1
        Mz[z][x][y] += 1
        # Mz[z][y][x] += 1
        
        
        # Mz[x] += 1
        # Mz[:, y] += 1
        
        # 
        # Mx[x] += Ox[x]
        # My[y] += Oy[y]
        # Mz[z][x][y] += Oz[z][x][y]
        # 
        
        
        # assert np.isclose(np.sum(Mz[0]), Mx[0]), f"Mz[0] = {Mz[0]} should sum to Mx[0] = {Mx[0]}"
        # assert np.isclose(np.sum(Mz[1]), Mx[1]), f"Mz[1] = {Mz[1]} should sum to Mx[1] = {Mx[1]}"
        # assert np.isclose(np.sum(Mz[:, 0]), My[0]), f"Mz[:, 0] = {Mz[:, 0]} should sum to My[0] = {My[0]}"
        # assert np.isclose(np.sum(Mz[:, 1]), My[1]), f"Mz[:, 1] = {Mz[:, 1]} should sum to My[1] = {My[1]}"
        #         
            
        i += 1
    
    print("CPD")
    print(cpd)
    
    
            
    return Mx, My, Mz


def m_step(Mx, My, Mz):
    """
    Convert from sufficient statistics to parameters. What elemets should sum to one?
    """
    
    qx = np.zeros(2)
    qy = np.zeros(2)
    qz = np.zeros((2, 2, 2))
    
    qx = Mx / np.sum(Mx)
    qy = My / np.sum(My)
    
    E =  0.0
    
    # qz[0,0] = Mz[0,0]/(np.sum(Mz[0,0]) + E)
    # qz[0,1] = Mz[0,1]/(np.sum(Mz[0,1]) + E)
    # qz[1,0] = Mz[1,0]/(np.sum(Mz[1,0]) + E)
    # qz[1,1] = Mz[1,1]/(np.sum(Mz[1,1]) + E)
    
    # 
    # tot = Mz[0][0][0] + Mz[0][0][1] + Mz[0][1][0] + Mz[0][1][1]  + Mz[1][0][0]  + Mz[1][0][1]  + Mz[1][1][0] + Mz[1][1][1] 
    # 
    # qz[0][0][0] = Mz[0][0][0]/(tot + E)
    # qz[0][0][1] = Mz[0][0][1]/(tot + E)
    # 
    # qz[0][1][0] = Mz[0][1][0]/(tot + E)
    # qz[0][1][1] = Mz[0][1][1]/(tot + E)
    # 
    # qz[1][0][0] = Mz[1][0][0]/(tot + E)
    # qz[1][0][1] = Mz[1][0][1]/(tot + E)
    # 
    # qz[1][1][0] = Mz[1][1][0]/(tot + E)
    # qz[1][1][1] = Mz[1][1][1]/(tot + E)
    
    qz[0][0][0] = Mz[0][0][0]/(Mz[1][0][0] + Mz[0][0][0] + E)
    qz[0][0][1] = Mz[1][0][0]/(Mz[1][0][0] + Mz[0][0][0] + E)
    
    qz[0][1][0] = Mz[0][0][1]/(Mz[0][0][1] + Mz[1][0][1] + E)
    qz[0][1][1] = Mz[1][0][1]/(Mz[0][0][1] + Mz[1][0][1] + E)
    
    qz[1][0][0] = Mz[0][1][0]/(Mz[0][1][0] + Mz[1][1][0] + E)
    qz[1][0][1] = Mz[1][1][0]/(Mz[0][1][0] + Mz[1][1][0] + E)
    
    qz[1][1][0] = Mz[0][1][1]/(Mz[1][1][1] + Mz[0][1][1] + E)
    qz[1][1][1] = Mz[1][1][1]/(Mz[1][1][1] + Mz[0][1][1] + E)
    
    
    return qx, qy, qz

    
    
def expectation_maximization(x, y, z, num_iter):

    n = len(x)
    qx, qy, qz = initialize_parameters()
    for i in range(num_iter):
        Mx, My, Mz = e_step(qx, qy, qz, x, y, z)
        
        print("Mx = ", Mx)
        print("My = ", My)
        print("Mz = ", Mz)
        qx, qy, qz = m_step(Mx, My, Mz)

    return qx, qy, qz
    

np.random.seed(1337)
px = np.array([0.6, 0.4])
py = np.array([0.3, 0.7])
pz = np.array([[[0.2, 0.8], [0.7, 0.3]], [[0.9, 0.1], [0.1, 0.9]]])  # p(z|x, y) = pz[x, y, z]
n_data = 500

print()
x, y, z = generate_data(px, py, pz, n_data)
v = [x.tolist(), y.tolist(), z.tolist()]
v = np.array(v).T
print([x,y,z])
print()
print(v)

n_iter = 10
qx, qy, qz = expectation_maximization(x, y, z, n_iter)




# ans = ve.query(variables = ['delay'], evidence = {'age': 0})
# print()
# print(ans['delay'])
# 
# 
# graph = BayesianModel([('x', 'z'),('y', 'z')])
# cpd_x = TabularCPD('x', 2, [[0.6, 0.4]])
# cpd_y = TabularCPD('y', 2, [[0.5, 0.5]])
# 
# cpd_z = TabularCPD(
# 'z', 2,
# [[0.6, 0.8, 0.1, 0.6], 
# [0.4, 0.2, 0.9, 0.4]],
# ['x', 'y'], [2, 2])
# graph.add_cpds(cpd_x, cpd_y, cpd_z)
# 
# inference = VariableElimination(graph)
# d = inference.query(variables=['z'])
# print(d['z'])

# Mx =  [284. 216.]
# My =  [147. 353.]
# Mz =  [[[ 21. 125.]
#   [ 53.  16.]]
# 
#  [[ 66.  72.]
#   [  7. 140.]]]