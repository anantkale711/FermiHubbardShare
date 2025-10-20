def check_inside(x, y, lines):
    for line in lines:
        if line(x,y) < 0:
            return False
    return True

def generate_lattice_sites(e1, e2, lines):
    sites = []
    sites_dict = dict()
    count = 0
    for i in range(-10,10):
        for j in range(-10,10):
            x,y = j*e1 + i*e2
#             print(i,j, (x,y))
            if check_inside(x,y, lines) == True:
                sites.append((j,i))
                sites_dict[(j,i)]= count
                count += 1
    return count, sites, sites_dict

def generate_connectivity_list(sites, sdict, lines, Tvecs, cvecs, e1, e2):
    connectivity = [[] for i in range(len(sites))]
    keys = sdict.keys()
#     print(keys)
    for n,s in enumerate(sites):
        for cvec in cvecs:
            s1 = s + cvec
            if tuple(s1) not in keys:
#                 print('here', s1)
                for lidx,l in enumerate(lines):
                    x,y = s1[0]*e1 + s1[1]*e2
#                     print(x,y)
                    if check_inside(x,y,[l]) == False:
                        s1 += Tvecs[lidx]
#                         print('u', s1)
#             print(s1)
            x,y = s1[0]*e1 + s1[1]*e2
            if check_inside(x,y, lines) == False:
                print('site ', s1, ' not in unit cell')
                return connectivity
            else:
                if tuple(s1) not in keys:
#                     print('Skipping site: ', s1)
                    continue
            connectivity[n].append(sdict[tuple(s1)])    
    return connectivity


def generate_connectivity_list_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2):
    connectivity = [[] for i in range(len(sites))]
    keys = sdict.keys()
#     print(keys)
    for n,s in enumerate(sites):
        for cvec in cvecs:
            s1 = s + cvec
            x,y = s1[0]*e1 + s1[1]*e2
            if check_inside(x,y, lines) == False:
#                 print('site ', s1, ' not in unit cell')
#                 return connectivity
                continue
            else:
                if tuple(s1) not in keys:
                    print('Skipping site: ', s1)
                    continue
                if sdict[tuple(s1)] not in connectivity[n]:
                    connectivity[n].append(sdict[tuple(s1)])   
    return connectivity


def generate_connectivity_list_PBC_OBC(sites, sdict, lines, Tvecs, cvecs, e1, e2):
    ''' Tvecs should only have the two tvecs which have PBC'''
    Tvecs2 = Tvecs[:2]
    lines2 = lines[:2]
    connectivity = [[] for i in range(len(sites))]
    keys = sdict.keys()
#     print(keys)
    for n,s in enumerate(sites):
        for cvec in cvecs:
            s1 = s + cvec
            if tuple(s1) not in keys:
#                 print('here', s1)
                for lidx,l in enumerate(lines2):
                    x,y = s1[0]*e1 + s1[1]*e2
#                     print(x,y)
                    if check_inside(x,y,[l]) == False:
                        s1 += Tvecs2[lidx]
#                         print('u', s1)
#             print(s1)
            x,y = s1[0]*e1 + s1[1]*e2
            if check_inside(x,y, lines) == False:
                print('site ', s1, ' not in unit cell')
                return connectivity
            else:
                if tuple(s1) not in keys:
#                     print('Skipping site: ', s1)
                    continue
            connectivity[n].append(sdict[tuple(s1)])    
    return connectivity


def generate_line_functions(Tvecs, e1, e2, flip=False):
    if flip == True:
        s = -1
    else: 
        s = 1
        
    Tvecs12 = Tvecs[0]
    Tvecs34 = Tvecs[2]
    def l1(x,y):
        (lx,ly) = Tvecs34[0]*e1 + Tvecs34[1]*e2
        return s*(x*ly - y*lx)
    def l2(x,y):
        (lx,ly) = Tvecs34[0]*e1 + Tvecs34[1]*e2
        (x0, y0) = (Tvecs12[0]*e1 + Tvecs12[1]*e2)*0.99
        return s*(-(x-x0)*ly + (y-y0)*lx)
    def l3(x,y):
        (lx,ly) = Tvecs12[0]*e1 + Tvecs12[1]*e2
        return s*(-x*ly + y*lx)
    def l4(x,y):
        (lx,ly) = Tvecs12[0]*e1 + Tvecs12[1]*e2
        (x0, y0) = (Tvecs34[0]*e1 + Tvecs34[1]*e2)*0.99
        return s*((x-x0)*ly - (y-y0)*lx)

    return [l1, l2, l3, l4]

def generate_lattice_sites_lieb(e1, e2, lines):
    sites = []
    sites_dict = dict()
    count = 0
    for i in range(-10,10):
        for j in range(-10,10):
            if (i%2 == 1) and (j%2==1):
                continue
            x,y = j*e1 + i*e2
#             print(i,j, (x,y))
            if check_inside(x,y, lines) == True:
                sites.append((j,i))
                sites_dict[(j,i)]= count
                count += 1
    return count, sites, sites_dict


def generate_bonds_list_PBC(sites, sdict, lines, Tvecs, disp_vecs, e1, e2):
    bonds = []
    keys = sdict.keys()
    for n,s in enumerate(sites):
        for cvec in disp_vecs:
            s1 = s + cvec
            if tuple(s1) not in keys:
#                 print('here', s1)
                for lidx,l in enumerate(lines):
                    x,y = s1[0]*e1 + s1[1]*e2
#                     print(x,y)
                    if check_inside(x,y,[l]) == False:
                        s1 += Tvecs[lidx]
#                         print('u', s1)
#             print(s1)
            x,y = s1[0]*e1 + s1[1]*e2
            if check_inside(x,y, lines) == False:
                print('site ', s1, ' not in unit cell')
                return bonds
            else:
                if tuple(s1) not in keys:
#                     print('Skipping site: ', s1)
                    continue
            bonds.append((n,sdict[tuple(s1)]))    
    return bonds

def generate_bonds_list_OBC(sites, sdict, lines, Tvecs, disp_vecs, e1, e2):
    bonds = []
    keys = sdict.keys()
    for n,s in enumerate(sites):
        for cvec in disp_vecs:
            s1 = s + cvec
            if tuple(s1) in sdict.keys():
                bonds.append((n,sdict[tuple(s1)]))
    return bonds
