"""

### Gadget2+3 reader

The GadgetFile class is stolen, isolated and adapted from the open source project "pynbody",
particoularly from the module pynbody.snapshot.gadget, revision 0618999
(see https://github.com/pynbody/pynbody/blob/master/pynbody/snapshot/gadget.py)

The routines that read particles in a box with the superindexing (e.g. Magneticum runs)
are ported from Klaus Dolag IDL scripts.

This library has not been tested and is distributed without any warranty. Have fun.

Antonio Ragagnin.

"""
import numpy as np
import struct
import sys
import copy
import os.path as path
import warnings
import errno
import itertools
import collections
import os
import math
import sys


debug=False
N_TYPE = 6

PY3 = sys.version_info[0] == 3

if PY3:
    string_types = str,
    _periodic=["POS "]
else:
    string_types = basestring,
    _periodic=[b"POS "]

def iterable(arg):
    return isinstance(arg, collections.Iterable) and not isinstance(arg, string_types)

def iterate(arg):
    if iterable(arg):
        return arg
    return [arg]


def _to_raw(s):
    if isinstance(s, str) and sys.version_info[0] > 2:
        return s.encode('utf-8')
    else:
        return s

def _to_str(s):
    if sys.version_info[0] > 2:
        return s.decode('utf-8')
    else:
        return s


_b = _to_raw
_s = _to_str

def printf(s,e=False):
    fd=sys.stderr if e else sys.stdout
    fd.write(s)


def periodic_axis(x,   periodic=None, center_c=None):
                    dx=np.array(x-center_c)
                    maskx1=dx-(periodic * 0.5)>0.
                    maskx2=dx-(-periodic * 0.5)<0.
                    dx[maskx1] = (dx-periodic)[maskx1]
                    dx[maskx2] =(dx+periodic)[maskx2]
                    return dx+center_c


def join_res(res, blocks,  join_ptypes, only_joined_ptypes, f=None):
        ptypes = [i for i in res if i!=-1]
        if join_ptypes:
            res[-1]={}
            for block in iterate(blocks):
                t=[]
                for i in res:
                    if i==-1: continue
                    if len(res[i][block])>0:
                        t.append(res[i][block])
                    else:
                        if f is not None and f.info is not None:
                            cols, dtype = f.get_data_shape(block,i)
                            shape = f.header.npart[i]
                            if cols>1:
                                shape = (f.header.npart[i],cols)
                            b =np.full(shape,np.nan, dtype=dtype)
                            if len(b)>0:
                                t.append(b)
                if len(t)>0:
                    res[-1][block] =np.concatenate(tuple(t))
                else:
                    res[-1][block] = np.array([])
            #for block in iterate(blocks):
            #    for i in res:
            #        print (block, i, res[i][block].shape)
            if debug:
                print(res)
            res[-1]['PTYPE']=np.concatenate(tuple(
                [np.zeros(len(res[i][iterate(blocks)[0]]),dtype=np.int32)+i  for i in res if i!=-1   ]
            ))
            #print(res[-1])
        if only_joined_ptypes:
            if iterable(blocks):
                res = res[-1]
            else:
                res = res[-1][blocks]
        else:

            if not iterable(ptypes):
                res = res[ptypes]
            if not iterable(blocks):
                res = res[blocks]
        return res

def to_spherical(xyzt,center):
    xyz = xyzt.T
    x       = xyz[0]-center[0]
    y       = xyz[1]-center[1]
    z       = xyz[2]-center[2]
    r       =  np.sqrt(x*x+y*y+z*z)
    r[r==0.]=-1.
    theta   =  np.arccos(z/r)
    r[r==-1.]=0.
    phi     =  np.arctan2(y,x)
    phi[r==0.]=0.
    theta[r==0.]=0.
    return np.array([r,theta,phi]).T

def to_cartesian(rthetaphi):

    r       = rthetaphi.T[0]
    theta   = rthetaphi.T[1]
    phi     = rthetaphi.T[2]
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return np.array([x,y,z])

def non_periodic_distance(pos1, pos2):
    return np.sqrt((pos1[0]-pos2[0])**2.+(pos1[1]-pos2[1])**2.+(pos1[2]-pos2[2])**2.)



def periodic_position(x, periodic=None, center=None):
    if periodic is not None:
        for axis in (0,1,2):
            x[:,axis]=periodic_axis(x[:,axis], periodic, center[axis])
    return x

def periodic_distance(pos1, pos2, periodic=None):
    pos2 = periodic_position(np.array([pos2]), periodic=periodic, center=pos1)[0]
    return non_periodic_distance(pos1, pos2)


class GadgetBlock(object):

    """Class to describe each block.
    Each block has a start, a length, and a length-per-particle"""

    def __init__(self, start=0, length=0, partlen=0, dtype=np.float32, ptypes=np.zeros(N_TYPE, bool)):
        # Start of block in file
        self.start = start
        # Length of block in file
        self.length = length
        # Bytes per particle in file
        self.partlen = partlen
        # Data type of block
        self.data_type = dtype
        # Types of particle this block contains
        self.ptypes = ptypes

def hexdump(src, length=16):
    FILTER = ''.join([(len(repr(chr(x))) == 3) and chr(x) or '.' for x in range(256)])
    lines = []
    for c in xrange(0, len(src), length):
        chars = src[c:c+length]
        hex = ' '.join(["%02x" % ord(x) for x in chars])
        printable = ''.join(["%s" % ((ord(x) <= 127 and FILTER[ord(x)]) or '.') for x in chars])
        lines.append("%04x  %-*s  %s\n" % (c, length*3, hex, printable))
    return ''.join(lines)


def _output_order_gadget(all_keys):
    out = []
    out_dregs = copy.copy(all_keys)
    for X in map(str.strip, config_parser.get('gadget-default-output', 'field-ordering').split(',')):
        if X in out_dregs:
            del out_dregs[out_dregs.index(X)]
            out.append(X)
    return out + out_dregs


def _construct_gadget_header(data, endian='='):
    """Create a GadgetHeader from a byte range read from a file."""
    npart = np.zeros(N_TYPE, dtype=np.uint32)
    mass = np.zeros(N_TYPE)
    time = 0.
    redshift = 0.
    npartTotal = np.zeros(N_TYPE, dtype=np.int32)
    num_files = 0
    BoxSize = 0.
    Omega0 = 0.
    OmegaLambda = 0.
    HubbleParam = 0.
    NallHW = np.zeros(N_TYPE, dtype=np.int32)
    if data == '':
        return
    fmt = endian + "IIIIIIddddddddiiIIIIIIiiddddiiIIIIIIiiif48s"
    if struct.calcsize(fmt) != 256:
        raise Exception(
            "There is a bug in gadget.py; the header format string is not 256 bytes")
    (npart[0], npart[1], npart[2], npart[3], npart[4], npart[5],
     mass[0], mass[1], mass[2], mass[3], mass[4], mass[5],
     time, redshift,  flag_sfr, flag_feedback,
     npartTotal[0], npartTotal[1], npartTotal[2], npartTotal[3], npartTotal[4], npartTotal[5],
     flag_cooling, num_files, BoxSize, Omega0, OmegaLambda, HubbleParam, flag_stellarage, flag_metals,
     NallHW[0], NallHW[1], NallHW[2], NallHW[3], NallHW[4], NallHW[5],
     flag_entropy_instead_u, flag_doubleprecision, flag_ic_info, lpt_scalingfactor, fill) = struct.unpack(fmt, data)

    header = GadgetHeader(npart, mass, time, redshift,
                          BoxSize, Omega0, OmegaLambda, HubbleParam, num_files)
    header.flag_sfr = flag_sfr
    header.flag_feedback = flag_feedback
    header.npartTotal = npartTotal
    header.flag_cooling = flag_cooling
    header.flag_stellarage = flag_stellarage
    header.flag_metals = flag_metals
    header.NallHW = NallHW
    header.flag_entropy_instead_u = flag_entropy_instead_u
    header.flag_doubleprecision = flag_doubleprecision
    header.flag_ic_info = flag_ic_info
    header.lpt_scalingfactor = lpt_scalingfactor
    header.endian = endian

    return header


class GadgetHeader(object):

    """Describes the header of gadget class files; this is all our metadata, so we are going to store it inline"""

    def __init__(self, npart, mass, time, redshift, BoxSize, Omega0, OmegaLambda, HubbleParam, num_files=1):
        "Construct a header from values, instead of a datastring."""
        assert(len(mass) == 6)
        assert(len(npart) == 6)
        # Massa of each particle type in this file. If zero,
        # particle mass stored in snapshot.
        self.mass = mass
        # Time of snapshot
        self.time = time
        # Redshift of snapshot
        self.redshift = redshift
        # Boolean to test the presence of star formation
        self.flag_sfr = False
        # Boolean to test the presence of feedback
        self.flag_feedback = False
        # Boolean to test the presence of cooling
        self.flag_cooling = False
        # Number of files expected in this snapshot
        self.num_files = num_files
        # Box size of the simulation
        self.BoxSize = BoxSize
        # Omega_Matter. Note this is Omega_DM + Omega_Baryons
        self.Omega0 = Omega0
        # Dark energy density
        self.OmegaLambda = OmegaLambda
        # Hubble parameter, in units where it is around 70.
        self.HubbleParam = HubbleParam
        # Boolean to test whether stars have an age
        self.flag_stellarage = False
        # Boolean to test the presence of metals
        self.flag_metals = False
        # flags that IC-file contains entropy instead of u
        self.flag_entropy_instead_u = False
        # flags that snapshot contains double-precision instead of single
        # precision
        self.flag_doubleprecision = False
        self.flag_ic_info = False
        # flag to inform whether IC files are generated with Zeldovich approximation,
        # or whether they contain 2nd order lagrangian perturbation theory ICs.
        #    FLAG_ZELDOVICH_ICS     (1)   - IC file based on Zeldovich
        #    FLAG_SECOND_ORDER_ICS  (2)   - Special IC-file containing 2lpt masses
        #    FLAG_EVOLVED_ZELDOVICH (3)   - snapshot evolved from Zeldovich ICs
        #    FLAG_EVOLVED_2LPT      (4)   - snapshot evolved from 2lpt ICs
        #    FLAG_NORMALICS_2LPT    (5)   - standard gadget file format with 2lpt ICs
        # All other values, including 0 are interpreted as "don't know" for
        # backwards compatability.
        self.lpt_scalingfactor = 0.    # scaling factor for 2lpt initial conditions
        self.endian = ""
        # Number of particles
        self.npart = np.array(npart, dtype=np.uint32)
        if (npart < 2 ** 31).all():
            # First 32-bits of total number of particles in the simulation
            self.npartTotal = np.array(npart, dtype=np.int32)
            # Long word of the total number of particles in the simulation.
            # At least one version of N-GenICs sets this to something entirely
            # different.
            self.NallHW = np.zeros(N_TYPE, dtype=np.int32)
        else:
            self.header.NallHW = np.array(npart // 2 ** 32, dtype=np.int32)
            self.header.npartTotal = np.array(
                npart - 2 ** 32 * self.header.NallHW, dtype=np.int32)

    def serialize(self):
        """This takes the header structure and returns it as a packed string"""
        fmt = self.endian + "IIIIIIddddddddiiIIIIIIiiddddiiIIIIIIiiif"
        # Do not attempt to include padding in the serialised data; the most common use of serialise
        # is to write to a file and we don't want to overwrite extra data that
        # might be present
        if struct.calcsize(fmt) != 256 - 48:
            raise Exception(
                "There is a bug in gadget.py; the header format string is not 256 bytes")
        # WARNING: On at least python 2.6.3 and numpy 1.3.0 on windows, castless code fails with:
        # SystemError: ..\Objects\longobject.c:336: bad argument to internal function
        # This is because self.npart, etc, has type np.uint32 and not int.
        # This is I think a problem with python/numpy, but cast things to ints
        # until I can determine how widespread it is.
        data = struct.pack(
            fmt, int(self.npart[0]), int(self.npart[1]), int(self.npart[
                                                             2]), int(
                                                                 self.npart[3]), int(
                                                                     self.npart[4]), int(
                                                                         self.npart[
                                                                             5]),
            self.mass[0], self.mass[1], self.mass[
                2], self.mass[3], self.mass[4], self.mass[5],
            self.time, self.redshift,  self.flag_sfr, self.flag_feedback,
            int(self.npartTotal[0]), int(self.npartTotal[1]), int(self.npartTotal[
                                                                  2]), int(
                                                                      self.npartTotal[3]), int(
                                                                          self.npartTotal[4]), int(
                                                                              self.npartTotal[
                                                                                  5]),
            self.flag_cooling, self.num_files, self.BoxSize, self.Omega0, self.OmegaLambda, self.HubbleParam, self.flag_stellarage, self.flag_metals,
            int(self.NallHW[0]), int(self.NallHW[1]), int(self.NallHW[
                                                          2]), int(
                                                              self.NallHW[3]), int(
                                                                  self.NallHW[4]), int(
                                                                      self.NallHW[
                                                                          5]),
            self.flag_entropy_instead_u, self.flag_doubleprecision, self.flag_ic_info, self.lpt_scalingfactor)
        return data


class GadgetFile(object):

    def __init__(self, filename,is_snap=True):
        #print('#patched __init__',filename)
        if(debug):
            printf("g3.read: opening '%s'\n"%(filename), e=True)
        self._filename = filename
        self.blocks = {}
        self.info = None
        self.endian = ''
        self.format2 = True
        t_part = 0

        fd = open(filename, "rb")
        self.check_format(fd)

        # If format 1, load the block definitions.
        if not self.format2:
            self.block_names = config_parser.get(
                'gadget-1-blocks', "blocks").split(",")
            self.block_names = [q.upper().ljust(4) for q in self.block_names]
            if sys.version_info[0] > 2:
                self.block_names = map(
                    lambda x: str.encode(x, 'utf-8'), self.block_names)
            # This is a counter for the fallback
            self.extra = 0
        while True:
            block = GadgetBlock()

            (name, block.length) = self.read_block_head(fd)
            #name = _s(name.decode('ascii'))

            if name == "    ":
                break
            # Do special things for the HEAD block
            if name[0:4] == "HEAD":
                if block.length != 256:
                    raise IOError("Mis-sized HEAD block in " + filename)
                self.header = fd.read(256)
                if len(self.header) != 256:
                    raise IOError("Could not read HEAD block in " + filename)
                self.header = _construct_gadget_header(
                    self.header, self.endian)
                record_size = self.read_block_foot(fd)
                if record_size != 256:
                    raise IOError("Bad record size for HEAD in " + filename)
                t_part = self.header.npart.sum()
                if  ((not self.format2) and
                        ((self.header.npart != 0) * (self.header.mass == 0)).sum()==0):
                    # The "Spec" says that if all the existing particle masses
                    # are in the header, we shouldn't have a MASS block
                    self.block_names.remove("MASS")
                continue
            # Set the partlen, using our amazing heuristics
            success = False
            if name[0:4] == "POS " or name[0:4] == "VEL ":
                if block.length == t_part * 24:
                    block.partlen = 24
                    block.data_type = np.float64
                else:
                    block.partlen = 12
                    block.data_type = np.float32
                block.ptypes = self.header.npart != 0
                success = True
            elif name[0:4] == "ID  ":
                # Heuristic for long (64-bit) IDs
                if block.length == t_part * 4:
                    block.partlen = 4
                    block.data_type = np.int32
                else:
                    block.partlen = 8
                    block.data_type = np.int64
                block.ptypes = self.header.npart != 0
                success = True

            block.start = fd.tell()
            # Check for the case where the record size overflows an int.
            # If this is true, we can't get record size from the length and we just have to guess
            # At least the record sizes at either end should be consistently wrong.
            # Better hope this only happens for blocks where all particles are
            # present.
            extra_len = t_part * block.partlen
            if extra_len >= 2 ** 32:
                fd.seek(extra_len, 1)
            else:
                fd.seek(block.length, 1)
            record_size = self.read_block_foot(fd)
            if record_size != block.length:
                raise IOError("Corrupt record in " +
                              filename + " footer for block " + name + "dtype" + str(block.data_type))
            if extra_len >= 2 ** 32:
                block.length = extra_len

            if not success:
                # Figure out what particles are here and what types
                # they have. This also is a heuristic, which assumes
                # that blocks are either fully present or not for a
                # given particle. It also has to try all
                # possibilities of dimensions of array and data type.
                for dim, tp in (1, np.float32), (1, np.float64), (3, np.float32), (3, np.float64), (11, np.float32):
                    try:
                        block.data_type = tp
                        block.partlen = np.dtype(tp).itemsize * dim
                        block.ptypes = self.get_block_types(
                            block, self.header.npart)
                        success = True
                        break
                    except ValueError:
                        continue
            self.blocks[name[0:4]] = block

            if not success and name=="INFO":
                self.info = {}
                il = 4+8+4*7
                oldpos = fd.tell()
                nblocks = block.length//il
                for i in range(nblocks):
                    fd.seek(block.start+il*i)
                    b = fd.read(il)
                    s= list(struct.unpack(self.endian+'4s8s7i', b))
                    s[0]=_s(s[0])
                    s[1]=_s(s[1])
                    self.info[s[0]]=s
                    if (debug):
                        printf("block='%s' type='%s' size='%d' ptype=%d %d %d %d %d %d"%(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8]),e =True
)
                fd.seek(oldpos)
                success=True

            if not success:
                pass
                #warnings.warn("Encountered a gadget block %r which could not be interpreted - is it a strange length or data type (length=%d)?" %
                #          (name, block.length), RuntimeWarning)
#            else:


        # and we're done.
        fd.close()

        # Make a mass block if one isn't found.
        if is_snap and 'MASS' not in self.blocks:
            block = GadgetBlock()
            block.length = 0
            block.start = 0

            #print(self.blocks)
            # Mass should be the same type as POS (see issue #321)
            block.data_type = self.blocks['POS '].data_type
            block.partlen = np.dtype(block.data_type).itemsize
            self.blocks[b'MASS'] = block

#        print('blocchi:', self.blocks)
    def get_block_types(self, block, npart):
        """ Set up the particle types in the block, with a heuristic,
        which assumes that blocks are either fully present or not for a given particle type"""

        totalnpart = npart.astype(np.int64).sum()
        if block.length == totalnpart * block.partlen:
            ptypes = np.ones(N_TYPE, bool)
            return ptypes
        ptypes = np.zeros(N_TYPE, bool)
        for blocknpart in [1, 2, 3, 4, 5]:
            # iterate of differeent possible combinations of particles in the bloc
            # we stop when we can we match the length of the block
            for perm in itertools.permutations(range(0, N_TYPE), blocknpart):
                # the 64-bit calculation is important here
                if block.length == (npart[list(perm)]).astype(np.int64).sum() * block.partlen:
                    ptypes[list(perm)] = True
                    return ptypes
        raise ValueError("Could not determine particle types for block")

    def check_format(self, fd):
        """This function reads the first character of a file and, depending on its value, determines
        whether we have a format 1 or 2 file, and whether the endianness is swapped. For the endianness,
        it then determines the correct byteorder string to pass to struct.unpack. There is not string
        for 'not native', so this is more complex than it needs to be"""
        fd.seek(0, 0)
        (r,) = struct.unpack('=I', fd.read(4))
        if r == 8:
            self.endian = '='
            self.format2 = True
        elif r == 134217728:
            if sys.byteorder == 'little':
                self.endian = '>'
            else:
                self.endian = '<'
            self.format2 = True
        elif r == 65536:
            if sys.byteorder == 'little':
                self.endian = '>'
            else:
                self.endian = '<'
            self.format2 = False
        elif r == 256:
            self.endian = '='
            self.format2 = False
        else:
            raise IOError("File corrupt. First integer is: " + str(r))
        fd.seek(0, 0)
        return

    def read_block_foot(self, fd):
        """Unpacks the block footer, into a single integer"""
        record_size = fd.read(4)
        if len(record_size) != 4:
            raise IOError("Could not read block footer")
        (record_size,) = struct.unpack(self.endian + 'I', record_size)
        return record_size

    def read_block_head(self, fd):
        """Read the Gadget 2 "block header" record, ie, 8 name, length, 8.
           Takes an open file and returns a (name, length) tuple """
        if self.format2:
            head = fd.read(5 * 4)
            # If we have run out of file, we don't want an exception,
            # we just want a zero length empty block
            if len(head) != 5 * 4:
                return ("    ", 0)
            head = struct.unpack(self.endian + 'I4sIII', head)
            if head[0] != 8 or head[3] != 8 or head[4] != head[2] - 8:
                raise IOError(
                    "Corrupt header record. Possibly incorrect file format",head)
                return ("____", 0)

            # Don't include the two "record_size" indicators in the total
            # length count
            return (_s(head[1]), head[2] - 8)
        else:
            record_size = fd.read(4)
            if len(record_size) != 4:
                return (_b("____"), 0)
            (record_size,) = struct.unpack(self.endian + 'I', record_size)
            try:
                name = self.block_names[0]
                self.block_names = self.block_names[1:]
            except IndexError:
                if self.extra == 0:
                    warnings.warn(
                        "Run out of block names in the config file. Using fallbacks: UNK*", RuntimeWarning)
                name = _to_raw("UNK" + str(self.extra))
                self.extra += 1
            return (name, record_size)
    def get_block(self, name, ptype, p_toread, p_start=None):
        """Get a particle range from this file, starting at p_start,
        and reading a maximum of p_toread particles"""
        #name = _to_raw(name)


        p_read = 0

        cur_block = self.blocks[name]
        parts = self.get_block_parts(name, ptype)


        if p_start==None: p_start = self.get_start_part(name, ptype)
        if p_toread > parts:
            p_toread = parts
        fd = open(self._filename, 'rb')
        if (debug):
            printf("partlen=%d"%cur_block.partlen,e=True)
        fd.seek(cur_block.start + int(cur_block.partlen * p_start), 0)
        # This is just so that we can get a size for the type
        dt = np.dtype(cur_block.data_type)
        if (debug):
            print("p_toread", p_toread, "cur_block.partlen", cur_block.partlen, " dt.itemsize",dt.itemsize)
        n_type = p_toread * cur_block.partlen // dt.itemsize
        if (debug):
            print("read dtype=", cur_block.data_type, " n_type=", n_type)
        data = np.fromfile(
            fd, dtype=cur_block.data_type, count=n_type, sep='')
        if self.endian != '=':
            data = data.byteswap(True)
        return (p_toread, data)



    def get_block_parts(self, name, ptype):
        """Get the number of particles present in a block in this file"""
        if name not in self.blocks:
            return 0
        cur_block = self.blocks[name]
        if ptype == -1:
            if (debug):
                print("cur block length=", cur_block.length," cur_block.partlen=",cur_block.partlen,"result", cur_block.length // cur_block.partlen)
            return cur_block.length // cur_block.partlen
        else:
            if (debug):
                print(" self.header.npart[ptype] =", self.header.npart[ptype] ," cur_block.ptypes[ptype]",  cur_block.ptypes[ptype]," res",  self.header.npart[ptype] * cur_block.ptypes[ptype])
            return self.header.npart[ptype] * cur_block.ptypes[ptype]

    def get_start_part(self, name, ptype):
        """Find particle to skip to before starting, if reading particular type"""
        if ptype == -1:
            return 0
        else:
            if name not in self.blocks:
                return 0
            cur_block = self.blocks[name]
            return (cur_block.ptypes * self.header.npart)[0:ptype].sum().astype(int)

    def get_block_dims(self, name):
        """Get the dimensionality of the block, eg, 3 for POS, 1 for most other things"""
        if name not in self.blocks:
            return 0
        cur_block = self.blocks[name]
        dt = np.dtype(cur_block.data_type)
        return cur_block.partlen // dt.itemsize

    # The following functions are for writing blocks back to the file
    def write_block(self, name, ptype, big_data, filename=None):
        """Write a full block of data in this file. Any particle type can be written. If the particle type is not present in this file,
        an exception KeyError is thrown. If there are too many particles, ValueError is thrown.
        big_data contains a reference to the data to be written. Type -1 is all types"""
        try:
            cur_block = self.blocks[name]
        except KeyError:
            raise KeyError("Block " + name + " not in file " + self._filename)

        parts = self.get_block_parts(name, ptype)
        p_start = self.get_start_part(name, ptype)
        MinType = np.ravel(np.where(cur_block.ptypes * self.header.npart))[0]
        MaxType = np.ravel(np.where(cur_block.ptypes * self.header.npart))[-1]
        # Have we been given the right number of particles?
        if np.size(big_data) > parts * self.get_block_dims(name):
            raise ValueError("Space for " + str(parts) + " particles of type " + str(
                ptype) + " in file " + self._filename + ", " + str(np.shape(big_data)[0]) + " requested.")
        # Do we have the right type?
        dt = np.dtype(cur_block.data_type)
        bt = big_data.dtype
        if bt.kind != dt.kind:
            raise ValueError("Data of incorrect type passed to write_block")
        # Open the file
        if filename == None:
            fd = open(self._filename, "r+b")
        else:
            fd = open(filename, "r+b")
        # Seek to the start of the block
        fd.seek(cur_block.start + cur_block.partlen * p_start, 0)
        # Add the block header if we are at the start of a block
        if ptype == MinType or ptype < 0:
            data = self.write_block_header(name, cur_block.length)
            # Better seek back a bit first.
            fd.seek(-len(data), 1)
            fd.write(data)

        if self.endian != '=':
            big_data = big_data.byteswap(False)

        # Actually write the data
        # Make sure to ravel it, otherwise the wrong amount will be written,
        # because it will also write nulls every time the first array dimension
        # changes.
        d = np.ravel(big_data.astype(dt)).tostring()
        fd.write(d)
        if ptype == MaxType or ptype < 0:
            data = self.write_block_footer(name, cur_block.length)
            fd.write(data)

        fd.close()

    def read_new(self, blocks, ptypes, join_ptypes=True, only_joined_ptypes=True, periodic=_periodic, center=None):

        if iterable(blocks): _blocks=blocks
        else: _blocks = [blocks]

        if iterable(ptypes):
            _ptypes=ptypes
        else:
            if ptypes==-1:
                only_joined_ptypes=True
                _ptypes=[0,1,2,3,4,5]
            else:
                _ptypes = [ptypes]

        res={}
        for block in _blocks:

            for ptype in _ptypes:
                if ptype not in res:
                    res[ptype]={}
                f_data = self.read(block, ptype, center=center)
                res[ptype][block] = f_data

        return  join_res(res, blocks, join_ptypes, only_joined_ptypes, f=self)



    def get_data_shape(self, g_name, ptype):
        try:
            if g_name=="INFO" or self.info is None or (g_name not in self.info and g_name in self.blocks):
                dtype=np.float32
        except:
            print (g_name)
            print (ptype)
            print (self.info)
            print(self.blocks)
        if g_name=="INFO" or self.info is None or (g_name not in self.info and g_name in self.blocks):
           dtype=np.float32
           partlen = self.blocks[g_name].partlen
           if g_name=="ID  ":
               dtype=np.uint64
           dim = np.dtype(dtype).itemsize
           cols = int(partlen/dim)
           return cols,dtype
        elif g_name == "MASS" and "MASS" not in self.info:
            return 1,np.float32
        else:
           info = self.info
           if g_name not in self.info and g_name!="MASS":
               #print(ptype, self.header.mass, self.header.mass[ptype])
               raise Exception("block not found %s"%g_name)
           elif g_name not in self.info:
               return 1,np.float32
           binfo = self.info[g_name]
           stype = binfo[1]
           sdim = int(binfo[2])
           cols=1
           if stype=="FLOAT   ": dtype = np.float32
           if stype=="FLOATN  ": dtype,cols = np.float32,sdim
           if stype=="LONG    ": dtype=np.int32
           if stype=="LLONG   ": dtype=np.int64
           if stype=="DOUBLE  ": dtype=np.float64
           if stype=="DOUBLEN ": dtype,cols=np.float64,sdim
           self.blocks[g_name].partlen = dtype().nbytes*cols
           partlen = self.blocks[g_name].partlen
        return cols,dtype

    def read(self, block, ptype, p_toread=None, p_start=None, periodic=_periodic, center=None):

       if(debug):
            printf("g3.read: reading '%s'/%s/'%s'\n"%(self._filename, str(block), str(ptype)), e=True)

       if block=='MASS' and self.header.mass[ptype]>0:
            if p_toread == None:
                return np.zeros(self.header.npart[ptype])+self.header.mass[ptype]
            else:
                l = p_toread
                return np.zeros(l)+self.header.mass[ptype]


       g_name = block
       cols,dtype = self.get_data_shape (g_name, ptype)
       #print(g_name, cols, dtype)
       if p_toread is None:
           if (debug):
               print("get block parts()")
           f_parts = self.get_block_parts(g_name, ptype)
       else:
           f_parts = p_toread

       if (debug):
           print(f_parts, p_start)

       #print(ptype, g_name,  f_parts, p_start, self.header.npart[ptype], self.header.mass[ptype])
       if (g_name == "MASS" and self.info is not None and
           "MASS" not in self.info and
           p_start is None and
           self.header.npart[ptype]>0 and
           self.header.mass[ptype]>0.):
           f_read = f_parts
           f_data = np.full(f_read, self.header.mass[ptype])
       else:
           (f_read, f_data) = self.get_block(g_name, ptype, f_parts, p_start)


       if f_read != f_parts:
            raise IOError("Read of " + self._filename + " asked for " + str(
                    f_parts) + " particles but got " + str(f_read))
       if(debug):
           print(f_data.shape, cols, len(f_data)/cols)
       f_data.dtype = dtype

       if(debug):
           print(block, ptype, f_data.shape,'cols', cols, 'f_data',len(f_data), 'f_data/cols', len(f_data)/cols)
       if cols>1: f_data.shape = (int(len(f_data)/cols),cols)

       if periodic is not None and block in _periodic and center is not None:
           f_data = periodic_position(f_data, periodic=self.header.BoxSize, center=center)
       return f_data

    def add_file_block(self, name, blocksize, partlen=4, dtype=np.float32, ptypes=-1):
        """Add a block to the block table at the end of the file. Do not actually write anything"""
        name = _to_raw(name)

        if name in self.blocks:
            raise KeyError(
                "Block " + name + " already present in file. Not adding")

        def st(val):
            return val.start
        # Get last block
        lb = max(self.blocks.values(), key=st)

        if np.issubdtype(dtype, float):
            dtype = np.float32  # coerce to single precision

        # Make new block
        block = GadgetBlock(length=blocksize, partlen=partlen, dtype=dtype)
        block.start = lb.start + lb.length + 6 * \
            4  # For the block header, and footer of the previous block
        if ptypes == -1:
            block.ptypes = np.ones(N_TYPE, bool)
        else:
            block.ptypes = ptypes
        self.blocks[name] = block

    def write_block_header(self, name, blocksize):
        """Create a string for a Gadget-style block header, but do not actually write it, for atomicity."""
        if self.format2:
            # This is the block header record, which we want for format two
            # files only
            blkheadsize = 4 + 4 * 1
            # 1 int and 4 chars
            nextblock = blocksize + 2 * 4
            # Relative location of next block; the extra 2 uints are for storing the headers.
            # Write format 2 header header
            head = struct.pack(
                self.endian + 'I4sII', blkheadsize, name, nextblock, blkheadsize)
        # Also write the record size, which we want for all files*/
        head += self.write_block_footer(name, blocksize)
        return head

    def write_block_footer(self, name, blocksize):
        """(Re) write a Gadget-style block footer."""
        return struct.pack(self.endian + 'I', blocksize)

    def write_header(self, head_in, filename=None):
        """Write a file header. Overwrites npart in the argument with the npart of the file, so a consistent file is always written."""
        # Construct new header with the passed header and overwrite npart with the file header.
        # This has ref. semantics so use copy
        head = copy.deepcopy(head_in)
        head.npart = np.array(self.header.npart)
        data = self.write_block_header("HEAD", 256)
        data += head.serialize()
        if filename == None:
            filename = self._filename
        # a mode will ignore the file position, and w truncates the file.
        try:
            fd = open(filename, "r+b")
        except IOError as err:
            # If we couldn't open it because it doesn't exist open it for
            # writing.
            if err == errno.ENOENT:
                fd = open(filename, "w+b")
            # If we couldn't open it for any other reason, reraise exception
            else:
                raise IOError(err)
        fd.seek(0)  # Header always at start of file
        # Write header
        fd.write(data)
        # Seek 48 bytes forward, to skip the padding (which may contain extra
        # data)
        fd.seek(48, 1)
        data = self.write_block_footer("HEAD", 256)
        fd.write(data)
        fd.close()


class GadgetWriteFile (GadgetFile):

    """Class for write-only snapshots, as when we are creating a new set of files from, eg, a TipsySnap.
        Should not be used directly. block_names is a list so we can specify an on-disc ordering."""

    def __init__(self, filename, npart, block_names, header, format2=True):
        self.header = header
        self._filename = filename
        self.endian = '='  # write with default endian of this system
        self.format2 = format2
        self.blocks = {}
        self.header.npart = np.array(npart)
        # Set up the positions
        header_size = 4
        if format2:
            header_size += 3 * 4 + 4
        footer_size = 4
        # First block is just past the header.
        cur_pos = 256 + header_size + footer_size
        for block in block_names:
            # Add block if present for some types
            if block.types.sum():
                b_part = npart * block.types
                b = GadgetBlock(
                    start=cur_pos + header_size, partlen=block.partlen,
                    length=block.partlen * b_part.sum(), dtype=block.dtype, ptypes=block.types)
                cur_pos += b.length + header_size + footer_size
                self.blocks[_to_raw(block.name)] = b


class WriteBlock:

    """Internal structure for passing data around between file and snapshot"""

    def __init__(self, partlen=4, dtype=np.float32, types=np.zeros(N_TYPE, bool), name="    "):

        if np.issubdtype(dtype, float):
            dtype = np.float32
        if np.issubdtype(dtype, int):
            dtype = np.int32

        self.partlen = partlen * np.dtype(dtype).itemsize
        self.dtype = dtype

        self.types = types
        self.name = name




rottable3= np.array([
        [36, 28, 25, 27, 10, 10, 25, 27],
        [29, 11, 24, 24, 37, 11, 26, 26],
        [8, 8, 25, 27, 30, 38, 25, 27],
        [9, 39, 24, 24, 9, 31, 26, 26],
        [40, 24, 44, 32, 40, 6, 44, 6],
        [25, 7, 33, 7, 41, 41, 45, 45],
        [4, 42, 4, 46, 26, 42, 34, 46],
        [43, 43, 47, 47, 5, 27, 5, 35],
        [33, 35, 36, 28, 33, 35, 2, 2],
        [32, 32, 29, 3, 34, 34, 37, 3],
        [33, 35, 0, 0, 33, 35, 30, 38],
        [32, 32, 1, 39, 34, 34, 1, 31],
        [24, 42, 32, 46, 14, 42, 14, 46],
        [43, 43, 47, 47, 25, 15, 33, 15],
        [40, 12, 44, 12, 40, 26, 44, 34],
        [13, 27, 13, 35, 41, 41, 45, 45],
        [28, 41, 28, 22, 38, 43, 38, 22],
        [42, 40, 23, 23, 29, 39, 29, 39],
        [41, 36, 20, 36, 43, 30, 20, 30],
        [37, 31, 37, 31, 42, 40, 21, 21],
        [28, 18, 28, 45, 38, 18, 38, 47],
        [19, 19, 46, 44, 29, 39, 29, 39],
        [16, 36, 45, 36, 16, 30, 47, 30],
  [37, 31, 37, 31, 17, 17, 46, 44],
  [12, 4, 1, 3, 34, 34, 1, 3],
  [5, 35, 0, 0, 13, 35, 2, 2],
  [32, 32, 1, 3, 6, 14, 1, 3],
  [33, 15, 0, 0, 33, 7, 2, 2],
  [16, 0, 20, 8, 16, 30, 20, 30],
  [1, 31, 9, 31, 17, 17, 21, 21],
  [28, 18, 28, 22, 2, 18, 10, 22],
  [19, 19, 23, 23, 29, 3, 29, 11],
  [9, 11, 12, 4, 9, 11, 26, 26],
  [8, 8, 5, 27, 10, 10, 13, 27],
  [9, 11, 24, 24, 9, 11, 6, 14],
  [8, 8, 25, 15, 10, 10, 25, 7],
  [0, 18, 8, 22, 38, 18, 38, 22],
  [19, 19, 23, 23, 1, 39, 9, 39],
  [16, 36, 20, 36, 16, 2, 20, 10],
  [37, 3, 37, 11, 17, 17, 21, 21],
  [4, 17, 4, 46, 14, 19, 14, 46],
  [18, 16, 47, 47, 5, 15, 5, 15],
  [17, 12, 44, 12, 19, 6, 44, 6],
  [13, 7, 13, 7, 18, 16, 45, 45],
  [4, 42, 4, 21, 14, 42, 14, 23],
  [43, 43, 22, 20, 5, 15, 5, 15],
  [40, 12, 21, 12, 40, 6, 23, 6],
  [13, 7, 13, 7, 41, 41, 22, 20]])

subpix3 = np.array([
        [0, 7, 1, 6, 3, 4, 2, 5],
  [7, 4, 6, 5, 0, 3, 1, 2],
  [4, 3, 5, 2, 7, 0, 6, 1],
  [3, 0, 2, 1, 4, 7, 5, 6],
  [1, 0, 6, 7, 2, 3, 5, 4],
  [0, 3, 7, 4, 1, 2, 6, 5],
  [3, 2, 4, 5, 0, 1, 7, 6],
  [2, 1, 5, 6, 3, 0, 4, 7],
  [6, 1, 7, 0, 5, 2, 4, 3],
  [1, 2, 0, 3, 6, 5, 7, 4],
  [2, 5, 3, 4, 1, 6, 0, 7],
  [5, 6, 4, 7, 2, 1, 3, 0],
  [7, 6, 0, 1, 4, 5, 3, 2],
  [6, 5, 1, 2, 7, 4, 0, 3],
  [5, 4, 2, 3, 6, 7, 1, 0],
  [4, 7, 3, 0, 5, 6, 2, 1],
  [6, 7, 5, 4, 1, 0, 2, 3],
  [7, 0, 4, 3, 6, 1, 5, 2],
  [0, 1, 3, 2, 7, 6, 4, 5],
  [1, 6, 2, 5, 0, 7, 3, 4],
  [2, 3, 1, 0, 5, 4, 6, 7],
  [3, 4, 0, 7, 2, 5, 1, 6],
  [4, 5, 7, 6, 3, 2, 0, 1],
  [5, 2, 6, 1, 4, 3, 7, 0],
  [7, 0, 6, 1, 4, 3, 5, 2],
  [0, 3, 1, 2, 7, 4, 6, 5],
  [3, 4, 2, 5, 0, 7, 1, 6],
  [4, 7, 5, 6, 3, 0, 2, 1],
  [6, 7, 1, 0, 5, 4, 2, 3],
  [7, 4, 0, 3, 6, 5, 1, 2],
  [4, 5, 3, 2, 7, 6, 0, 1],
  [5, 6, 2, 1, 4, 7, 3, 0],
  [1, 6, 0, 7, 2, 5, 3, 4],
  [6, 5, 7, 4, 1, 2, 0, 3],
  [5, 2, 4, 3, 6, 1, 7, 0],
  [2, 1, 3, 0, 5, 6, 4, 7],
  [0, 1, 7, 6, 3, 2, 4, 5],
  [1, 2, 6, 5, 0, 3, 7, 4],
  [2, 3, 5, 4, 1, 0, 6, 7],
  [3, 0, 4, 7, 2, 1, 5, 6],
  [1, 0, 2, 3, 6, 7, 5, 4],
  [0, 7, 3, 4, 1, 6, 2, 5],
  [7, 6, 4, 5, 0, 1, 3, 2],
  [6, 1, 5, 2, 7, 0, 4, 3],
  [5, 4, 6, 7, 2, 3, 1, 0],
  [4, 3, 7, 0, 5, 2, 6, 1],
  [3, 2, 0, 1, 4, 5, 7, 6],
  [2, 5, 1, 6, 3, 4, 0, 7]])


def  peano_hilbert_key (head, x, y, z, integer_pos=False):
    bits = head.header.flag_feedback
    rotation = 0;
    key = 0;
    mask = 1<<(bits-1);
    for imask in range( 0, bits ) :
     #print('pix',x,y,z,mask)
     #print('mask',  (4  if ((x & mask) > 0) else 0) , ( 2 if     (    (y & mask) > 0) else  0) , ( 1 if ((z & mask) > 0) else 0))
     pix = (  4  if ((x & mask) > 0) else 0) + ( 2 if     (    (y & mask) > 0) else  0) + ( 1 if ((z & mask) > 0) else 0)
     #print('pix2',pix)
     key = key<<3
     #print('prk',pix,rotation,key,rottable3.shape,subpix3.shape)
     m=subpix3[rotation,pix]
     key = key | m
     rotation = rottable3[rotation,pix]

     mask=mask>>1
     #print key
    return key


def find_files_for_keys(myname,keylist):
    import os
    name=myname+'.key.index'

    f=np.fromfile(name, dtype=np.int32)
    n=f[0]
    size= os.stat(name).st_size
    f=f[1:]
    f.dtype=np.int32
    if size == 4+n*8*2+n*4:
            low_list=np.array(f[0:2*n])
            low_list.dtype=np.int64
            high_list=np.array(f[2*n:4*n])
            high_list.dtype=np.int64
            file_list=np.array(f[4*n:5*n],dtype=np.int32)
    else:
            low_list=np.array(f[0:n],dtype=np.int32)
            high_list=np.array(f[n:2*n],dtype=np.int32)
            file_list=np.array(f[2*n:3*n],dtype=np.int32)



    mask=np.array(np.zeros(len(low_list)),dtype=np.bool_)
    for key in keylist:
        #print('keys',n,low_list.shape,high_list.shape,file_list.shape,keylist,key)
        #print(file_list[low_list >= key])
        #print(file_list[high_list <= key])
        mask = mask | ((key >= low_list ) & (key <=high_list))
    ifiles=np.unique(np.sort(file_list[mask]))


    return ifiles


def read_particles_in_files(myname,blocks,ptypes, periodic=True , center=None, join_ptypes=True, only_joined_ptypes=True):
    if not iterable(myname):
        myname = [myname]
    res = {}
    for mysnap in myname:
      f=GadgetFile(mysnap,is_snap=False)
      for ptype in iterate(ptypes):
          if ptype not in res:
              res[ptype]={}
          for block in iterate(blocks):
            x = f.read_new(block, ptype, join_ptypes=True, only_joined_ptypes=True, center=center if periodic else None)
            res[ptype][block]=x
    return  join_res(res, blocks, join_ptypes, only_joined_ptypes, f=f)

def read_particles_only_superindex(mmyname,blocks,keylist, ptypes, periodic=True, center=None, join_ptypes=True, only_joined_ptypes=True):
    res={}
    ifiles=find_files_for_keys(mmyname,keylist)
    mynames=[]

    for ifile in ifiles:
        mynames.append(mmyname+'.'+str(ifile))
    return read_particles_in_files(mynames,blocks,ptypes, periodic=periodic, center=center, join_ptypes=join_ptypes, only_joined_ptypes=only_joined_ptypes)


def read_particles_given_key(mmyname,blocks,keylist, ptypes,periodic=True,center=None, join_ptypes=True, only_joined_ptypes=True):
    res={}
    ifiles=find_files_for_keys(mmyname,keylist)
    f=None
    if iterable(blocks): _blocks=blocks
    else: _blocks = [blocks]
    if iterable(ptypes):
        _ptypes=ptypes
    else:
        if ptypes==-1:
            only_joined_ptypes=True
            _ptypes=[0,1,2,3,4,5]
        else:
            _ptypes = [ptypes]

    for ifile in ifiles:
      myname=mmyname+'.'+str(ifile)+'.key'
      mysnap=mmyname+'.'+str(ifile)
      gkeyf=GadgetFile(myname,is_snap=False)
      f=GadgetFile(mysnap, is_snap=False)
      for ptype in iterate(_ptypes):
        if ptype not in res:
            res[ptype]={}
        tr=gkeyf.get_block_parts("KEY ",ptype)
        rkeys_in_file=gkeyf.get_block("KEY ",ptype,tr)
        keys_in_file=rkeys_in_file[1]
        keys_in_file.dtype=np.int32
        ii=np.arange(len(keys_in_file))[np.in1d(keys_in_file,keylist)]
        tr=gkeyf.get_block_parts("NKEY",ptype)
        pkey=gkeyf.get_block("NKEY",ptype,tr)[1]
        pkey.dtype=np.int32
        tr=gkeyf.get_block_parts("OKEY",ptype)
        okey=gkeyf.get_block("OKEY",ptype,tr)[1]
        okey.dtype=np.int32
        pkey=pkey[ii]
        okey=okey[ii]
        jj=np.argsort(okey)

        pkey=pkey[jj]
        okey=okey[jj]

        use_block=(np.zeros(len(okey),dtype=np.bool_)) | True
        icount=0

        for i in range(1,len(okey)-1):
            #WATHEVER
            if(okey[i] == okey[icount]+pkey[icount]):
                pkey[icount]+=pkey[i]
                use_block[i]=False
            else:
                icount=i

        okey=okey[use_block]
        pkey=pkey[use_block]

        for block in iterate(_blocks):
            if block!='MASS' and not f.blocks[block].ptypes[ptype]:
                continue
            if block not in res[ptype]:
                res[ptype][block]=[]

            if len(okey)==0:
                continue
            for i in range(0,len(okey)):

                o=okey[i]+f.get_start_part(block, ptype)
                p=pkey[i]
                x=f.read(block, ptype, p,p_start=o, center=center)
                res[ptype][block].append(x)
    for ptype in iterate(_ptypes):
        for block in iterate(blocks):
            if block in res[ptype] and len(res[ptype][block])>0:
                res[ptype][block] = np.concatenate(res[ptype][block])
            else:
                res[ptype][block] = np.array([])
    if debug:
        print(res)

    return join_res(res, blocks, join_ptypes, only_joined_ptypes, f=f)


def read_particles_in_box(snap_file_name,center,d,blocks,ptypes,has_super_index=True, has_keys=True,  join_ptypes=True, only_joined_ptypes=True):
        ce=np.array(center)
        if not os.path.isfile(snap_file_name+'.key.index'):
            has_super_index=False

        if debug:
            print("1", "has_super_index",  has_super_index, "has_keys", has_keys)

        if   not   os.path.isfile(snap_file_name+".0.key"):
            has_keys = False


        #print("2", "has_super_index",  has_super_index, "has_keys", has_keys)
        if not has_super_index:
            #print("3", "has_super_index",  has_super_index, "has_keys", has_keys)
            return read_particles_in_files(snap_file_name,blocks, ptypes=ptypes,center=ce,periodic=GadgetFile(snap_file_name).header.BoxSize,  join_ptypes=join_ptypes, only_joined_ptypes=only_joined_ptypes)

        fr=ce-d
        to=ce+d

        hkey=GadgetFile(snap_file_name+".0.key",is_snap=False)
        if "KEY " not in hkey.blocks.keys():
            has_keys = False
        corner=hkey.header.mass[0:3]
        fac=hkey.header.mass[3]
        bits=hkey.header.flag_feedback

        if debug:
            print (" r search ",d)
        #print(fr,to)
        fr=np.array(list(map(lambda x: min(x[0],x[1]), np.array([fr,to]).T)))
        to=np.array(list(map(lambda x: max(x[0],x[1]), np.array([fr,to]).T)))
        nkeys=1
        ifr=np.array(list(map(lambda x: math.floor(x), (fr-corner)*fac)   )    ,dtype=np.int64)
        ito=np.array(list(map(lambda x: math.ceil(x), (to-corner)*fac)    )   ,dtype=np.int64)
        keylist=[]
        """
        for i in range(ifr[0]-1,ito[0]+1):
            for j in range(ifr[1]-1,ito[1]+1):
                for k in range(ifr[2]-1,ito[2]+1):
                    #print(i,j,k)
                    keylist.append(peano_hilbert_key(hkey, i,j,k, integer_pos=True))
        """
        for i in range(ifr[0]-2,ito[0]+1):
            for j in range(ifr[1]-2,ito[1]+1):
                for k in range(ifr[2]-2,ito[2]+1):
                    keylist.append(peano_hilbert_key(hkey, i,j,k, integer_pos=True))
                    if debug:
                        print (i,j,k)
        if debug:
            print("4", "has_super_index",  has_super_index, "has_keys", has_keys)
        if has_keys:
            return read_particles_given_key(snap_file_name,blocks,keylist,ptypes=ptypes,
                                            center=ce,periodic=hkey.header.BoxSize, join_ptypes=join_ptypes, only_joined_ptypes=only_joined_ptypes)
        elif has_super_index:

            return read_particles_only_superindex(snap_file_name,blocks,keylist,ptypes=ptypes, center=ce,
                                                 periodic=hkey.header.BoxSize, join_ptypes=join_ptypes, only_joined_ptypes=only_joined_ptypes)
        else:
            raise Exception("%s not sure if the file has key/superindexes or not."%(snap_file_name))


def get_keys(snap_base, center, d):
    ce = np.array(center)
    fr=ce-d
    to=ce+d

    hkey=GadgetFile(snap_base+".0.key",is_snap=False)
    if "KEY " not in hkey.blocks.keys():
        has_keys = False
    corner=hkey.header.mass[0:3]
    fac=hkey.header.mass[3]

    if debug:
        print (" r search ",d)

    fr=np.array(list(map(lambda x: min(x[0],x[1]), np.array([fr,to]).T)))
    to=np.array(list(map(lambda x: max(x[0],x[1]), np.array([fr,to]).T)))
    nkeys=1
    ifr=np.array(list(map(lambda x: math.floor(x), (fr-corner)*fac)   )    ,dtype=np.int64)
    ito=np.array(list(map(lambda x: math.ceil(x), (to-corner)*fac)    )   ,dtype=np.int64)
    keylist=[]
    for i in range(ifr[0]-2,ito[0]+1):
        for j in range(ifr[1]-2,ito[1]+1):
            for k in range(ifr[2]-2,ito[2]+1):
                keylist.append(peano_hilbert_key(hkey, i,j,k, integer_pos=True))
                if debug:
                    print (i,j,k)

    return keylist


def get_multiple_keys(snap_base, center, d):


    hkey=GadgetFile(snap_base+".0.key",is_snap=False)
    if "KEY " not in hkey.blocks.keys():
        has_keys = False
    corner=hkey.header.mass[0:3]
    fac=hkey.header.mass[3]

    if debug:
        print (" r search ",d)

    keylist = len(center)*[None]
    ce = np.array(center)
    for center_idx in range(len(center)):
        fr=ce[center_idx,:] - d
        to=ce[center_idx,:] + d
        fr=np.array(list(map(lambda x: min(x[0],x[1]), np.array([fr,to]).T)))
        to=np.array(list(map(lambda x: max(x[0],x[1]), np.array([fr,to]).T)))
        nkeys=1
        ifr=np.array(list(map(lambda x: math.floor(x), (fr-corner)*fac)   )    ,dtype=np.int64)
        ito=np.array(list(map(lambda x: math.ceil(x), (to-corner)*fac)    )   ,dtype=np.int64)

        keylist[center_idx] = []
        for i in range(ifr[0]-2,ito[0]+1):
            for j in range(ifr[1]-2,ito[1]+1):
                for k in range(ifr[2]-2,ito[2]+1):
                    keylist[center_idx].append(peano_hilbert_key(hkey, i,j,k, integer_pos=True))
                    if debug:
                        print (i,j,k)

    return keylist


def read_new(filename, blocks, ptypes, join_ptypes=True, only_joined_ptypes=True, periodic=_periodic, center=None, is_snap=False):
    return GadgetFile(filename, is_snap=is_snap).read_new(blocks, ptypes, join_ptypes=True, only_joined_ptypes=True, periodic=_periodic, center=center)
