from nose.tools import *
from nose.plugins.skip import SkipTest

import mx

# mxmodule: creating (non-wrapped) mxArray objects of various sizes.

def bare_cell(dims):
    return mx.create_cell_array(dims, wrap=False)    

def test_bare_cell():
    '''
    Testing bare cell creation
    '''
    size = 3
    yield bare_cell, ()    
    for i in range(size): yield bare_cell, (i,)
    for i in range(size):
        for j in range(size):
            yield bare_cell, (i,j)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                yield bare_cell, (i,j,k)

@raises(ValueError)
def test_negative_cell():
    '''
    Test that cells of negative size not allowed
    '''
    bare_cell((-1,))

def bare_array(dims, mxclass=mx.DOUBLE):
    return mx.create_numeric_array(dims, mxclass, wrap=False)

def test_bare_numerics():
    '''
    Testing bare numeric creation
    '''
    size = 3
    mxclasses = [mx.DOUBLE, mx.SINGLE,
                 mx.INT8, mx.UINT8,
                 mx.INT16, mx.UINT16,
                 mx.INT32, mx.UINT32,
                 mx.INT64, mx.UINT64,
                 mx.CHAR, mx.LOGICAL]
    for c in mxclasses:
        yield bare_array, (), c
        for i in range(size): yield bare_array, (i,), c
        for i in range(size):
            for j in range(size):
                yield bare_array, (i,j), c
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    yield bare_array, (i,j,k), c

@raises(ValueError)
def test_negative_array():
    '''
    Test that arrays of negative size not allowed
    '''
    bare_array((-1,))

def bare_struct(dims):
    return mx.create_struct_array(dims, wrap=False)


def test_bare_struct():
    '''
    Testing bare struct creation
    '''
    size = 3
    yield bare_struct, ()    
    for i in range(size): yield bare_struct, (i,)
    for i in range(size):
        for j in range(size):
            yield bare_struct, (i,j)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                yield bare_struct, (i,j,k)

@raises(ValueError)
def test_negative_struct():
    '''
    Test that structs of negative size not allowed
    '''
    bare_struct((-1,))

def bare_char(dims):
    return mx.create_char_array(dims, wrap=False)

def test_bare_char():
    '''
    Testing bare char creation
    '''
    size = 3
    yield bare_char, ()    
    for i in range(size): yield bare_char, (i,)
    for i in range(size):
        for j in range(size):
            yield bare_char, (i,j)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                yield bare_char, (i,j,k)

@raises(ValueError)
def test_negative_char():
    '''
    Test that strings of negative size not allowed
    '''
    bare_char((-1,))


def bare_string(text):
    return mx.create_string(text, wrap=False)

loremipsum = '''
    Lorem ipsum dolor sit amet, consectetuer adipiscing elit, 
    sed diam nonummy nibh euismod tincidunt ut laoreet dolore 
    magna aliquam erat volutpat. Ut wisi enim ad minim veniam, 
    quis nostrud exerci tation ullamcorper suscipit lobortis 
    nisl ut aliquip ex ea commodo consequat. Duis autem vel 
    eum iriure dolor in hendrerit in vulputate velit esse 
    molestie consequat, vel illum dolore eu feugiat nulla 
    facilisis at vero eros et accumsan et iusto odio dignissim 
    qui blandit praesent luptatum zzril delenit augue duis 
    dolore te feugait nulla facilisi.
    '''

def test_bare_string():
    '''
    Testing bare string creation
    '''
    for i in range(0,len(loremipsum),16):
        yield bare_string, loremipsum[0:i]

def test_bare_funchandle():
    '''
    Test creation of bare function handle
    '''
    return mx.create_function_handle(name="sin", wrap=False)
    
def test_bare_closure():
    '''
    Test creation of bare anonymous function handle
    '''
    return mx.create_function_handle(closure="@(x)x+1", wrap=False)

@raises(ValueError)
def test_bare_func_notboth():
    '''
    Function handle may not be instantiated with both
    a name and a closure string.
    '''
    return mx.create_function_handle(name="sin",
                                     closure="@(x)x+1",
                                     wrap=False)
@raises(mx.MATLABError)
def test_bare_func_badsyntax():
    '''
    Closure must be valid MATLAB syntax.
    '''
    return mx.create_function_handle(closure="@(x)$$$", wrap=False)

@raises(ValueError)
def test_bare_func_notafunc():
    '''
    Closure must evaluate to a function handle
    '''
    return mx.create_function_handle(closure="42", wrap=False)

def test_wrap_bare_cell():
    '''
    Test that the wrapping function seems to work
    '''
    return mx.wrap_pycobject(bare_cell(()))

@raises(ValueError)
def test_wrap_wrongthing():
    '''
    Wrapping things that don't belong raises error
    '''
    return mx.wrap_pycobject(42)

############################################################
# Tests for member methods of mx.Array
# (NOT its subclasses - those go in mltypes/tests.py
############################################################

mxclasses = {'double' : mx.DOUBLE,
             'single' : mx.SINGLE,
             'int8' : mx.INT8,
             'uint8' : mx.UINT8,
             'int16' : mx.INT16,
             'uint16' : mx.UINT16,
             'int32' : mx.INT32,
             'uint32' : mx.UINT32,
             'int64' : mx.INT64,
             'uint64' : mx.UINT64,
             'char' : mx.CHAR,
             'logical' : mx.LOGICAL,
             'struct' : mx.STRUCT,
             'function_handle': mx.FUNCTION,
             'unknown' : mx.UNKNOWN,
             }

class Test_mxArray(object):
    def setUp(self):
        cobj = bare_array(())
        self.obj = mx.Array(mxpointer=cobj)
    def tearDown(self):
        del self.obj
    def test_mxclass(self):
        classname = self.obj._get_class_name()
        classid = self.obj._get_class_id()
        if classname in mxclasses:
            eq_(classid, mxclasses[classname])
        elif classid in mxclasses.values():
            raise TypeError, ("Class %s apparently has class ID %d, which "
                              "belongs to a builtin class") % (classname,
                                                               classid)
        else:
            raise RuntimeError, ("Something weird happened (%s, %d)" %
                                 (classname, classid))
    def test_str(self):
        str(self.obj)
    def test_repr(self):
        repr(self.obj)
    def test_bool(self):
        bool(self.obj)

        
class Test_Struct(Test_mxArray):
    def setUp(self):
        cobj = bare_struct((1,2))
        self.obj = mx.Array(mxpointer=cobj)
    def test_setfield(self):
        '''
        _set_field works in normal case
        '''
        self.obj._set_field(fieldname='foo',
                               value='bar',
                               index=0)
    def test_getfield(self):
        '''
        _get_field works in normal case
        '''
        self.test_setfield()
        val = self.obj._get_field(fieldname='foo',
                                  index=0)
        eq_(val, 'bar')
    def test_getempty(self):
        '''
        _get_field returns [] for uninitialized field
        '''
        self.test_setfield()
        val = self.obj._get_field(fieldname='foo',
                                     index=1)
        eq_(val._get_number_of_elements(),0)
    @raises(KeyError)
    def test_missingfield(self):
        '''
        _get_field raises KeyError for non-existent fields
        '''
        val = self.obj._get_field(fieldname='foo',
                                     index=0)
    @raises(IndexError)
    def test_getbadindex(self):
        '''
        _get_field checks upper index bound
        '''
        self.test_setfield()
        val = self.obj._get_field(fieldname='foo',
                                     index=2)
    @raises(IndexError)
    def test_getnegindex(self):
        '''
        _get_field checks lower index bound
        '''
        self.test_setfield()
        val = self.obj._get_field(fieldname='foo',
                                     index=-1)
    @raises(IndexError)
    def test_setbadindex(self):
        '''
        _set_field checks upper index bound
        '''
        self.obj._set_field(fieldname='foo',
                               value='bar',
                               index=2)
    @raises(IndexError)
    def test_setnegindex(self):
        '''
        _set_field checks lower index bound
        '''
        self.obj._set_field(fieldname='foo',
                               value='bar',
                               index=-1)
    @raises(TypeError)
    def test_setfieldnotstring(self):
        '''
        _set_field rejects non-string fieldnames
        '''
        self.obj._set_field(fieldname=42,
                               value='bar',
                               index=0)
    @raises(TypeError)
    def test_getfieldnotstring(self):
        '''
        _get_field rejects non-string fieldnames
        '''
        self.test_setfield()
        self.obj._get_field(fieldname=42,
                               index=0)
    def test_getfields(self):
        '''
        _get_fields lists the correct set of fieldnames
        '''
        infields = set(['foo', 'bar', 'spam', 'eggs'])
        for f in infields:
            self.obj._set_field(fieldname=f,
                                   value="contents of %s"%f,
                                   index=0)
        outfields = set(self.obj._get_fields())
        eq_(infields, outfields)
    def test_emptyfields(self):
        '''
        _get_fields acts appropriately for struct with no fields
        '''
        eq_(len(self.obj._get_fields()), 0)


