# License: BSD 3 clause

# import warnings
import os
import inspect
from datetime import datetime
from abc import ABCMeta, ABC, abstractmethod
from time import time
import json
import pydoc
import numpy as np
import numpydoc as nd
from numpydoc import docscrape
import copy
import pandas as pd
from scipy.linalg.special_matrices import toeplitz
from inspect import signature

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


import scipy
from scipy.stats import norm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from lifelines.utils import concordance_index
import statsmodels.api as sm
import pylab as pl
import warnings
warnings.filterwarnings('ignore')

class _ProxBinarsityDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxBinarsityFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxElasticNetDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxElasticNetFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxEqualityDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxEqualityFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxGroupL1Double:
    def __init__(self, *args, **kwargs):
        pass
class _ProxGroupL1Float:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL1Double:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL1Float:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL1wDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL1wFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL2Double:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL2Float:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL2sqDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxL2sqFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxMultiDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxMultiFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxSortedL1Double:
    def __init__(self, *args, **kwargs):
        pass
class _ProxSortedL1Float:
    def __init__(self, *args, **kwargs):
        pass
class _ProxPositiveDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxPositiveFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxSlopeDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxSlopeFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxTVDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxTVFloat:
    def __init__(self, *args, **kwargs):
        pass
class _ProxZeroDouble:
    def __init__(self, *args, **kwargs):
        pass
class _ProxZeroFloat:
    def __init__(self, *args, **kwargs):
        pass


class SVRG_VarianceReductionMethod_Last:
    def __init__(self, *args, **kwargs):
        pass
class SVRG_VarianceReductionMethod_Average:
    def __init__(self, *args, **kwargs):
        pass
class SVRG_VarianceReductionMethod_Random:
    def __init__(self, *args, **kwargs):
        pass

class SVRG_StepType_Fixed:
    def __init__(self, *args, **kwargs):
        pass
class SVRG_StepType_BarzilaiBorwein:
    def __init__(self, *args, **kwargs):
        pass


class _SVRGFloat:
    def __init__(self, *args, **kwargs):
        pass
class _SVRGDouble:
    def __init__(self, *args, **kwargs):
        pass
class _SDCAFloat:
    def __init__(self, *args, **kwargs):
        pass
class _SDCADouble:
    def __init__(self, *args, **kwargs):
        pass


variance_reduction_methods_mapper = {
    'last': SVRG_VarianceReductionMethod_Last,
    'avg': SVRG_VarianceReductionMethod_Average,
    'rand': SVRG_VarianceReductionMethod_Random
}
step_types_mapper = {
    'fixed': SVRG_StepType_Fixed,
    'bb': SVRG_StepType_BarzilaiBorwein
}
dtype_class_mapper = {
    np.dtype('float32'): _SVRGFloat,
    np.dtype('float64'): _SVRGDouble,
    np.dtype('float32'): _SDCAFloat,
    np.dtype('float64'): _SDCADouble
}

dtype_map = {
    np.dtype("float64"): _ProxBinarsityDouble,
    np.dtype("float32"): _ProxBinarsityFloat,
    np.dtype("float64"): _ProxElasticNetDouble,
    np.dtype("float32"): _ProxElasticNetFloat,
    np.dtype("float64"): _ProxEqualityDouble,
    np.dtype("float32"): _ProxEqualityFloat,
    np.dtype("float64"): _ProxGroupL1Double,
    np.dtype("float32"): _ProxGroupL1Float,
    np.dtype("float64"): _ProxL1Double,
    np.dtype("float32"): _ProxL1Float,
    np.dtype("float64"): _ProxL1wDouble,
    np.dtype("float32"): _ProxL1wFloat,
    np.dtype("float64"): _ProxL2Double,
    np.dtype("float32"): _ProxL2Float,
    np.dtype("float64"): _ProxL2sqDouble,
    np.dtype("float32"): _ProxL2sqFloat,
    np.dtype("float64"): _ProxMultiDouble,
    np.dtype("float32"): _ProxMultiFloat,
    np.dtype("float64"): _ProxSortedL1Double,
    np.dtype("float32"): _ProxSortedL1Float,
    np.dtype("float64"): _ProxPositiveDouble,
    np.dtype("float32"): _ProxPositiveFloat,
    np.dtype("float64"): _ProxSlopeDouble,
    np.dtype("float32"): _ProxSlopeFloat,
    np.dtype("float64"): _ProxTVDouble,
    np.dtype("float32"): _ProxTVFloat,
    np.dtype("float64"): _ProxZeroDouble,
    np.dtype("float32"): _ProxZeroFloat
}



# The metaclass inherits from ABCMeta and not type, since we'd like to
# do abstract classes in tick that inherits from ABC

# TODO: readonly attributes cannot be linked to c++ setters


class BaseMeta(ABCMeta):
    # Default behaviour of an attribute is writable, with no C++ setter
    default_attrinfo = {"writable": True, "cpp_setter": None}
    default_classinfo = {
        'is_prop': False,
        'in_doc': False,
        'doc': [],
        'in_init': False
    }

    @staticmethod
    def hidden_attr(attr_name):
        return '__' + attr_name

    @staticmethod
    def set_cpp_attribute(self, val, cpp_setter):
        """ Set the linked cpp attribute if possible

        Parameters
        ----------
        self : `object`
            the instance

        val : `object`
            the value to be set

        cpp_setter : `function`
            the function to use
        """
        # First we get the C++ object from its name (its name is an attribute
        # in the class)
        if not hasattr(self, "_cpp_obj_name"):
            raise NameError("_cpp_obj_name must be set as class attribute to "
                            "use automatic C++ setters")

        # Retrieve C++ associated object if it has been instantiated
        cpp_obj = None
        if hasattr(self, self._cpp_obj_name):
            cpp_obj = getattr(self, self._cpp_obj_name)

        # If the cpp_obj is instantiated, we update it
        if cpp_obj is not None:
            # Get the setter for this attribute in the C++
            if not hasattr(cpp_obj, cpp_setter):
                raise NameError("%s is not a method of %s" %
                                (cpp_setter, cpp_obj.__class__))
            cpp_obj_setter = getattr(cpp_obj, cpp_setter)
            cpp_obj_setter(val)

    @staticmethod
    def detect_if_called_in_init(self):
        """This function examine stacktrace in order to determine if it has
        been called from the __init__ function of the given instance

        Parameters
        ----------
        self : `object`
            The instance

        Returns
        -------
        set_int_init : `bool`
            True if this function was called by __init__
        """
        # It is forbidden to set a readonly (non writable) attribute
        # expect from __init__ function of the class
        set_in_init = False
        # trace contains information of the current execution
        # environment
        trace = inspect.currentframe()
        while trace is not None:
            # We retrieve the name of the executor (for example the
            # function that launched this command)
            exec_name = trace.f_code.co_name
            # We inspect the local variables
            if 'self' in trace.f_locals:
                local_self = trace.f_locals['self']
            else:
                local_self = None
            # We check if environment corresponds to our instance's
            # __init__
            if exec_name == '__init__' and local_self == self:
                set_in_init = True
                break
            # If this frame was not the good one, we try the previous
            # one, the one that has launched it
            # If there is no previous one, `None` will be returned
            trace = trace.f_back

        return set_in_init

    @staticmethod
    def build_property(class_name, attrs, attr_name, writable, cpp_setter):
        """
        Builds a property

        Parameters
        ----------
        class_name : `str`
            Name of the class

        attrs : `dict`
            The attributes of the class

        attr_name : `str`
            Name of the attribute for which we build a property

        writable : `bool`
            If True, we attribute can be changed by the user. If not,
            then an error will be raise when trying to change it.

        override : `bool`
            Not implemented yet

        cpp_setter : `str` or `None`
            Name of the setter in the c++ object embedded in the class
            for the attribute

        Returns
        -------
        output : property
            The required property
        """

        hidden_name = BaseMeta.hidden_attr(attr_name)

        def getter(self):
            # if it was not assigned yet we raise the correct error message
            if not hasattr(self, hidden_name):
                raise AttributeError("'%s' object has no attribute '%s'" %
                                     (class_name, attr_name))
            # Get the attribute
            return object.__getattribute__(self, hidden_name)

        def create_base_setter():
            if cpp_setter is None:
                # There is no C++ setter, we just set the attribute
                def setter(self, val):
                    object.__setattr__(self, hidden_name, val)
            else:
                # There is a C++ setter to apply
                def setter(self, val):
                    object.__setattr__(self, hidden_name, val)
                    # We update the C++ object embedded in the class
                    # as well.
                    BaseMeta.set_cpp_attribute(self, val, cpp_setter)

            return setter

        base_setter = create_base_setter()
        if writable:
            setter = base_setter
        else:
            # If it is not writable we wrap the base setter with something
            # that detect if attribute setting was called in __init__
            def setter(self, val):
                set_in_init = Base.detect_if_called_in_init(self)
                # If and only if this was launched from our instance's __init__
                # we allow user to set the attribute
                if set_in_init:
                    base_setter(self, val)
                else:
                    raise AttributeError(
                        "%s is readonly in %s" % (str(attr_name), class_name))

        def deletter(self):
            raise AttributeError(
                "can't delete %s in %s" % (str(attr_name), class_name))

        # We set doc to None otherwise it will interfere with
        # the docstring of the class.
        # This is very useful as we can have bugs with sphinx
        # otherwise (conflict between an attribute's docstring
        # and a docstring of a property with the same name).
        # All attributes are actually properties when the base
        # class is Base.
        # The docstring of all properties are then putted back
        # in the __init__ of the Base class below.
        prop = property(getter, setter, deletter, None)
        return prop

    @staticmethod
    def create_property_doc(class_name, attr_doc):
        """Create doc that will be attached to property

        Parameters
        ----------
        class_name : `str`
            Name of the class the property comes from

        attr_doc : `list`
            List output by numpydoc contained parsed documentation

        Returns
        -------
        The formatted doc
        """
        attr_type = attr_doc[1]
        attr_docstring = [
            line for line in attr_doc[2] if len(line.strip()) > 0
        ]
        attr_from = 'from %s' % class_name

        doc = [attr_type] + attr_docstring + [attr_from]
        return doc

    @staticmethod
    def find_init_params(attrs):
        """Find the parameters passed to the class's __init__
        """
        ignore = ['self', 'args', 'kwargs']

        # if class has no __init__ method
        if "__init__" not in attrs:
            return []

        return [
            key
            for key in inspect.signature(attrs["__init__"]).parameters.keys()
            if key not in ignore
        ]

    @staticmethod
    def find_properties(attrs):
        """Find all native properties of the class
        """
        return [
            attr_name for attr_name, value in attrs.items()
            if isinstance(value, property)
        ]

    @staticmethod
    def find_documented_attributes(class_name, attrs):
        """Parse the documentation to retrieve all attributes that have been
        documented and their documentation
        """
        # If a class is not documented we return an empty list
        if '__doc__' not in attrs:
            return []

        current_class_doc = inspect.cleandoc(attrs['__doc__'])
        parsed_doc = docscrape.ClassDoc(None, doc=current_class_doc)
        attr_docs = parsed_doc['Parameters'] + parsed_doc['Attributes'] + \
            parsed_doc['Other Parameters']

        attr_and_doc = []

        create_property_doc = BaseMeta.create_property_doc
        for attr_doc in attr_docs:
            attr_name = attr_doc[0]
            if ':' in attr_name:
                raise ValueError("Attribute '%s' has not a proper "
                                 "documentation, a space might be missing "
                                 "before colon" % attr_name)
            attr_and_doc += [(attr_name,
                              create_property_doc(class_name, attr_doc))]
        return attr_and_doc

    @staticmethod
    def extract_attrinfos(class_name, attrs):
        """Inspect class attrs to create aggregate all attributes info of the
        current class

        In practice, we inspect documented attributes, properties,
        parameters given to __init__ function and finally what user has
        filled in _attrinfos

        Parameters
        ----------
        class_name : `str`
            The name of the class (needed to create associated doc)

        atts : `dict`
            Dictionary of all futures attributes of the class

        Returns
        -------
        current_attrinfos : `dict`
            Subdict of the global classinfos dict concerning the attributes
            of the current class.
        """
        current_attrinfos = {}

        # First we look at all documented attributes
        for attr_name, attr_doc in \
                BaseMeta.find_documented_attributes(class_name, attrs):
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name]['in_doc'] = True
            current_attrinfos[attr_name]['doc'] = attr_doc

        # Second we look all native properties
        for attr_name in BaseMeta.find_properties(attrs):
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name]['is_prop'] = True

        # Third we look at parameters given to __init__
        for attr_name in BaseMeta.find_init_params(attrs):
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name]['in_init'] = True

        # Finally we use _attrinfos provided dictionary
        attrinfos = attrs.get("_attrinfos", {})
        for attr_name in attrinfos.keys():
            # Check that no unexpected key appears
            for key in attrinfos[attr_name].keys():
                if key not in BaseMeta.default_attrinfo:
                    raise ValueError('_attrinfos does not handle key %s' % key)
            # Add behavior specified in attrinfo
            current_attrinfos.setdefault(attr_name, {})
            current_attrinfos[attr_name].update(attrinfos[attr_name])

        return current_attrinfos

    @staticmethod
    def inherited_classinfos(bases):
        """Looks at all classinfos dictionary of bases class and merge them to
        create the initial classinfos dictionary

        Parameters
        ----------
        bases : `list`
            All the bases of the class

        Returns
        -------
        The initial classinfos dictionary

        Notes
        -----
        index corresponds to the distance in terms of inheritance.
        The highest index (in terms of inheritance) at
        which this class has been seen. We take the highest in
        case of multiple inheritance. If we have the following :
                   A0
                  / \
                 A1 B1
                 |  |
                 A2 |
                  \/
                  A3
        We want index of A0 to be higher than index of A1 which
        inherits from A0.
        In this example:
            * A3 has index 0
            * A2 and B1 have index 1
            * A1 has index 2
            * A0 has index 3 (even if it could be 2 through B1)
        """
        classinfos = {}
        for base in bases:
            if hasattr(base, "_classinfos"):
                for cls_key in base._classinfos:
                    base_infos = base._classinfos[cls_key]

                    if cls_key in classinfos:
                        current_info = classinfos[cls_key]
                        current_info['index'] = max(current_info['index'],
                                                    base_infos['index'] + 1)
                    else:
                        classinfos[cls_key] = {}
                        classinfos[cls_key]['index'] = base_infos['index'] + 1
                        classinfos[cls_key]['attr'] = base_infos['attr']

        return classinfos

    @staticmethod
    def create_attrinfos(classinfos):
        """Browse all class in classinfos dict to create a final attrinfo dict

        Parameters
        ----------
        classinfos : `dict`
            The final classinfos dict

        Returns
        -------
        attrinfos : `dict`
            Dictionary in which key is an attribute name and value is a dict
            with all its information.
        """
        # We sort the doc reversely by index, in order to see the
        # furthest classes first (and potentially override infos of
        # parents for an attribute if two classes document it)
        attrinfos = {}
        for cls_key, info_index in sorted(classinfos.items(),
                                          key=lambda item: item[1]['index'],
                                          reverse=True):
            classinfos = info_index['attr']

            for attr_name in classinfos:
                attrinfos.setdefault(attr_name, {})
                attrinfos[attr_name].update(classinfos[attr_name])

        return attrinfos

    def __new__(mcs, class_name, bases, attrs):

        # Initialize classinfos dictionnary with all classinfos dictionnary
        # of bases
        classinfos = BaseMeta.inherited_classinfos(bases)

        # Inspect current class to have get information about its atributes
        # cls_key is an unique hashable identifier for the class
        cls_key = '%s.%s' % (attrs['__module__'], attrs['__qualname__'])
        classinfos[cls_key] = {'index': 0}
        extract_attrinfos = BaseMeta.extract_attrinfos
        classinfos[cls_key]['attr'] = extract_attrinfos(class_name, attrs)

        # Once we have collected all classinfos we can extract from it all
        # attributes information
        attrinfos = BaseMeta.create_attrinfos(classinfos)

        attrs["_classinfos"] = classinfos
        attrs["_attrinfos"] = attrinfos

        build_property = BaseMeta.build_property

        # Create properties for all attributes described in attrinfos if they
        # are not already a property; This allow us to set a special behavior
        for attr_name, info in attrinfos.items():
            attr_is_property = attrinfos[attr_name].get('is_prop', False)

            # We create the corresponding property if our item is not a property
            if not attr_is_property:
                writable = info.get("writable",
                                    BaseMeta.default_attrinfo["writable"])

                cpp_setter = info.get("cpp_setter",
                                      BaseMeta.default_attrinfo["cpp_setter"])

                attrs[attr_name] = build_property(class_name, attrs, attr_name,
                                                  writable, cpp_setter)

        # Add a __setattr__ method that forbids to add an non-existing
        # attribute
        def __setattr__(self, key, val):
            if key in attrinfos:
                object.__setattr__(self, key, val)
            else:
                raise AttributeError("'%s' object has no settable attribute "
                                     "'%s'" % (class_name, key))

        attrs["__setattr__"] = __setattr__

        # Add a method allowing to force set an attribute
        def _set(self, key: str, val):
            """A method allowing to force set an attribute
            """
            if not isinstance(key, str):
                raise ValueError(
                    'In _set function you must pass key as string')

            if key not in attrinfos:
                raise AttributeError("'%s' object has no settable attribute "
                                     "'%s'" % (class_name, key))

            object.__setattr__(self, BaseMeta.hidden_attr(key), val)

            cpp_setter = self._attrinfos[key].get(
                "cpp_setter", BaseMeta.default_attrinfo["cpp_setter"])
            if cpp_setter is not None:
                BaseMeta.set_cpp_attribute(self, val, cpp_setter)

        attrs["_set"] = _set

        return ABCMeta.__new__(mcs, class_name, bases, attrs)

    def __init__(cls, class_name, bases, attrs):
        return ABCMeta.__init__(cls, class_name, bases, attrs)


class Base(metaclass=BaseMeta):
    """The BaseClass of the tick project. This relies on some dark
    magic based on a metaclass. The aim is to have read-only attributes,
    docstring for all parameters, and some other nasty features

    Attributes
    ----------
    name : str (read-only)
        Name of the class
    """

    _attrinfos = {
        "name": {
            "writable": False
        },
    }

    def __init__(self, *args, **kwargs):
        # We add the name of the class
        self._set("name", self.__class__.__name__)

        for attr_name, prop in self.__class__.__dict__.items():
            if isinstance(prop, property):
                if attr_name in self._attrinfos and len(
                        self._attrinfos[attr_name].get('doc', [])) > 0:
                    # we create the property documentation based o what we
                    # have found in the docstring.
                    # First we will have the type of the property, then the
                    # documentation and finally the closest class (in terms
                    # of inheritance) in which it is documented
                    # Note: We join doc with '-' instead of '\n'
                    # because multiline doc does not print well in iPython

                    prop_doc = self._attrinfos[attr_name]['doc']
                    prop_doc = ' - '.join([
                        str(d).strip() for d in prop_doc
                        if len(str(d).strip()) > 0
                    ])

                    # We copy property and add the doc found in docstring
                    setattr(
                        self.__class__, attr_name,
                        property(prop.fget, prop.fset, prop.fdel, prop_doc))

    @staticmethod
    def _get_now():
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    def _as_dict(self):
        dd = {}
        for key in self._attrinfos.keys():
            # private and protected attributes are not shown in the
            # dict
            if not key.startswith("_"):
                dd[key] = getattr(self, key)
        return dd

    def _inc_attr(self, key: str, step: int = 1):
        """Increment an attribute of the class by ``step``

        Parameters
        ----------
        key : `str`
            Name of the class's attribute

        step : `int`
            Size of the increase
        """
        self._set(key, getattr(self, key) + step)

    def __str__(self):
        dic = self._as_dict()
        if 'dtype' in dic and isinstance(dic['dtype'], np.dtype):
            dic['dtype'] = dic['dtype'].name
        return json.dumps(dic, sort_keys=True, indent=2)


def actual_kwargs(function):
    """
    Decorator that provides the wrapped function with an attribute
    'actual_kwargs'
    containing just those keyword arguments actually passed in to the function.

    References
    ----------
    http://stackoverflow.com/questions/1408818/getting-the-the-keyword
    -arguments-actually-passed-to-a-python-method

    Notes
    -----
    We override the signature of the decorated function to ensure it will be
    displayed correctly in sphinx
    """

    original_signature = signature(function)

    def inner(*args, **kwargs):
        inner.actual_kwargs = kwargs
        return function(*args, **kwargs)

    inner.__signature__ = original_signature

    return inner

def safe_array(X, dtype=np.float64):
    """Checks if the X has the correct type, dtype, and is contiguous.

    Parameters
    ----------
    X : `pd.DataFrame` or `np.ndarray` or `crs_matrix`
        The input data.

    dtype : np.dtype object or string
        Expected dtype of each X element.

    Returns
    -------
    output : `np.ndarray` or `csr_matrix`
        The input with right type, dtype.

    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(X, np.ndarray) and not X.flags['C_CONTIGUOUS']:
        warn(
            'Copying array of size %s to create a C-contiguous '
            'version of it' % str(X.shape), RuntimeWarning)
        X = np.ascontiguousarray(X)

    if X.dtype != dtype:
        warn(
            'Copying array of size %s to convert it in the right '
            'format' % str(X.shape), RuntimeWarning)
        X = X.astype(dtype)

    return X



def extract_dtype(dtype_or_object_with_dtype):
    import six
    if (isinstance(dtype_or_object_with_dtype, six.string_types)
            or isinstance(dtype_or_object_with_dtype, np.dtype)):
        return np.dtype(dtype_or_object_with_dtype)
    elif hasattr(dtype_or_object_with_dtype, 'dtype'):
        return np.dtype(dtype_or_object_with_dtype.dtype)
    else:
        raise ValueError("unsupported type used for prox creation, "
                         "expects dtype or class with dtype , type: {}".format(
                             dtype_or_object_with_dtype.__class__.__name__))


def get_typed_class(clazz, dtype_or_object_with_dtype, dtype_map):
    clazz.dtype = extract_dtype(dtype_or_object_with_dtype)
    if np.dtype(clazz.dtype) not in dtype_map:
        raise ValueError("dtype does not exist in type map for {}".format(
            clazz.__class__.__name__))
    return dtype_map[np.dtype(clazz.dtype)]


def copy_with(clazz, ignore_fields: list = None):
    """Copies clazz, temporarily sets values to None to avoid copying.
       not thread safe
    """
    from copy import deepcopy

    if not isinstance(clazz, Base):
        raise ValueError("Only objects inheriting from Base class should be"
                         "copied with copy_with.")

    fields = {}
    for field in ignore_fields:
        if hasattr(clazz, field) and getattr(clazz, field) is not None:
            fields[field] = getattr(clazz, field)
            clazz._set(field, None)

    new_clazz = deepcopy(clazz)

    for field in fields:
        clazz._set(field, fields[field])

    return new_clazz









LOSS = "loss"
GRAD = "grad"
LOSS_AND_GRAD = "loss_and_grad"
HESSIAN_NORM = "hessian_norm"

N_CALLS_LOSS = "n_calls_loss"
N_CALLS_GRAD = "n_calls_grad"
N_CALLS_LOSS_AND_GRAD = "n_calls_loss_and_grad"
N_CALLS_HESSIAN_NORM = "n_calls_hessian_norm"
PASS_OVER_DATA = "n_passes_over_data"


class Model(ABC, Base):
    """Abstract class for a model. It describes a zero-order model,
    namely only with the ability to compute a loss (goodness-of-fit
    criterion).

    Attributes
    ----------
    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_calls_loss : `int` (read-only)
        Number of times ``loss`` has been called so far

    n_passes_over_data : `int` (read-only)
        Number of effective passes through the data

    dtype : `{'float64', 'float32'}`
        Type of the data arrays used.

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    # A dict which specifies for each operation how many times we
    # pass through data
    pass_per_operation = {LOSS: 1}

    _attrinfos = {
        "_fitted": {
            "writable": False
        },
        N_CALLS_LOSS: {
            "writable": False
        },
        PASS_OVER_DATA: {
            "writable": False
        },
        "n_coeffs": {
            "writable": False
        },
        "_model": {
            "writable": False
        }
    }

    # The name of the attribute that might contain the C++ model object
    _cpp_obj_name = "_model"

    def __init__(self):
        Base.__init__(self)
        self._fitted = False
        self._model = None
        setattr(self, N_CALLS_LOSS, 0)
        setattr(self, PASS_OVER_DATA, 0)
        self.dtype = None

    def fit(self, *args):
        self._set_data(*args)
        self._set("_fitted", True)
        self._set(N_CALLS_LOSS, 0)
        self._set(PASS_OVER_DATA, 0)
        return self

    @abstractmethod
    def _get_n_coeffs(self) -> int:
        """An abstract method that forces childs to be able to give
        the number of parameters
        """
        pass

    @property
    def n_coeffs(self):
        if not self._fitted:
            raise ValueError(("call ``fit`` before using " "``n_coeffs``"))
        return self._get_n_coeffs()

    @abstractmethod
    def _set_data(self, *args):
        """Must be overloaded in child class. This method is called to
        fit data onto the gradient.
        Useful when pre-processing is necessary, etc...
        It should also set the dtype
        """
        pass

    def loss(self, coeffs: np.ndarray) -> float:
        """Computes the value of the goodness-of-fit at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            The loss is computed at this point

        Returns
        -------
        output : `float`
            The value of the loss

        Notes
        -----
        The ``fit`` method must be called to give data to the model,
        before using ``loss``. An error is raised otherwise.
        """
        # This is a bit of a hack as I don't see how to control the dtype of
        #  coeffs returning from scipy through lambdas
        if coeffs.dtype != self.dtype:
            warnings.warn(
                'coeffs vector of type {} has been cast to {}'.format(
                    coeffs.dtype, self.dtype))
            coeffs = coeffs.astype(self.dtype)

        if not self._fitted:
            raise ValueError("call ``fit`` before using ``loss``")
        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                ("``coeffs`` has size %i while the model" +
                 " expects %i coefficients") % (coeffs.shape[0], self.n_coeffs))
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[LOSS])

        return self._loss(coeffs)

    @abstractmethod
    def _loss(self, coeffs: np.ndarray) -> float:
        """Must be overloaded in child class
        """
        pass

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        """Deduce dtype and return true if C++ _model should be set
        """
        #import tick.base.dtype_to_cpp_type
        #return tick.base.dtype_to_cpp_type.get_typed_class(
        #    self, dtype_or_object_with_dtype, dtype_map)
        return get_typed_class(self, dtype_or_object_with_dtype, dtype_map)

    def astype(self, dtype_or_object_with_dtype):
        #import tick.base.dtype_to_cpp_type
        #new_model = tick.base.dtype_to_cpp_type.copy_with(
        #    self,
        #    ["_model"]  # ignore _model on deepcopy
        #)
        new_model = copy_with(
            self,
            ["_model"]  # ignore _model on deepcopy
        )
        new_model._set('_model',
                       new_model._build_cpp_model(dtype_or_object_with_dtype))
        return new_model

    def _build_cpp_model(self, dtype: str):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())


class ModelFirstOrder(Model):
    """An abstract class for models that implement a model with first
    order information, namely gradient information

    Attributes
    ----------
    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_calls_loss : `int` (read-only)
        Number of times ``loss`` has been called so far

    n_passes_over_data : `int` (read-only)
        Number of effective passes through the data

    n_calls_grad : `int` (read-only)
        Number of times ``grad`` has been called so far

    n_calls_loss_and_grad : `int` (read-only)
        Number of times ``loss_and_grad`` has been called so far

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """
    # A dict which specifies for each operation how many times we pass
    # through data
    pass_per_operation = {
        k: v
        for d in [Model.pass_per_operation, {
            GRAD: 1,
            LOSS_AND_GRAD: 2
        }] for k, v in d.items()
    }

    _attrinfos = {
        N_CALLS_GRAD: {
            "writable": False
        },
        N_CALLS_LOSS_AND_GRAD: {
            "writable": False
        },
    }

    def __init__(self):
        Model.__init__(self)
        setattr(self, N_CALLS_GRAD, 0)
        setattr(self, N_CALLS_LOSS_AND_GRAD, 0)

    def fit(self, *args):
        Model.fit(self, *args)
        self._set(N_CALLS_GRAD, 0)
        self._set(N_CALLS_LOSS_AND_GRAD, 0)
        return self

    def grad(self, coeffs: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """Computes the gradient of the model at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            Vector where gradient is computed

        out : `numpy.ndarray` or `None`
            If `None` a new vector containing the gradient is returned,
            otherwise, the result is saved in ``out`` and returned

        Returns
        -------
        output : `numpy.ndarray`
            The gradient of the model at ``coeffs``

        Notes
        -----
        The ``fit`` method must be called to give data to the model,
        before using ``grad``. An error is raised otherwise.
        """
        if coeffs.dtype != self.dtype:
            warnings.warn(
                'coeffs vector of type {} has been cast to {}'.format(
                    coeffs.dtype, self.dtype))
            coeffs = coeffs.astype(self.dtype)

        if not self._fitted:
            raise ValueError("call ``fit`` before using ``grad``")

        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                ("``coeffs`` has size %i while the model" +
                 " expects %i coefficients") % (len(coeffs), self.n_coeffs))

        if out is not None:
            grad = out
        else:
            grad = np.empty(self.n_coeffs, dtype=self.dtype)

        self._inc_attr(N_CALLS_GRAD)
        self._inc_attr(PASS_OVER_DATA, step=self.pass_per_operation[GRAD])

        self._grad(coeffs, out=grad)
        return grad

    @abstractmethod
    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        """Computes the gradient of the model at ``coeffs``
        The gradient must be stored in ``out``

        Notes
        -----
        Must be overloaded in child class
        """
        pass

    # TODO: better method annotation giving the type in the tuple
    def loss_and_grad(self, coeffs: np.ndarray,
                      out: np.ndarray = None) -> tuple:
        """Computes the value and the gradient of the function at
        ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`
            Vector where the loss and gradient are computed

        out : `numpy.ndarray` or `None`
            If `None` a new vector containing the gradient is returned,
            otherwise, the result is saved in ``out`` and returned

        Returns
        -------
        loss : `float`
            The value of the loss

        grad : `numpy.ndarray`
            The gradient of the model at ``coeffs``

        Notes
        -----
        The ``fit`` method must be called to give data to the model,
        before using ``loss_and_grad``. An error is raised otherwise.

        """
        if not self._fitted:
            raise ValueError("call ``fit`` before using " "``loss_and_grad``")

        if coeffs.shape[0] != self.n_coeffs:
            raise ValueError(
                ("``coeffs`` has size %i while the model" +
                 "expects %i coefficients") % (len(coeffs), self.n_coeffs))
        if out is not None:
            grad = out
        else:
            grad = np.empty(self.n_coeffs, dtype=self.dtype)

        self._inc_attr(N_CALLS_LOSS_AND_GRAD)
        self._inc_attr(N_CALLS_LOSS)
        self._inc_attr(N_CALLS_GRAD)
        self._inc_attr(PASS_OVER_DATA,
                       step=self.pass_per_operation[LOSS_AND_GRAD])
        loss = self._loss_and_grad(coeffs, out=grad)
        return loss, grad

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray) -> float:
        self._grad(coeffs, out=out)
        return self._loss(coeffs)

deep_copy_ignore_fields = ["_prox"]

class Prox(ABC, Base):
    """An abstract base class for a proximal operator

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {"_prox": {"writable": False}, "_range": {"writable": False}}

    # The name of the attribute that will contain the C++ prox object
    _cpp_obj_name = "_prox"

    def __init__(self, range: tuple = None):
        Base.__init__(self)
        self._range = None
        self._prox = None
        self.range = range
        self.dtype = None

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, val):
        if val is not None:
            if len(val) != 2:
                raise ValueError("``range`` must be a tuple with 2 "
                                 "elements")
            if val[0] >= val[1]:
                raise ValueError("first element must be smaller than "
                                 "second element in ``range``")
            self._set("_range", val)
            _prox = self._prox
            if _prox is not None:
                _prox.set_start_end(val[0], val[1])

    def call(self, coeffs, step=1., out=None):
        """Apply proximal operator on a vector.
        It computes:

        .. math::
            argmin_x \\big( f(x) + \\frac{1}{2} \|x - v\|_2^2 \\big)

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            Input vector on which is applied the proximal operator

        step : `float` or `np.array`, default=1.
            The amount of penalization is multiplied by this amount

            * If `float`, the amount of penalization is multiplied by
              this amount
            * If `np.array`, then each coordinate of coeffs (within
              the given range), receives an amount of penalization
              multiplied by t (available only for separable prox)

        out : `numpy.ndarray`, shape=(n_params,), default=None
            If not `None`, the output is stored in the given ``out``.
            Otherwise, a new vector is created.

        Returns
        -------
        output : `numpy.ndarray`, shape=(n_coeffs,)
            Same object as out

        Notes
        -----
        ``step`` must have the same size as ``coeffs`` whenever range is
        `None`, or a size matching the one given by the range
        otherwise
        """
        if out is None:
            # We don't have an output vector, we create a fresh copy
            out = coeffs.copy()
        else:
            # We do an inplace copy of coeffs into out
            out[:] = coeffs
        # Apply the proximal, the output is in out
        self._call(coeffs, step, out)
        return out

    @abstractmethod
    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        pass

    @abstractmethod
    def value(self, coeffs: np.ndarray) -> float:
        pass

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        #import tick.base.dtype_to_cpp_type
        #return tick.base.dtype_to_cpp_type.get_typed_class(
        #    self, dtype_or_object_with_dtype, dtype_map)
        return get_typed_class(self, dtype_or_object_with_dtype, dtype_map)

    def _extract_dtype(self, dtype_or_object_with_dtype):
        #import tick.base.dtype_to_cpp_type
        #return tick.base.dtype_to_cpp_type.extract_dtype(
        #    dtype_or_object_with_dtype)
        return extract_dtype(dtype_or_object_with_dtype)

    def astype(self, dtype_or_object_with_dtype):
        #import tick.base.dtype_to_cpp_type
        #new_prox = tick.base.dtype_to_cpp_type.copy_with(
        #    self,
        #    ["_prox"]  # ignore _prox on deepcopy
        #)
        new_prox = copy_with(
            self,
            ["_prox"]  # ignore _prox on deepcopy
        )
        new_prox._set('_prox',
                      new_prox._build_cpp_prox(dtype_or_object_with_dtype))
        return new_prox

    def _build_cpp_prox(self, dtype):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())

class ProxZero(Prox):
    """Proximal operator of the null function (identity)

    Parameters
    ----------
    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.

    Notes
    -----
    Using ``ProxZero`` means no penalization is applied on the model.
    """

    def __init__(self, range: tuple = None):
        Prox.__init__(self, range)
        self._prox = self._build_cpp_prox("float64")

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)
        if self.range is None:
            return prox_class(0.)
        else:
            return prox_class(0., self.range[0], self.range[1])


class ProxL1(Prox):
    """Proximal operator of the L1 norm (soft-thresholding)

    Parameters
    ----------
    strength : `float`
        Level of L1 penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L1 penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self._prox = self._build_cpp_prox("float64")

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)
        if self.range is None:
            return prox_class(self.strength, self.positive)
        else:
            return prox_class(self.strength, self.range[0], self.range[1],
                              self.positive)

class ProxL2Sq(Prox):
    """Proximal operator of the squared L2 norm (ridge penalization)

    Parameters
    ----------
    strength : `float`, default=0.
        Level of L2 penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L2 penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self._prox = self._build_cpp_prox("float64")

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """

        return self._prox.value(coeffs)

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)
        if self.range is None:
            return prox_class(self.strength, self.positive)
        else:
            return prox_class(self.strength, self.range[0], self.range[1],
                              self.positive)


class ProxElasticNet(Prox):
    """
    Proximal operator of the ElasticNet regularization.

    Parameters
    ----------
    strength : `float`
        Level of ElasticNet regularization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    ratio : `float`, default=0
        The ElasticNet mixing parameter, with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.

    positive : `bool`, default=`False`
        If True, apply the penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "ratio": {
            "writable": True,
            "cpp_setter": "set_ratio"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, ratio: float, range: tuple = None,
                 positive=False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self.ratio = ratio
        self._prox = self._build_cpp_prox("float64")

    def _call(self, coeffs: np.ndarray, step: object, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)
        print(f"prox_class: {prox_class}")  # Add this for debugging
        if isinstance(prox_class, str):
            raise TypeError(f"prox_class is a string: {prox_class}, expected callable object")
 
        if self.range is None:
            return prox_class(self.strength, self.ratio, self.positive)
        else:
            return prox_class(self.strength, self.ratio, self.range[0],
                              self.range[1], self.positive)


class ProxTV(Prox):
    """Proximal operator of the total-variation penalization

    Parameters
    ----------
    strength : `float`
        Level of total-variation penalization

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply L1 penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`
        Type of the arrays used.

    Notes
    -----
    Uses the fast-TV algorithm described in:

    * "A Direct Algorithm for 1D Total Variation Denoising"
      by Laurent Condat, *Ieee Signal Proc. Letters*
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        }
    }

    def __init__(self, strength: float, range: tuple = None,
                 positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self._prox = self._build_cpp_prox("float64")

    def _call(self, coeffs: np.ndarray, step: float, out: np.ndarray):
        self._prox.call(coeffs, step, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)
        if self.range is None:
            return prox_class(self.strength, self.positive)
        else:
            return prox_class(self.strength, self.range[0], self.range[1],
                              self.positive)

class ProxWithGroups(Prox):
    """Base class of a proximal operator with groups. It applies specific
    proximal operator in each group, or block. Blocks (non-overlapping) are
    specified by the ``blocks_start`` and ``blocks_length`` parameters.
    This base class is not intented for end-users, but for developers only.

    Parameters
    ----------
    strength : `float`
        Level of penalization

    blocks_start : `list` or `numpy.array`, shape=(n_blocks,)
        First entry of each block

    blocks_length : `list` or `numpy.array`, shape=(n_blocks,)
        Size of each block

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply the penalization together with a projection
        onto the set of vectors with non-negative entries

    Attributes
    ----------
    n_blocks : `int`
        Number of blocks
    """

    _attrinfos = {
        "strength": {
            "writable": True,
            "cpp_setter": "set_strength"
        },
        "positive": {
            "writable": True,
            "cpp_setter": "set_positive"
        },
        "blocks_start": {
            "writable": True,
            "cpp_setter": "set_blocks_start"
        },
        "blocks_length": {
            "writable": True,
            "cpp_setter": "set_blocks_length"
        }
    }

    def __init__(self, strength: float, blocks_start, blocks_length,
                 range: tuple = None, positive: bool = False):
        Prox.__init__(self, range)

        if any(length <= 0 for length in blocks_length):
            raise ValueError("all blocks must be of positive size")
        if any(start < 0 for start in blocks_start):
            raise ValueError("all blocks must have positive starting indices")

        if type(blocks_start) is list:
            blocks_start = np.array(blocks_start, dtype=np.uint64)
        if type(blocks_length) is list:
            blocks_length = np.array(blocks_length, dtype=np.uint64)

        if blocks_start.dtype is not np.uint64:
            blocks_start = blocks_start.astype(np.uint64)
        if blocks_length.dtype is not np.uint64:
            blocks_length = blocks_length.astype(np.uint64)

        if blocks_start.shape != blocks_length.shape:
            raise ValueError("``blocks_start`` and ``blocks_length`` "
                             "must have the same size")
        if np.any(blocks_start[1:] < blocks_start[:-1]):
            raise ValueError('``block_start`` must be sorted')
        if np.any(blocks_start[1:] < blocks_start[:-1] + blocks_length[:-1]):
            raise ValueError("blocks must not overlap")

        self.strength = strength
        self.positive = positive
        self.blocks_start = blocks_start
        self.blocks_length = blocks_length

        # Get the C++ prox class, given by an overloaded method
        self._prox = self._build_cpp_prox("float64")

    @property
    def n_blocks(self):
        return self.blocks_start.shape[0]

    def _call(self, coeffs: np.ndarray, t: float, out: np.ndarray):
        self._prox.call(coeffs, t, out)

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.array`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        return self._prox.value(coeffs)

    def _as_dict(self):
        dd = Prox._as_dict(self)
        del dd["blocks_start"]
        del dd["blocks_length"]
        return dd

    def _build_cpp_prox(self, dtype_or_object_with_dtype):
        dtype_map = self._get_dtype_map()
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        prox_class = self._get_typed_class(dtype_or_object_with_dtype,
                                           dtype_map)

        if self.range is None:
            return prox_class(self.strength, self.blocks_start,
                              self.blocks_length, self.positive)
        else:
            start, end = self.range
            i_max = self.blocks_start.argmax()
            if end - start < self.blocks_start[i_max] + self.blocks_length[i_max]:
                raise ValueError("last block is not within the range "
                                 "[0, end-start)")
            return prox_class(self.strength, self.blocks_start,
                              self.blocks_length, start, end, self.positive)

    def _get_dtype_map(self):
        raise ValueError("""This function is expected to
                            overriden in a subclass""".strip())


class ProxBinarsity(ProxWithGroups):
    """Proximal operator of binarsity. It is simply a succession of two steps on
    different intervals: ``ProxTV`` plus a centering translation. More
    precisely, total-variation regularization is applied on a coefficient vector
    being a concatenation of multiple coefficient vectors corresponding to
    blocks, followed by centering within sub-blocks. Blocks (non-overlapping)
    are specified by the ``blocks_start`` and ``blocks_length`` parameters.

    Parameters
    ----------
    strength : `float`
        Level of total-variation penalization

    blocks_start : `np.array`, shape=(n_blocks,)
        First entry of each block

    blocks_length : `np.array`, shape=(n_blocks,)
        Size of each block

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply in the end a projection onto the set of vectors with
        non-negative entries

    Attributes
    ----------
    n_blocks : `int`
        Number of blocks

    dtype : `{'float64', 'float32'}`
        Type of the arrays used.

    References
    ----------
    ProxBinarsity uses the fast-TV algorithm described in:

    Condat, L. (2012).
    `A Direct Algorithm for 1D Total Variation Denoising`_.

    .. _A Direct Algorithm for 1D Total Variation Denoising: https://hal.archives-ouvertes.fr/hal-00675043v2/document
    """

    def __init__(self, strength: float, blocks_start, blocks_length,
                 range: tuple = None, positive: bool = False):
        ProxWithGroups.__init__(self, strength, blocks_start, blocks_length,
                                range, positive)
        self._prox = self._build_cpp_prox("float64")

    def _get_dtype_map(self):
        return dtype_map



from numpy.linalg import norm
from collections import defaultdict
import dill

def spars_func(coeffs, **kwargs):
    eps = np.finfo(coeffs.dtype).eps
    return np.sum(np.abs(coeffs) > eps, axis=None)


class History(Base):
    """A class to manage the history along iterations of a solver

    Attributes
    ----------
    print_order : `list` or `str`
        The list of values to print along iterations

    values : `dict`
        A `dict` containing the history. Key is the value name and
        values are the values taken along the iterations

    last_values : `dict`
        A `dict` containing all the last history values

    _minimum_col_width : `int`
        Minimal size of a column when printing the history

    _minimizer : `None` or `numpy.ndarray`
        The minimizer of the objective. `None` if not specified.
        This is useful to compute a distance to the optimum.

    _minimum : `None` or `float`
        The minimal (optimal) value of the objective. `None` if not
        specified. This is useful to compute a distance to the optimum.

    _print_style : `list` or `str`
        The display style of all printed numbers

    _history_func : `dict`
        A dict given for all values the function to be applied before
        saving and displaying in history. This is useful for computing
        the sparsity, the rank, among other things, of the iterates
        along iterations of the solver

    _n_iter : `int`
        The current iteration number

    _col_widths : `list` or `int`
        A list containing the computed width of each column used for
        printing the history, based on the name length of the column
    """

    _attrinfos = {
        "values": {
            "writable": False
        },
        "last_values": {
            "writable": False
        },
    }

    def __init__(self):
        Base.__init__(self)
        self._minimum_col_width = 9
        self.print_order = ["n_iter", "obj", "step", "rel_obj"]
        # Instantiate values of the history
        self._clear()

        self._minimizer = None
        self._minimum = None
        self._set("values", None)
        self._col_widths = None
        self._n_iter = None

        # History function to compute history values based on parameters
        # used in a solver
        history_func = {}
        self._history_func = history_func

        # Default print style of history values. Default is %.2e
        print_style = defaultdict(lambda: "%.2e")
        print_style["n_iter"] = "%d"
        print_style["n_epoch"] = "%d"
        print_style["n_inner_prod"] = "%d"
        print_style["spars"] = "%d"
        print_style["rank"] = "%d"
        self._print_style = print_style

    def _clear(self):
        """Reset history values"""
        self._set("values", defaultdict(list))

    def _update(self, **kwargs):
        """Update the history along the iterations.

        For each keyword argument, we apply the history function corresponding
        to this keyword, and use its results in the history
        """
        self._n_iter = kwargs["n_iter"]
        history_func = self._history_func
        # We loop on both, history functions and kerword arguments
        keys = set(kwargs.keys()).union(set(history_func.keys()))
        for key in keys:
            # Either it has a corresponding history function which we
            # apply on all keywords
            if key in history_func:
                func = history_func[key]
                self.values[key].append(func(**kwargs))
            # Either we only record the value
            else:
                value = kwargs[key]
                self.values[key].append(value)

    def _format(self, name, index):
        try:
            formatted_str = self._print_style[name] % \
                            self.values[name][index]
        except TypeError:
            formatted_str = str(self.values[name][index])
        return formatted_str

    def _print_header(self):
        min_width = self._minimum_col_width
        line = ' | '.join(
            list([
                name.center(min_width) for name in self.print_order
                if name in self.values
            ]))
        names = [name.center(min_width) for name in self.print_order]
        self._col_widths = list(map(len, names))
        print(line)

    def _print_line(self, index):
        line = ' | '.join(
            list([
                self._format(name, index).rjust(self._col_widths[i])
                for i, name in enumerate(self.print_order)
                if name in self.values
            ]))
        print(line)

    def _print_history(self):
        """Verbose the current line of history
        """
        # If this is the first iteration, plot the history's column
        # names
        if self._col_widths is None:
            self._print_header()

        self._print_line(-1)

    def print_full_history(self):
        """Verbose the whole history
        """
        self._print_header()
        n_lines = len(next(iter(self.values.values())))

        for i in range(n_lines):
            self._print_line(i)

    @property
    def last_values(self):
        last_values = {}
        for key, hist in self.values.items():
            last_values[key] = hist[-1]
        return last_values

    def set_minimizer(self, minimizer: np.ndarray):
        """Set the minimizer of the objective, to compute distance
        to it along iterations

        Parameters
        ----------
        minimizer : `numpy.ndarray`, shape=(n_coeffs,)
            The minimizer of the objective

        Notes
        -----
        This adds dist_coeffs in history (distance to the minimizer)
        which is printed along iterations
        """
        self._minimizer = minimizer.copy()
        self._history_func["dist_coeffs"] = \
            lambda x, **kwargs: norm(x - self._minimizer)
        print_order = self.print_order
        if "dist_coeffs" not in print_order:
            print_order.append("dist_coeffs")

    def set_minimum(self, minimum: float):
        """Set the minimum of the objective, to compute distance to the
        optimum along iterations

        Parameters
        ----------
        minimum : `float`
            The minimizer of the objective

        Notes
        -----
        This adds dist_obj in history (distance to the minimum) which
        is printed along iterations
        """
        self._minimum = minimum
        self._history_func["dist_obj"] = \
            lambda obj, **kwargs: obj - self._minimum
        print_order = self.print_order
        if "dist_obj" not in print_order:
            print_order.append("dist_obj")

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("values", None)
        return dd

    # We use dill for serialization because history uses lambda functions
    def __getstate__(self):
        return dill.dumps(self.__dict__)

    def __setstate__(self, state):
        object.__setattr__(self, '__dict__', dill.loads(state))


class Solver(Base):
    """
    The base class for a solver. In only deals with verbosing
    information, creating an History object, etc.

    Parameters
    ----------
    tol : `float`, default=0
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default = 10
        Print history information every time the iteration number is a
        multiple of ``print_every``

    record_every : `int`, default = 1
        Information along iteration is recorded in history each time the
        iteration number of a multiple of ``record_every``

    Attributes
    ----------
    time_start : `str`
        Start date of the call to solve()

    time_elapsed : `float`
        Duration of the call to solve(), in seconds

    time_end : `str`
        End date of the call to solve()

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {
        "history": {
            "writable": False
        },
        "solution": {
            "writable": False
        },
        "time_start": {
            "writable": False
        },
        "time_elapsed": {
            "writable": False
        },
        "time_end": {
            "writable": False
        },
        "_time_start": {
            "writable": False
        },
        "_record_every": { }
    }

    def __init__(self, tol=0., max_iter=100, verbose=True, print_every=10,
                 record_every=1):
        Base.__init__(self)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.print_every = print_every
        self.record_every = record_every
        # Create an history object which deals with printing information
        # along the optimization loop, and stores information
        self.history = History()
        self.time_start = None
        self._time_start = None
        self.time_elapsed = None
        self.time_end = None
        self.solution = None

    def _start_solve(self):
        # Reset history
        self.history._clear()
        self._set("time_start", self._get_now())
        self._set("_time_start", time())
        if self.verbose:
            print("Launching the solver " + self.name + "...")

    def _end_solve(self):
        t = time()
        self._set("time_elapsed", t - self._time_start)
        if self.verbose:
            print("Done solving using " + self.name + " in " +
                  str(self.time_elapsed) + " seconds")

    def solve(self, *args, **kwargs):
        self._start_solve()
        self._solve(*args, **kwargs)
        self._end_solve()
        return self.solution

    def _should_record_iter(self, n_iter):
        """Should solver record this iteration or not?
        """
        # If we are never supposed to record
        if self.max_iter < self.print_every and \
                self.max_iter < self.record_every:
            return False
        # Otherwise check that we are either at a specific moment or at the end
        elif n_iter % self.print_every == 0 or n_iter % self.record_every == 0:
            return True
        elif n_iter + 1 == self.max_iter:
            return True
        return False

    def _handle_history(self, n_iter: int, force: bool = False, **kwargs):
        """Handles history for keywords and current iteration

        Parameters
        ----------
        n_iter : `int`
            The current iteration (will determine if we record it or
            not)
        force : `bool`
            If True, we will record no matter the value of ``n_iter``

        **kwargs : `dict`
            key, value pairs of the values to record in the History of
            the solver
        """

        # TODO: this should be protected : _handle_history
        verbose = self.verbose
        print_every = self.print_every
        record_every = self.record_every
        should_print = verbose and (force or n_iter % print_every == 0)
        should_record = force or n_iter % print_every == 0 or \
                        n_iter % record_every == 0
        if should_record:
            iter_time = kwargs.get('iter_time', time() - self._time_start)
            self.history._update(n_iter=n_iter, time=iter_time,
                                 **kwargs)
        if should_print:
            self.history._print_history()

    def print_history(self):
        self.history.print_full_history()

    @abstractmethod
    def _solve(self, *args, **kwargs):
        """Method to be overloaded of the child solver
        """
        pass

    @abstractmethod
    def objective(self, coeffs, loss: float = None):
        """Compute the objective minimized by the solver at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The objective is computed at this point

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given ``coeffs``
        """

    def get_history(self, key=None):
        """Returns history of the solver

        Parameters
        ----------
        key : `str`, default=None
            * If `None` all history is returned as a `dict`
            * If `str`, name of the history element to retrieve

        Returns
        -------
        output : `list` or `dict`
            * If ``key`` is None or ``key`` is not in history then
              output is a dict containing history of all keys
            * If ``key`` is the name of an element in the history,
              output is a `list` containing the history of this element
        """
        val = self.history.values.get(key, None)
        if val is None:
            return self.history.values
        else:
            return val

    @property
    def record_every(self):
        return self._record_every

    @record_every.setter
    def record_every(self, val):
        self._record_every = val

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("coeffs", None)
        dd.pop("history", None)
        return dd


class SolverFirstOrder(Solver):
    """The base class for a first order solver. It defines methods for
    setting a model (giving first order information) and a proximal
    operator

    In only deals with verbosing information, and setting parameters

    Parameters
    ----------
    step : `float` default=None
        Step-size of the algorithm

    tol : `float`, default=0
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default = 10
        Print history information every time the iteration number is a
        multiple of ``print_every``

    record_every : `int`, default = 1
        Information along iteration is recorded in history each time the
        iteration number of a multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model to solve

    prox : `Prox`
        Proximal operator to solve

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {
        "model": {
            "writable": False
        },
        "prox": {
            "writable": False
        },
        "_initial_n_calls_loss_and_grad": {
            "writable": False
        },
        "_initial_n_calls_loss": {
            "writable": False
        },
        "_initial_n_calls_grad": {
            "writable": False
        },
        "_initial_n_passes_over_data": {
            "writable": False
        },
    }

    def __init__(self, step: float = None, tol: float = 0.,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):

        self.dtype = None

        Solver.__init__(self, tol, max_iter, verbose, print_every,
                        record_every)
        self.model = None
        self.prox = None
        self.step = step
        # Martin's complicated and useless stuff :)
        self._initial_n_calls_loss_and_grad = 0
        self._initial_n_calls_loss = 0
        self._initial_n_calls_grad = 0
        self._initial_n_passes_over_data = 0

    def validate_model(self, model: Model):
        if not isinstance(model, Model):
            raise ValueError('Passed object of class %s is not a '
                             'Model class' % model.name)
        if not model._fitted:
            raise ValueError('Passed object %s has not been fitted. You must '
                             'call ``fit`` on it before passing it to '
                             '``set_model``' % model.name)

    def set_model(self, model: Model):
        """Set model in the solver

        Parameters
        ----------
        model : `Model`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things)

        Returns
        -------
        output : `Solver`
            The same instance with given model
        """
        self.validate_model(model)
        self.dtype = model.dtype
        self._set("model", model)
        return self

    def _initialize_values(self, x0: np.ndarray = None, step: float = None,
                           n_empty_vectors: int = 0):
        """Initialize values

        Parameters
        ----------
        x0 : `numpy.ndarray`
            Starting point

        step : `float`
            Initial step

        n_empty_vectors : `int`
            Number of empty vector of like x0 needed

        Returns
        -------
        step : `float`
            Initial step

        obj : `float`
            Initial value of objective function

        iterate : `numpy.ndarray`
            copy of starting point

        empty vectors : `numpy.ndarray`
            n_empty_vectors empty vectors shaped as x0. For example, those
            vectors can be used to store previous iterate values during
            a solver execution.
        """
        # Initialization
        if step is None:
            if self.step is None:
                raise ValueError("No step specified.")
            else:
                step = self.step
        else:
            self.step = step
        if x0 is None:
            x0 = np.zeros(self.model.n_coeffs, dtype=self.dtype)
        iterate = x0.copy()
        obj = self.objective(iterate)

        result = [step, obj, iterate]
        for _ in range(n_empty_vectors):
            result.append(np.zeros_like(x0))

        return tuple(result)

    def set_prox(self, prox: Prox):
        """Set proximal operator in the solver

        Parameters
        ----------
        prox : `Prox`
            The proximal operator of the penalization function

        Returns
        -------
        output : `Solver`
            The solver with given prox

        Notes
        -----
        In some solvers, ``set_model`` must be called before
        ``set_prox``, otherwise and error might be raised
        """
        if not isinstance(prox, Prox):
            raise ValueError('Passed object of class %s is not a '
                             'Prox class' % prox.name)
        if self.dtype is None or self.model is None:
            raise ValueError("Solver must call set_model before set_prox")
        if prox.dtype != self.dtype:
            prox = prox.astype(self.dtype)
        self._set("prox", prox)
        return self

    def astype(self, dtype_or_object_with_dtype):
        if self.model is None:
            raise ValueError("Cannot reassign solver without a model")

        #import tick.base.dtype_to_cpp_type
        #new_solver = tick.base.dtype_to_cpp_type.copy_with(
        #    self,
        #    ["prox", "model"]  # ignore on deepcopy
        #)
        #new_solver.dtype = tick.base.dtype_to_cpp_type.extract_dtype(
        #    dtype_or_object_with_dtype)
        new_solver = copy_with(
            self,
            ["prox", "model"]  # ignore on deepcopy
        )
        new_solver.dtype = extract_dtype(
            dtype_or_object_with_dtype)
        new_solver.set_model(self.model.astype(new_solver.dtype))
        if self.prox is not None:
            new_solver.set_prox(self.prox.astype(new_solver.dtype))
        return new_solver

    def _as_dict(self):
        dd = Solver._as_dict(self)
        if self.model is not None:
            dd["model"] = self.model._as_dict()
        if self.prox is not None:
            dd["prox"] = self.prox._as_dict()
        return dd

    def objective(self, coeffs, loss: float = None):
        """Compute the objective function

        Parameters
        ----------
        coeffs : `np.array`, shape=(n_coeffs,)
            Point where the objective is computed

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given ``coeffs``
        """
        if self.prox is None:
            prox_value = 0
        else:
            prox_value = self.prox.value(coeffs)

        if loss is None:
            return self.model.loss(coeffs) + prox_value
        else:
            return loss + prox_value

    def solve(self, x0=None, step=None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : `np.array`, shape=(n_coeffs,), default=`None`
            Starting point of the solver

        step : `float`, default=`None`
            Step-size or learning rate for the solver. This can be tuned also
            using the ``step`` attribute

        Returns
        -------
        output : `np.array`, shape=(n_coeffs,)
            Obtained minimizer for the problem, same as ``solution`` attribute
        """
        if x0 is not None and self.dtype is not "float64":
            x0 = x0.astype(self.dtype)

        if self.model is None:
            raise ValueError('You must first set the model using '
                             '``set_model``.')
        if self.prox is None:
            raise ValueError('You must first set the prox using '
                             '``set_prox``.')
        solution = Solver.solve(self, x0, step)
        return solution

    def _handle_history(self, n_iter: int, force: bool = False, **kwargs):
        """Updates the history of the solver.

        Parameters
        ----------

        Notes
        -----
        This should not be used by end-users.
        """
        # self.model.n_calls_loss_and_grad is shared by all
        # solvers using this model
        # hence it might not be at 0 while starting
        # /!\ beware if parallel computing...
        if n_iter == 1:
            self._set("_initial_n_calls_loss_and_grad",
                      self.model.n_calls_loss_and_grad)
            self._set("_initial_n_calls_loss", self.model.n_calls_loss)
            self._set("_initial_n_calls_grad", self.model.n_calls_grad)
            self._set("_initial_n_passes_over_data",
                      self.model.n_passes_over_data)
        n_calls_loss_and_grad = \
            self.model.n_calls_loss_and_grad - \
            self._initial_n_calls_loss_and_grad
        n_calls_loss = \
            self.model.n_calls_loss - self._initial_n_calls_loss
        n_calls_grad = \
            self.model.n_calls_grad - self._initial_n_calls_grad
        n_passes_over_data = \
            self.model.n_passes_over_data - \
            self._initial_n_passes_over_data
        Solver.\
            _handle_history(self, n_iter, force=force,
                            n_calls_loss_and_grad=n_calls_loss_and_grad,
                            n_calls_loss=n_calls_loss,
                            n_calls_grad=n_calls_grad,
                            n_passes_over_data=n_passes_over_data,
                            **kwargs)


def relative_distance(new_vector, old_vector, use_norm=None):
    """Computes the relative error with respect to some norm
    It is useful to evaluate relative change of a vector

    Parameters
    ----------
    new_vector : `np.ndarray`
        New value of the vector

    old_vector : `np.ndarray`
        old value of the vector to compare with

    use_norm : `int` or `str`
        The norm to use among those proposed by :func:`.np.linalg.norm`

    Returns
    -------
    output : `float`
        Relative distance
    """
    norm_old_vector = norm(old_vector, use_norm)
    if norm_old_vector == 0:
        norm_old_vector = 1.
    return norm(new_vector - old_vector, use_norm) / norm_old_vector


class AGD(SolverFirstOrder):
    """Accelerated proximal gradient descent

    For the minimization of objectives of the form

    .. math::
        f(w) + g(w),

    where :math:`f` has a smooth gradient and :math:`g` is prox-capable.
    Function :math:`f` corresponds to the ``model.loss`` method of the model
    (passed with ``set_model`` to the solver) and :math:`g` corresponds to
    the ``prox.value`` method of the prox (passed with the ``set_prox`` method).
    One iteration of :class:`AGD <tick.solver.AGD>` is as follows:

    .. math::
        w^{k} &\\gets \\mathrm{prox}_{\\eta g} \\big(z^k - \\eta \\nabla f(z^k)
        \\big) \\\\
        t_{k+1} &\\gets \\frac{1 + \sqrt{1 + 4 t_k^2}}{2} \\\\
        z^{k+1} &\\gets w^k + \\frac{t_k - 1}{t_{k+1}} (w^k - w^{k-1})

    where :math:`\\nabla f(w)` is the gradient of :math:`f` given by the
    ``model.grad`` method and :math:`\\mathrm{prox}_{\\eta g}` is given by the
    ``prox.call`` method. The step-size :math:`\\eta` can be tuned with
    ``step``. The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` iterations. The obtained solution :math:`w` is returned
    by the ``solve`` method, and is also stored in the ``solution`` attribute
    of the solver.

    Parameters
    ----------
    step : `float`, default=None
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be automatically tuned as
        ``step = 1 / model.get_lip_best()``. If ``linesearch=True``, this is
        the first step-size to be used in the linesearch (that should be taken
        as too large).

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=100
        Maximum number of iterations of the solver.

    linesearch : `bool`, default=True
        If `True`, use backtracking linesearch to tune the step automatically.

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    print_every : `int`, default=10
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    linesearch_step_increase : `float`, default=2.
        Factor of step increase when using linesearch

    linesearch_step_decrease : `float`, default=0.5
        Factor of step decrease when using linesearch

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    References
    ----------
    * A. Beck and M. Teboulle, A fast iterative shrinkage-thresholding
      algorithm for linear inverse problems,
      *SIAM journal on imaging sciences*, 2009
    """

    def __init__(self, step: float = None, tol: float = 1e-10,
                 max_iter: int = 100, linesearch: bool = True,
                 linesearch_step_increase: float = 2.,
                 linesearch_step_decrease: float = 0.5, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)
        self.linesearch = linesearch
        self.linesearch_step_increase = linesearch_step_increase
        self.linesearch_step_decrease = linesearch_step_decrease

    def _initialize_values(self, x0=None, step=None):
        if step is None:
            if self.step is None:
                if self.linesearch:
                    # If we use linesearch, then we can choose a large
                    # initial step
                    step = 1e9
                else:
                    raise ValueError("No step specified.")
        step, obj, x, prev_x, grad_y = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=2)
        y = x.copy()
        t = 1.
        return x, prev_x, y, grad_y, t, step, obj

    def _gradient_step(self, x, prev_x, y, grad_y, t, prev_t, step):
        if self.linesearch:
            step *= self.linesearch_step_increase
            loss_y, _ = self.model.loss_and_grad(y, out=grad_y)
            obj_y = self.objective(y, loss=loss_y)
            while True:
                x[:] = self.prox.call(y - step * grad_y, step)
                obj_x = self.objective(x)
                envelope = obj_y + np.sum(grad_y * (x - y), axis=None) \
                           + 1. / (2 * step) * norm(x - y) ** 2
                test = (obj_x <= envelope)
                if test:
                    break
                step *= self.linesearch_step_decrease
                if step == 0:
                    break
        else:
            grad_y = self.model.grad(y)
            x[:] = self.prox.call(y - step * grad_y, step)
        t = np.sqrt((1. + (1. + 4. * t * t))) / 2.
        y[:] = x + (prev_t - 1) / t * (x - prev_x)
        return x, y, t, step

    def _solve(self, x0: np.ndarray = None, step: float = None):
        minimizer, prev_minimizer, y, grad_y, t, step, obj = \
            self._initialize_values(x0, step)
        for n_iter in range(self.max_iter):
            prev_t = t
            prev_minimizer[:] = minimizer

            # We will record on this iteration and we must be ready
            if self._should_record_iter(n_iter):
                prev_obj = self.objective(prev_minimizer)

            minimizer, y, t, step = self._gradient_step(
                minimizer, prev_minimizer, y, grad_y, t, prev_t, step)
            if step == 0:
                print('Step equals 0... at %i' % n_iter)
                break

            # Let's record metrics
            if self._should_record_iter(n_iter):
                rel_delta = relative_distance(minimizer, prev_minimizer)
                obj = self.objective(minimizer)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                converged = rel_obj < self.tol
                # If converged, we stop the loop and record the last step
                # in history
                self._handle_history(n_iter + 1, force=converged, obj=obj,
                                     x=minimizer.copy(), rel_delta=rel_delta,
                                     step=step, rel_obj=rel_obj)
                if converged:
                    break

        self._set("solution", minimizer)
        return minimizer


class GD(SolverFirstOrder):
    """Proximal gradient descent

    For the minimization of objectives of the form

    .. math::
        f(w) + g(w),

    where :math:`f` has a smooth gradient and :math:`g` is prox-capable.
    Function :math:`f` corresponds to the ``model.loss`` method of the model
    (passed with ``set_model`` to the solver) and :math:`g` corresponds to
    the ``prox.value`` method of the prox (passed with the ``set_prox`` method).
    One iteration of :class:`GD <tick.solver.GD>` is as follows:

    .. math::
        w \\gets \\mathrm{prox}_{\\eta g} \\big(w - \\eta \\nabla f(w) \\big),

    where :math:`\\nabla f(w)` is the gradient of :math:`f` given by the
    ``model.grad`` method and :math:`\\mathrm{prox}_{\\eta g}` is given by the
    ``prox.call`` method. The step-size :math:`\\eta` can be tuned with
    ``step``. The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` iterations. The obtained solution :math:`w` is returned
    by the ``solve`` method, and is also stored in the ``solution`` attribute
    of the solver.

    Parameters
    ----------
    step : `float`, default=None
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be automatically tuned as
        ``step = 1 / model.get_lip_best()``. If ``linesearch=True``, this is
        the first step-size to be used in the linesearch (that should be taken
        as too large).

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=100
        Maximum number of iterations of the solver.

    linesearch : `bool`, default=True
        If `True`, use backtracking linesearch to tune the step automatically.

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    print_every : `int`, default=10
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    linesearch_step_increase : `float`, default=2.
        Factor of step increase when using linesearch

    linesearch_step_decrease : `float`, default=0.5
        Factor of step decrease when using linesearch

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    References
    ----------
    * A. Beck and M. Teboulle, A fast iterative shrinkage-thresholding
      algorithm for linear inverse problems,
      *SIAM journal on imaging sciences*, 2009
    """

    def __init__(self, step: float = None, tol: float = 0.,
                 max_iter: int = 100, linesearch: bool = True,
                 linesearch_step_increase: float = 2.,
                 linesearch_step_decrease: float = 0.5, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1):
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)
        self.linesearch = linesearch
        self.linesearch_step_increase = linesearch_step_increase
        self.linesearch_step_decrease = linesearch_step_decrease

    def _initialize_values(self, x0=None, step=None):
        if step is None:
            if self.step is None:
                if self.linesearch:
                    # If we use linesearch, then we can choose a large
                    # initial step
                    step = 1e9
                else:
                    raise ValueError("No step specified.")
        step, obj, x, prev_x, x_new = \
            SolverFirstOrder._initialize_values(self, x0, step,
                                                n_empty_vectors=2)
        return x, prev_x, x_new, step, obj

    def _gradient_step(self, x, x_new, step):
        if self.linesearch:
            step *= self.linesearch_step_increase
            loss_x, grad_x = self.model.loss_and_grad(x)
            obj_x = self.objective(x, loss=loss_x)
            while True:
                x_new[:] = self.prox.call(x - step * grad_x, step)
                obj_x_new = self.objective(x_new)
                envelope = obj_x + np.sum(grad_x * (x_new - x),
                                          axis=None) \
                           + 1. / (2 * step) * norm(x_new - x) ** 2
                test = (obj_x_new <= envelope)
                if test:
                    break
                step *= self.linesearch_step_decrease
                if step == 0:
                    break
        else:
            grad_x = self.model.grad(x)
            x_new[:] = self.prox.call(x - step * grad_x, step)
            obj_x_new = self.objective(x_new)
        x[:] = x_new
        return x, step, obj_x_new

    def _solve(self, x0: np.ndarray = None, step: float = None):
        minimizer, prev_minimizer, x_new, step, obj = self._initialize_values(
            x0, step)
        for n_iter in range(self.max_iter):
            # We will record on this iteration and we must be ready
            if self._should_record_iter(n_iter):
                prev_minimizer[:] = minimizer
                prev_obj = self.objective(prev_minimizer)

            minimizer, step, obj = self._gradient_step(minimizer, x_new, step)

            if step == 0:
                print('Step equals 0... at %i' % n_iter)
                break

            # Let's record metrics
            if self._should_record_iter(n_iter):
                rel_delta = relative_distance(minimizer, prev_minimizer)
                obj = self.objective(minimizer)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                converged = rel_obj < self.tol
                # If converged, we stop the loop and record the last step
                # in history
                self._handle_history(n_iter + 1, force=converged, obj=obj,
                                     x=minimizer.copy(), rel_delta=rel_delta,
                                     step=step, rel_obj=rel_obj)
                if converged:
                    break

        self._set("solution", minimizer)
        return minimizer


from scipy.optimize import fmin_bfgs

class BFGS(SolverFirstOrder):
    """Broyden, Fletcher, Goldfarb, and Shanno algorithm

    This solver is actually a simple wrapping of `scipy.optimize.fmin_bfgs`
    BFGS (Broyden, Fletcher, Goldfarb, and Shanno) algorithm. This is a
    quasi-newton algotithm that builds iteratively approximations of the inverse
    Hessian. This solver can be used to minimize objectives of the form

    .. math::
        f(w) + g(w),

    for :math:`f` with a smooth gradient and only :math:`g` corresponding to
    the zero penalization (namely :class:`ProxZero <tick.prox.ProxZero>`)
    or ridge penalization (namely :class:`ProxL2sq <tick.prox.ProxL2sq>`).
    Function :math:`f` corresponds to the ``model.loss`` method of the model
    (passed with ``set_model`` to the solver) and :math:`g` corresponds to
    the ``prox.value`` method of the prox (passed with the ``set_prox`` method).
    The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` iterations. The obtained solution :math:`w` is returned
    by the ``solve`` method, and is also stored in the ``solution`` attribute
    of the solver.

    Parameters
    ----------
    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=10
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    print_every : `int`, default=10
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    References
    ----------
    * Quasi-Newton method of Broyden, Fletcher, Goldfarb and Shanno (BFGS),
      see Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.
    """

    _attrinfos = {"_prox_grad": {"writable": False}}

    def __init__(self, tol: float = 1e-10, max_iter: int = 10,
                 verbose: bool = True, print_every: int = 1,
                 record_every: int = 1):
        SolverFirstOrder.__init__(self, step=None, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)
        self._prox_grad = None

    def set_prox(self, prox: Prox):
        """Set proximal operator in the solver.

        Parameters
        ----------
        prox : `Prox`
            The proximal operator of the penalization function

        Returns
        -------
        output : `Solver`
            The solver with given prox

        Notes
        -----
        In some solvers, ``set_model`` must be called before
        ``set_prox``, otherwise and error might be raised.
        """
        if type(prox) is ProxZero:
            SolverFirstOrder.set_prox(self, prox)
            self._set("_prox_grad", lambda x: x)
        elif type(prox) is ProxL2Sq:
            SolverFirstOrder.set_prox(self, prox)
            self._set("_prox_grad", lambda x: prox.strength * x)
        else:
            raise ValueError("BFGS only accepts ProxZero and ProxL2sq "
                             "for now")
        return self

    def solve(self, x0=None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : `np.array`, shape=(n_coeffs,), default=`None`
            Starting point of the solver

        Returns
        -------
        output : `np.array`, shape=(n_coeffs,)
            Obtained minimizer for the problem
        """
        self._start_solve()
        coeffs = self._solve(x0)
        self._set("solution", coeffs)
        self._end_solve()
        return self.solution

    def _solve(self, x0: np.ndarray = None):
        if x0 is None:
            x0 = np.zeros(self.model.n_coeffs, dtype=self.dtype)
        obj = self.objective(x0)

        # A closure to maintain history along internal BFGS's iterations
        n_iter = [0]
        prev_x = x0.copy()

        def insp(xk):
            if self._should_record_iter(n_iter[0]):
                prev_obj = self.objective(prev_x)
                x = xk
                rel_delta = relative_distance(x, prev_x)

                obj = self.objective(x)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj)
                self._handle_history(n_iter[0], force=False, obj=obj,
                                     x=xk.copy(), rel_delta=rel_delta,
                                     rel_obj=rel_obj)
            prev_x[:] = xk
            n_iter[0] += 1

        insp.n_iter = n_iter
        insp.self = self
        insp.prev_x = prev_x

        # We simply call the scipy.optimize.fmin_bfgs routine
        x_min, f_min, _, _, _, _, _ = \
            fmin_bfgs(lambda x: self.model.loss(x) + self.prox.value(x),
                      x0,
                      lambda x: self.model.grad(x) + self._prox_grad(x),
                      maxiter=self.max_iter, gtol=self.tol,
                      callback=insp, full_output=True,
                      disp=False, retall=False)

        return x_min

    def set_model(self, model: Model):
        """Set model in the solver

        Parameters
        ----------
        model : `Model`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things)

        Returns
        -------
        output : `Solver`
            The `Solver` with given model
        """
        self.dtype = model.dtype
        if np.dtype(self.dtype) != np.dtype("float64"):
            raise ValueError(
                "Solver BFGS currenty only accepts float64 array types")
        return SolverFirstOrder.set_model(self, model)


class SolverSto(Base):
    """The base class for a stochastic solver.
    In only deals with verbosing information, and setting parameters.

    Parameters
    ----------
    epoch_size : `int`, default=0
        Epoch size. If given before calling set_model, then we'll
        use the specified value. If not, we ``epoch_size`` is specified
        by the model itself, when calling set_model

    rand_type : `str`
        Type of random sampling

        * if ``"unif"`` samples are uniformly drawn among all possibilities
        * if ``"perm"`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    seed : `int`
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {
        "_solver": {
            "writable": False
        },
        "epoch_size": {
            "writable": True,
            "cpp_setter": "set_epoch_size"
        },
        "_rand_max": {
            "writable": False,
            "cpp_setter": "set_rand_max"
        },
        "_rand_type": {
            "writable": False,
            "cpp_setter": "set_rand_type"
        },
        "seed": {
            "cpp_setter": "set_seed"
        }
    }

    # The name of the attribute that might contain the C++ solver object
    _cpp_obj_name = "_solver"

    def __init__(self, epoch_size: int = None, rand_type: str = "unif",
                 seed=-1):
        Base.__init__(self)
        # The C++ wrapped solver is to be given in child classes
        self._solver = None
        self._rand_type = None
        self._rand_max = None
        self.epoch_size = epoch_size
        self.rand_type = rand_type
        self.seed = seed

    def set_model(self, model: Model):
        # Give the C++ wrapped model to the solver
        self.dtype = model.dtype
        self._solver.set_model(model._model)
        # If not already specified, we use the model's epoch_size
        if self.epoch_size is None:
            self.epoch_size = model._epoch_size
        # We always use the _rand_max given by the model
        self._set_rand_max(model)
        return self

    def set_prox(self, prox: Prox):
        if prox._prox is None:
            raise ValueError("Prox %s is not compatible with stochastic "
                             "solver %s" % (prox.__class__.__name__,
                                            self.__class__.__name__))
            # Give the C++ wrapped prox to the solver
        if self.dtype is None or self.model is None:
            raise ValueError("Solver must call set_model before set_prox")
        if prox.dtype != self.dtype:
            prox = prox.astype(self.dtype)
        self._solver.set_prox(prox._prox)
        return self

    @property
    def rand_type(self):
        if self._rand_type == unif:
            return "unif"
        if self._rand_type == perm:
            return "perm"
        else:
            raise ValueError("No known ``rand_type``")

    @rand_type.setter
    def rand_type(self, val):
        if val not in ["unif", "perm"]:
            raise ValueError("``rand_type`` can be 'unif' or " "'perm'")
        else:
            if val == "unif":
                enum_val = unif
            if val == "perm":
                enum_val = perm
            self._set("_rand_type", enum_val)

    def _set_rand_max(self, model):
        model_rand_max = model._rand_max
        self._set("_rand_max", model_rand_max)

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        """Deduce dtype and return true if C++ _model should be set
        """
        #import tick.base.dtype_to_cpp_type
        #return tick.base.dtype_to_cpp_type.get_typed_class(
        #    self, dtype_or_object_with_dtype, dtype_map)
        return get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)


class SolverFirstOrderSto(SolverFirstOrder, SolverSto):
    """The base class for a first order stochastic solver.
    It only deals with verbosing information, and setting parameters.

    Parameters
    ----------
    epoch_size : `int`
        Epoch size

    rand_type : `str`
        Type of random sampling

        * if ``"unif"`` samples are uniformly drawn among all possibilities
        * if ``"perm"`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    tol : `float`, default=0
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default = 10
        Print history information every time the iteration number is a
        multiple of ``print_every``

    record_every : `int`, default = 1
        Information along iteration is recorded in history each time the
        iteration number of a multiple of ``record_every``

    seed : `int`
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    Attributes
    ----------
    model : `Solver`
        The model to solve

    prox : `Prox`
        Proximal operator to solve

    time_start : `str`
        Start date of the call to solve()

    time_elapsed : `float`
        Duration of the call to solve(), in seconds

    time_end : `str`
        End date of the call to solve()

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {"_step": {"writable": False}}

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type="unif", tol: float = 0., max_iter=100, verbose=True,
                 print_every=10, record_every=1, seed=-1):

        self._step = None

        # We must first construct SolverSto (otherwise self.step won't
        # work in SolverFirstOrder)
        SolverSto.__init__(self, epoch_size=epoch_size, rand_type=rand_type,
                           seed=seed)
        SolverFirstOrder.__init__(self, step=step, tol=tol, max_iter=max_iter,
                                  verbose=verbose, print_every=print_every,
                                  record_every=record_every)

        self._set_cpp_solver('float64')

    def set_model(self, model: Model):
        """Set model in the solver

        Parameters
        ----------
        model : `Model`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things)

        Returns
        -------
        output : `Solver`
            The `Solver` with given model
        """
        self.validate_model(model)
        if self.dtype != model.dtype or self._solver is None:
            self._set_cpp_solver(model.dtype)

        self.dtype = model.dtype
        SolverFirstOrder.set_model(self, model)
        SolverSto.set_model(self, model)
        return self

    def set_prox(self, prox: Prox):
        """Set proximal operator in the solver

        Parameters
        ----------
        prox : `Prox`
            The proximal operator of the penalization function

        Returns
        -------
        output : `Solver`
            The solver with given prox

        Notes
        -----
        In some solvers, ``set_model`` must be called before
        ``set_prox``, otherwise and error might be raised
        """
        SolverFirstOrder.set_prox(self, prox)
        SolverSto.set_prox(self, prox)
        return self

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        self._set("_step", val)
        if val is None:
            val = 0.
        if self._solver is not None:
            self._solver.set_step(val)

    @property
    def record_every(self):
        if hasattr(self, '_solver') and self._solver is not None:
            return self._solver.get_record_every()
        else:
            return self._record_every

    @record_every.setter
    def record_every(self, val):
        self._record_every = val
        if hasattr(self, '_solver') and self._solver is not None:
            self._solver.set_record_every(val)

    def _solve(self, x0: np.array = None, step: float = None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : np.array, shape=(n_coeffs,)
            Starting iterate for the solver

        step : float
            Step-size or learning rate for the solver

        Returns
        -------
        output : np.array, shape=(n_coeffs,)
            Obtained minimizer
        """
        #from tick.solver import SDCA
        if not isinstance(self, SDCA):
            if step is not None:
                self.step = step
            step, obj, minimizer, prev_minimizer = \
                self._initialize_values(x0, step, n_empty_vectors=1)
            self._solver.set_starting_iterate(minimizer)

        else:
            # In sdca case x0 is a dual vector
            step, obj, minimizer, prev_minimizer = \
                self._initialize_values(None, step, n_empty_vectors=1)
            if x0 is not None:
                self._solver.set_starting_iterate(x0)

        if self.verbose or self.tol != 0:
            self._solve_with_printing(prev_minimizer, minimizer)
        else:
            self._solve_and_record_in_cpp(minimizer)

        self._solver.get_minimizer(minimizer)
        self._set("solution", minimizer)
        return minimizer

    def _solve_with_printing(self, prev_minimizer, minimizer):
        # At each iteration we call self._solver.solve that does a full
        # epoch
        prev_minimizer[:] = minimizer
        prev_obj = self.objective(prev_minimizer)

        for n_iter in range(self.max_iter):

            # Launch one epoch using the wrapped C++ solver
            self._solver.solve()

            # Let's record metrics
            if self._should_record_iter(n_iter):
                self._solver.get_minimizer(minimizer)
                # The step might be modified by the C++ solver
                # step = self._solver.get_step()
                obj = self.objective(minimizer)
                rel_delta = relative_distance(minimizer, prev_minimizer)
                rel_obj = abs(obj - prev_obj) / abs(prev_obj) \
                    if prev_obj != 0 else abs(obj)
                converged = rel_obj < self.tol
                # If converged, we stop the loop and record the last step
                # in history
                self._handle_history(n_iter + 1, force=converged, obj=obj,
                                     x=minimizer.copy(), rel_delta=rel_delta,
                                     rel_obj=rel_obj)
                prev_minimizer[:] = minimizer
                prev_obj = self.objective(prev_minimizer)
                if converged:
                    break

    def _solve_and_record_in_cpp(self, minimizer):
        prev_obj = self.objective(minimizer)
        self._solver.set_prev_obj(prev_obj)
        self._solver.solve(self.max_iter)
        self._post_solve_and_record_in_cpp(minimizer, prev_obj)

    def _post_solve_and_record_in_cpp(self, minimizer, prev_obj):
        prev_iterate = minimizer
        for epoch, iter_time, iterate, obj in zip(
                self._solver.get_epoch_history(),
                self._solver.get_time_history(),
                self._solver.get_iterate_history(),
                self._solver.get_objectives()):
            if epoch is self._solver.get_epoch_history()[-1]:
                # This rel_obj is not exactly the same one as prev_obj is not the
                # objective of the previous epoch but of the previouly recorded
                # epoch
                self._handle_history(
                    epoch, force=True,
                    obj=obj, iter_time=iter_time, x=iterate,
                    rel_delta=relative_distance(iterate, prev_iterate),
                    rel_obj=abs(obj - prev_obj) / abs(prev_obj) \
                        if prev_obj != 0 else abs(obj))
            prev_obj = obj
            prev_iterate[:] = iterate
        minimizer = prev_iterate

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        #import tick.base.dtype_to_cpp_type
        #return tick.base.dtype_to_cpp_type.get_typed_class(
        #    self, dtype_or_object_with_dtype, dtype_map)
        return get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)

    def _extract_dtype(self, dtype_or_object_with_dtype):
        #import tick.base.dtype_to_cpp_type
        #return tick.base.dtype_to_cpp_type.extract_dtype(
        #    dtype_or_object_with_dtype)
        return extract_dtype(
            dtype_or_object_with_dtype)

    @abstractmethod
    def _set_cpp_solver(self, dtype):
        pass

    def astype(self, dtype_or_object_with_dtype):
        if self.model is None:
            raise ValueError("Cannot reassign solver without a model")

        #import tick.base.dtype_to_cpp_type
        #new_solver = tick.base.dtype_to_cpp_type.copy_with(
        #    self,
        #    ["prox", "model", "_solver"]  # ignore on deepcopy
        #)
        new_solver = copy_with(
            self,
            ["prox", "model", "_solver"]  # ignore on deepcopy
        )
        new_solver._set_cpp_solver(dtype_or_object_with_dtype)
        new_solver.set_model(self.model.astype(new_solver.dtype))
        if self.prox is not None:
            new_solver.set_prox(self.prox.astype(new_solver.dtype))
        return new_solver


class SGD(SolverFirstOrderSto):
    """Stochastic gradient descent solver

    For the minimization of objectives of the form

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(w) + g(w),

    where the functions :math:`f_i` have smooth gradients and :math:`g` is
    prox-capable. Function :math:`f = \\frac 1n \\sum_{i=1}^n f_i` corresponds
    to the ``model.loss`` method of the model (passed with ``set_model`` to the
    solver) and :math:`g` corresponds to the ``prox.value`` method of the
    prox (passed with the ``set_prox`` method).
    One iteration of :class:`SGD <tick.solver.SGD>` corresponds to the
    following iteration applied ``epoch_size`` times:

    .. math::
        w^{t+1} \\gets \\mathrm{prox}_{\\eta_t g} \\big(w^t - \\eta_t
        \\nabla f_i(w^t) \\big),

    where :math:`i` is sampled at random (strategy depends on ``rand_type``) at
    each iteration, where :math:`\\eta_t = \eta / (t + 1)`, with
    :math:`\\eta > 0` that can be tuned with ``step``. The seed of the random
    number generator for generation of samples :math:`i` can be seeded with
    ``seed``. The iterations stop whenever tolerance ``tol`` is achieved, or
    after ``max_iter`` epochs (namely ``max_iter``:math:`\\times` ``epoch_size``
    iterations).
    The obtained solution :math:`w` is returned by the ``solve`` method, and is
    also stored in the ``solution`` attribute of the solver.

    Parameters
    ----------
    step : `float`
        Step-size parameter, the most important parameter of the solver.
        A try-an-improve approach should be used.

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=100
        Maximum number of iterations of the solver, namely maximum number of
        epochs (by default full pass over the data, unless ``epoch_size`` has
        been modified from default)

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    seed : `int`, default=-1
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    epoch_size : `int`, default given by model
        Epoch size, namely how many iterations are made before updating the
        variance reducing term. By default, this is automatically tuned using
        information from the model object passed through ``set_model``.

    rand_type : {'unif', 'perm'}, default='unif'
        How samples are randomly selected from the data

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    print_every : `int`, default=10
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    References
    ----------
    * https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    """

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = "unif", tol: float = 1e-10,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1, seed: int = -1):

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type, tol,
                                     max_iter, verbose, print_every,
                                     record_every, seed)

    def _set_cpp_solver(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        solver_class = self._get_typed_class(dtype_or_object_with_dtype,
                                             dtype_class_mapper)

        # Type mapping None to unsigned long and double does not work...
        step = self.step
        if step is None:
            step = 0.
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        self._set(
            '_solver',
            solver_class(epoch_size, self.tol, self._rand_type, step,
                         self.record_every, self.seed))


variance_reduction_methods_mapper = {
    'last': SVRG_VarianceReductionMethod_Last,
    'avg': SVRG_VarianceReductionMethod_Average,
    'rand': SVRG_VarianceReductionMethod_Random
}

step_types_mapper = {
    'fixed': SVRG_StepType_Fixed,
    'bb': SVRG_StepType_BarzilaiBorwein
}

dtype_class_mapper = {
    np.dtype('float32'): _SVRGFloat,
    np.dtype('float64'): _SVRGDouble
}


class SVRG(SolverFirstOrderSto):
    """Stochastic Variance Reduced Gradient solver

    For the minimization of objectives of the form

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(w) + g(w),

    where the functions :math:`f_i` have smooth gradients and :math:`g` is
    prox-capable. Function :math:`f = \\frac 1n \\sum_{i=1}^n f_i` corresponds
    to the ``model.loss`` method of the model (passed with ``set_model`` to the
    solver) and :math:`g` corresponds to the ``prox.value`` method of the
    prox (passed with the ``set_prox`` method).
    One iteration of :class:`SVRG <tick.solver.SVRG>` corresponds to the
    following iteration applied ``epoch_size`` times:

    .. math::
        w \\gets \\mathrm{prox}_{\\eta g} \\big(w - \\eta (\\nabla f_i(w) -
        \\nabla f_i(\\bar{w}) + \\nabla f(\\bar{w}) \\big),

    where :math:`i` is sampled at random (strategy depends on ``rand_type``) at
    each iteration, and where :math:`\\bar w` and :math:`\\nabla f(\\bar w)`
    are updated at the beginning of each epoch, with a strategy that depend on
    the ``variance_reduction`` parameter. The step-size :math:`\\eta` can be
    tuned with ``step``, the seed of the random number generator for generation
    of samples :math:`i` can be seeded with ``seed``. The iterations stop
    whenever tolerance ``tol`` is achieved, or after ``max_iter`` epochs
    (namely ``max_iter`` :math:`\\times` ``epoch_size`` iterates).
    The obtained solution :math:`w` is returned by the ``solve`` method, and is
    also stored in the ``solution`` attribute of the solver.

    Internally, :class:`SVRG <tick.solver.SVRG>` has dedicated code when
    the model is a generalized linear model with sparse features, and a
    separable proximal operator: in this case, each iteration works only in the
    set of non-zero features, leading to much faster iterates.

    Moreover, when ``n_threads`` > 1, this class actually implements parallel
    and asynchronous updates of :math:`w`, which is likely to accelerate
    optimization, depending on the sparsity of the dataset, and the number of
    available cores.

    Parameters
    ----------
    step : `float`
        Step-size parameter, the most important parameter of the solver.
        Whenever possible, this can be automatically tuned as
        ``step = 1 / model.get_lip_max()``. Otherwise, use a try-an-improve
        approach

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=10
        Maximum number of iterations of the solver, namely maximum number of
        epochs (by default full pass over the data, unless ``epoch_size`` has
        been modified from default)

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    seed : `int`, default=-1
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    n_threads : `int`, default=1
        Number of threads to use for parallel optimization. The strategy used
        for this is asynchronous updates of the iterates.

    epoch_size : `int`, default given by model
        Epoch size, namely how many iterations are made before updating the
        variance reducing term. By default, this is automatically tuned using
        information from the model object passed through ``set_model``.

    variance_reduction : {'last', 'avg', 'rand'}, default='last'
        Strategy used for the computation of the iterate used in
        variance reduction (also called phase iterate). A warning will be
        raised if the ``'avg'`` strategy is used when the model is a
        generalized linear model with sparse features, since it is strongly
        sub-optimal in this case

        * ``'last'`` : the phase iterate is the last iterate of the previous
          epoch
        * ``'avg``' : the phase iterate is the average over the iterates in the
          past epoch
        * ``'rand'``: the phase iterate is a random iterate of the previous
          epoch

    rand_type : {'unif', 'perm'}, default='unif'
        How samples are randomly selected from the data

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    step_type : {'fixed', 'bb'}, default='fixed'
        How step will evoluate over stime

        * if ``'fixed'`` step will remain equal to the given step accross
          all iterations. This is the fastest solution if the optimal step
          is known.
        * if ``'bb'`` step will be chosen given Barzilai Borwein rule. This
          choice is much more adaptive and should be used if optimal step if
          difficult to obtain.

    print_every : `int`, default=1
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    References
    ----------
    * L. Xiao and T. Zhang, A proximal stochastic gradient method with
      progressive variance reduction, *SIAM Journal on Optimization* (2014)

    * Tan, C., Ma, S., Dai, Y. H., & Qian, Y.
      Barzilai-Borwein step size for stochastic gradient descent.
      *Advances in Neural Information Processing Systems* (2016)

    * Mania, H., Pan, X., Papailiopoulos, D., Recht, B., Ramchandran, K. and
      Jordan, M.I., 2015.
      Perturbed iterate analysis for asynchronous stochastic optimization.
    """
    _attrinfos = {"_step_type_str": {}, "_var_red_str": {}}

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = 'unif', tol: float = 1e-10,
                 max_iter: int = 10, verbose: bool = True,
                 print_every: int = 1, record_every: int = 1, seed: int = -1,
                 variance_reduction: str = 'last', step_type: str = 'fixed',
                 n_threads: int = 1):
        self.n_threads = n_threads
        # temporary to hold step type before dtype is known
        self._step_type_str = step_type
        # temporary to hold varience reduction type before dtype is known
        self._var_red_str = variance_reduction

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type, tol,
                                     max_iter, verbose, print_every,
                                     record_every, seed=seed)

    @property
    def variance_reduction(self):
        return next((k for k, v in variance_reduction_methods_mapper.items()
                     if v == self._solver.get_variance_reduction()), None)

    @variance_reduction.setter
    def variance_reduction(self, val: str):
        if val not in variance_reduction_methods_mapper:
            raise ValueError(
                'variance_reduction should be one of "{}", got "{}"'.format(
                    ', '.join(
                        sorted(variance_reduction_methods_mapper.keys())),
                    val))
        if self.model is not None:
            if val == 'avg' and self.model._model.is_sparse():
                warn(
                    "'avg' variance reduction cannot be used "
                    "with sparse datasets", UserWarning)
        self._solver.set_variance_reduction(
            variance_reduction_methods_mapper[val])

    @property
    def step_type(self):
        return next((k for k, v in step_types_mapper.items()
                     if v == self._solver.get_step_type()), None)

    @step_type.setter
    def step_type(self, val: str):
        if val not in step_types_mapper:
            raise ValueError(
                'step_type should be one of "{}", got "{}"'.format(
                    ', '.join(sorted(step_types_mapper.keys())), val))
        self._solver.set_step_type(step_types_mapper[val])

    def set_model(self, model: Model):
        """Set model in the solver

        Parameters
        ----------
        model : `Model`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things)

        Returns
        -------
        output : `Solver`
            The `Solver` with given model
        """
        # We need to check that the setted model is not sparse when the
        # variance reduction method is 'avg'
        if self._var_red_str == 'avg' and model._model.is_sparse():
            warn(
                "'avg' variance reduction cannot be used with sparse "
                "datasets. Please change `variance_reduction` before "
                "passing sparse data.", UserWarning)

        if hasattr(model, "n_threads"):
            model.n_threads = self.n_threads

        return SolverFirstOrderSto.set_model(self, model)

    def _set_cpp_solver(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        solver_class = self._get_typed_class(dtype_or_object_with_dtype,
                                             dtype_class_mapper)

        # Type mapping None to unsigned long and double does not work...
        step = self.step
        if step is None:
            step = 0.
        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        self._set(
            '_solver',
            solver_class(epoch_size, self.tol, self._rand_type, step,
                         self.record_every, self.seed, self.n_threads))

        self.variance_reduction = self._var_red_str
        self.step_type = self._step_type_str

    def multi_solve(self, coeffes, solvers, max_iter, threads = None, set_start = True):
        """Complete function for calling solve on multiple independent SVRG C++ instances
           Requires valid solvers setup with model and prox. Vectors of instances are
           peculiar with SWIG, so we use a vector of pointers, populate the C++ vector from
           Python, then run the solve on each object behind the pointer in C++

        Parameters
        ----------
        coeffes : `np.array`, shape=(n_coeffs,)
            First minimizer and possible starting_iterate for solvers
        solvers : `List of SVRG`
            Solver classes to be solved

        max_iter : `int`
            Default max number of iterations if tolerance not hit

        threads : `optional int`
            If None - len(solver) threads are spawned
            otherwise and threadpool with number "threads" is spawned
        set_start: `bool`
            If True, coeffes[i] is used for the starting iterate of solvers[i]
        """

        if len(coeffes) != len(solvers):
            raise ValueError("size mismatch between coeffes and solvers")
        mins = []
        sss = SVRGDoublePtrVector(0)
        for i in range(len(solvers)):
            solvers[i]._solver.reset()
            mins.append(coeffes[i].copy())
            if threads is None and set_start:
                solvers[i]._solver.set_starting_iterate(mins[-1])
            MultiSVRG.push_solver(sss, solvers[i]._solver) # push SVRG C++ pointer to vector sss
            solvers[i]._start_solve()
        if threads is None:
            MultiSVRG.multi_solve(sss, max_iter)
        elif set_start:
            MultiSVRG.multi_solve(sss, coeffes, max_iter, threads)
        else:
            MultiSVRG.multi_solve(sss, max_iter, threads)
        for i in range(len(solvers)):
            solvers[i]._set("time_elapsed", solvers[i]._solver.get_time_history()[-1])
            if solvers[i].verbose:
                print("Done solving using " + solvers[i].name + " in " +
                      str(solvers[i].time_elapsed) + " seconds")
            solvers[i]._post_solve_and_record_in_cpp(mins[i], solvers[i]._solver.get_first_obj())
        return mins


class SDCA(SolverFirstOrderSto):
    """Stochastic Dual Coordinate Ascent

    For the minimization of objectives of the form

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(w^\\top x_i) + g(w),

    where the functions :math:`f_i` have smooth gradients and :math:`g` is
    prox-capable. This solver actually requires more than that, since it is
    working in a Fenchel dual formulation of the primal problem given above.
    First, it requires that some ridge penalization is used, hence the mandatory
    parameter ``l_l2sq`` below: SDCA will actually minimize the objective

    .. math::
        \\frac 1n \\sum_{i=1}^n f_i(x_i^\\top w) + g(w) + \\frac{\\lambda}{2}
        \\| w \\|_2^2,

    where :math:`\lambda` is tuned with the ``l_l2sq`` (see below). Now, putting
    :math:`h(w) = g(w) + \lambda \|w\|_2^2 / 2`, SDCA maximize
    the Fenchel dual problem

    .. math::
        D(\\alpha) = \\frac 1n \\sum_{i=1}^n \\Bigg[ - f_i^*(-\\alpha_i) -
        \lambda h^*\\Big( \\frac{1}{\\lambda n} \\sum_{i=1}^n \\alpha_i x_i)
        \\Big) \\Bigg],

    where :math:`f_i^*` and :math:`h^*` and the Fenchel duals of :math:`f_i`
    and :math:`h` respectively.
    Function :math:`f = \\frac 1n \\sum_{i=1}^n f_i` corresponds
    to the ``model.loss`` method of the model (passed with ``set_model`` to the
    solver) and :math:`g` corresponds to the ``prox.value`` method of the
    prox (passed with the ``set_prox`` method). One iteration of
    :class:`SDCA <tick.solver.SDCA>` corresponds to the
    following iteration applied ``epoch_size`` times:

    .. math::
        \\begin{align*}
        \\delta_i &\\gets \\arg\\min_{\\delta} \\Big[ \\; f_i^*(-\\alpha_i -
        \\delta) + w^\\top x_i \\delta + \\frac{1}{2 \\lambda n} \\| x_i\\|_2^2
        \\delta^2 \\Big] \\\\
        \\alpha_i &\\gets \\alpha_i + \\delta_i \\\\
        v &\\gets v + \\frac{1}{\\lambda n} \\delta_i x_i \\\\
        w &\\gets \\nabla g^*(v)
        \\end{align*}

    where :math:`i` is sampled at random (strategy depends on ``rand_type``) at
    each iteration. The ridge regularization :math:`\\lambda` can be tuned with
    ``l_l2sq``, the seed of the random number generator for generation
    of samples :math:`i` can be seeded with ``seed``. The iterations stop
    whenever tolerance ``tol`` is achieved, or after ``max_iter`` epochs
    (namely ``max_iter`` :math:`\\times` ``epoch_size`` iterates).
    The obtained solution :math:`w` is returned by the ``solve`` method, and is
    also stored in the ``solution`` attribute of the solver. The dual solution
    :math:`\\alpha` is stored in the ``dual_solution`` attribute.

    Internally, :class:`SDCA <tick.solver.SDCA>` has dedicated code when
    the model is a generalized linear model with sparse features, and a
    separable proximal operator: in this case, each iteration works only in the
    set of non-zero features, leading to much faster iterates.

    Parameters
    ----------
    l_l2sq : `float`
        Level of L2 penalization. L2 penalization is mandatory for SDCA.
        Convergence properties of this solver are deeply connected to this
        parameter, which should be understood as the "step" used by the
        algorithm.

    tol : `float`, default=1e-10
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it)

    max_iter : `int`, default=10
        Maximum number of iterations of the solver, namely maximum number of
        epochs (by default full pass over the data, unless ``epoch_size`` has
        been modified from default)

    verbose : `bool`, default=True
        If `True`, solver verboses history, otherwise nothing is displayed,
        but history is recorded anyway

    seed : `int`, default=-1
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    epoch_size : `int`, default given by model
        Epoch size, namely how many iterations are made before updating the
        variance reducing term. By default, this is automatically tuned using
        information from the model object passed through ``set_model``.

    rand_type : {'unif', 'perm'}, default='unif'
        How samples are randomly selected from the data

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    print_every : `int`, default=1
        Print history information every time the iteration number is a
        multiple of ``print_every``. Used only is ``verbose`` is True

    record_every : `int`, default=1
        Save history information every time the iteration number is a
        multiple of ``record_every``

    Attributes
    ----------
    model : `Model`
        The model used by the solver, passed with the ``set_model`` method

    prox : `Prox`
        Proximal operator used by the solver, passed with the ``set_prox``
        method

    solution : `numpy.array`, shape=(n_coeffs,)
        Minimizer found by the solver

    dual_solution : `numpy.array`
        Dual vector corresponding to the primal solution obtained by the solver

    history : `dict`-like
        A dict-type of object that contains history of the solver along
        iterations. It should be accessed using the ``get_history`` method

    time_start : `str`
        Start date of the call to ``solve()``

    time_elapsed : `float`
        Duration of the call to ``solve()``, in seconds

    time_end : `str`
        End date of the call to ``solve()``

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    References
    ----------
    * S. Shalev-Shwartz and T. Zhang, Accelerated proximal stochastic dual
      coordinate ascent for regularized loss minimization, *ICML 2014*
    """

    _attrinfos = {'l_l2sq': {'cpp_setter': 'set_l_l2sq'}}

    def __init__(self, l_l2sq: float, epoch_size: int = None,
                 rand_type: str = 'unif', tol: float = 1e-10,
                 max_iter: int = 10, verbose: bool = True,
                 print_every: int = 1, record_every: int = 1, seed: int = -1):

        self.l_l2sq = l_l2sq
        SolverFirstOrderSto.__init__(
            self, step=0, epoch_size=epoch_size, rand_type=rand_type, tol=tol,
            max_iter=max_iter, verbose=verbose, print_every=print_every,
            record_every=record_every, seed=seed)

    def _set_cpp_solver(self, dtype_or_object_with_dtype):
        self.dtype = self._extract_dtype(dtype_or_object_with_dtype)
        solver_class = self._get_typed_class(dtype_or_object_with_dtype,
                                             dtype_class_mapper)

        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        self._set(
            '_solver',
            solver_class(self.l_l2sq, epoch_size, self.tol, self._rand_type,
                         self.record_every, self.seed))

    def objective(self, coeffs, loss: float = None):
        """Compute the objective minimized by the solver at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The objective is computed at this point

        loss : `float`, default=`None`
            Gives the value of the loss if already known (allows to
            avoid its computation in some cases)

        Returns
        -------
        output : `float`
            Value of the objective at given ``coeffs``
        """
        prox_l2_value = 0.5 * self.l_l2sq * np.linalg.norm(coeffs) ** 2
        return SolverFirstOrderSto.objective(self, coeffs,
                                             loss) + prox_l2_value

    def dual_objective(self, dual_coeffs):
        """Compute the dual objective at ``dual_coeffs``

        Parameters
        ----------
        dual_coeffs : `numpy.ndarray`, shape=(n_samples,)
            The dual objective objective is computed at this point

        Returns
        -------
        output : `float`
            Value of the dual objective at given ``dual_coeffs``
        """
        primal = self.model._sdca_primal_dual_relation(self.l_l2sq,
                                                       dual_coeffs)
        prox_l2_value = 0.5 * self.l_l2sq * np.linalg.norm(primal) ** 2
        return self.model.dual_loss(dual_coeffs) - prox_l2_value

    def _set_rand_max(self, model):
        try:
            # Some model, like Poisreg with linear link, have a special
            # rand_max for SDCA
            model_rand_max = model._sdca_rand_max
        except (AttributeError, NotImplementedError):
            model_rand_max = model._rand_max

        self._set("_rand_max", model_rand_max)

    @property
    def dual_solution(self):
        return self._solver.get_dual_vector()





class LearnerOptim(ABC, Base):
    """Learner for all models that are inferred with a `tick.solver`
    and a `tick.prox`
    Not intended for end-users, but for development only.
    It should be sklearn-learn compliant

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : 'none', 'l1', 'l2', 'elasticnet', 'tv', 'binarsity', default='l2'
        The penalization to use. Default 'l2', namely is ridge penalization.

    solver : 'gd', 'agd', 'bfgs', 'svrg', 'sdca'
        The name of the solver to use

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning. Used in 'gd', 'agd', 'sgd'
        and 'svrg' solvers

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Other Parameters
    ----------------
    sdca_ridge_strength : `float`, default=1e-3
        It controls the strength of the additional ridge penalization. Used in
        'sdca' solver

    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    random_state : int seed, RandomState instance, or None (default)
        The seed that will be used by stochastic solvers. Used in 'sgd',
        'svrg', and 'sdca' solvers

    blocks_start : `numpy.array`, shape=(n_features,), default=None
        The indices of the first column of each binarized feature blocks. It
        corresponds to the ``feature_indices`` property of the
        ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    blocks_length : `numpy.array`, shape=(n_features,), default=None
        The length of each binarized feature blocks. It corresponds to the
        ``n_values`` property of the ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty
    """


    _attrinfos = {
        "solver": {
            "writable": False
        },
        "_solver_obj": {
            "writable": False
        },
        "penalty": {
            "writable": False
        },
        "_prox_obj": {
            "writable": False
        },
        "_model_obj": {
            "writable": False
        },
        "_fitted": {
            "writable": False
        },
        "_C": {
            "writable": False
        },
        "random_state": {
            "writable": False
        },
        "_warm_start": {
            "writable": False
        },
        "_actual_kwargs": {
            "writable": False
        },
    }

    #_solvers = {
    #    'gd': 'GD',
    #    'agd': 'AGD',
    #    'sgd': 'SGD',
    #    'svrg': 'SVRG',
    #    'bfgs': 'BFGS',
    #    'sdca': 'SDCA'
    #}
    _solvers = {
        'gd': GD,
        'agd': AGD,
        'sgd': SGD,
        'svrg': SVRG,
        'bfgs': BFGS,
        'sdca': SDCA
    }
    
    _solvers_with_linesearch = ['gd', 'agd']
    _solvers_with_step = ['gd', 'agd', 'svrg', 'sgd']
    _solvers_stochastic = ['sgd', 'svrg', 'sdca']
    _penalties = {
        'none': ProxZero,
        'l1': ProxL1,
        'l2': ProxL2Sq,
        'elasticnet': ProxElasticNet,
        'tv': ProxTV,
        'binarsity': ProxBinarsity
    }

    def __init__(self, penalty='l2', C=1e3, solver="svrg", step=None, tol=1e-5,
                 max_iter=100, verbose=True, warm_start=False, print_every=10,
                 record_every=10, sdca_ridge_strength=1e-3,
                 elastic_net_ratio=0.95, random_state=None, blocks_start=None,
                 blocks_length=None, extra_model_kwargs=None,
                 extra_prox_kwarg=None):

        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}

        # Construct the model
        if extra_model_kwargs is None:
            extra_model_kwargs = {}
        self._model_obj = self._construct_model_obj(**extra_model_kwargs)

        # Construct the solver. The solver is created at creation of the
        # learner, and cannot be instantiated again (using another solver type)
        # afterwards.
        self.solver = solver
        self._set_random_state(random_state)
        self._solver_obj = self._construct_solver_obj(
            solver, step, max_iter, tol, print_every, record_every, verbose,
            sdca_ridge_strength)

        # Construct the prox. The prox is created at creation of the
        # learner, and cannot be instantiated again (using another prox type)
        # afterwards.
        self.penalty = penalty
        if extra_prox_kwarg is None:
            extra_prox_kwarg = {}
        self._prox_obj = self._construct_prox_obj(penalty, elastic_net_ratio,
                                                  blocks_start, blocks_length,
                                                  extra_prox_kwarg)

        # Set C after creating prox to set prox strength
        if 'C' in self._actual_kwargs or penalty != 'none':
            # Print self.C = C
            self.C = C

        self.record_every = record_every
        self.step = step
        self._fitted = False
        self.warm_start = warm_start

        if 'sdca_ridge_strength' in self._actual_kwargs or solver == 'sdca':
            self.sdca_ridge_strength = sdca_ridge_strength

        if 'elastic_net_ratio' in self._actual_kwargs or \
                        penalty == 'elasticnet':
            self.elastic_net_ratio = elastic_net_ratio

        if 'blocks_start' in self._actual_kwargs or penalty == 'binarsity':
            self.blocks_start = blocks_start

        if 'blocks_length' in self._actual_kwargs or penalty == 'binarsity':
            self.blocks_length = blocks_length

    @abstractmethod
    def _construct_model_obj(self, **kwargs):
        pass

    def _construct_solver_obj(self, solver, step, max_iter, tol, print_every,
                              record_every, verbose, sdca_ridge_strength):
        # Parameters of the solver
        #from tick.solver import AGD, GD, BFGS, SGD, SVRG, SDCA
        solvers = {
            'AGD': AGD,
            'BFGS': BFGS,
            'GD': GD,
            'SGD': SGD,
            'SVRG': SVRG,
            'SDCA': SDCA
        }
        solver_args = []
        solver_kwargs = {
            'max_iter': max_iter,
            'tol': tol,
            'print_every': print_every,
            'record_every': record_every,
            'verbose': verbose
        }

        allowed_solvers = list(self._solvers.keys())
        allowed_solvers.sort()
        if solver not in self._solvers:
            raise ValueError("``solver`` must be one of %s, got %s" %
                             (', '.join(allowed_solvers), solver))
        else:
            if solver in self._solvers_with_step:
                solver_kwargs['step'] = step
            if solver in self._solvers_stochastic:
                solver_kwargs['seed'] = self._seed
            if solver == 'sdca':
                solver_args += [sdca_ridge_strength]

            solver_obj = solvers[self._solvers[solver]](*solver_args, **solver_kwargs)

        return solver_obj

    def _construct_prox_obj(self, penalty, elastic_net_ratio, blocks_start,
                            blocks_length, extra_prox_kwarg):
        # Parameters of the penalty
        penalty_args = []

        allowed_penalties = list(self._penalties.keys())
        allowed_penalties.sort()
        if penalty not in allowed_penalties:
            raise ValueError("``penalty`` must be one of %s, got %s" %
                             (', '.join(allowed_penalties), penalty))

        else:
            if penalty != 'none':
                # strength will be set by setting C afterwards
                penalty_args += [0]
            if penalty == 'elasticnet':
                penalty_args += [elastic_net_ratio]
            if penalty == 'binarsity':
                if blocks_start is None:
                    raise ValueError(
                        "Penalty '%s' requires ``blocks_start``, got %s" %
                        (penalty, str(blocks_start)))
                elif blocks_length is None:
                    raise ValueError(
                        "Penalty '%s' requires ``blocks_length``, got %s" %
                        (penalty, str(blocks_length)))
                else:
                    penalty_args += [blocks_start, blocks_length]

            print(f"Penalty: {penalty}")
            print(f"Type of penalty function: {type(self._penalties.get(penalty))}")

            
            prox_obj = self._penalties[penalty](*penalty_args,
                                                **extra_prox_kwarg)

        return prox_obj

    @property
    def warm_start(self):
        return self._warm_start

    @warm_start.setter
    def warm_start(self, val):
        if val is True and self.solver == 'sdca':
            raise ValueError('SDCA cannot be warm started')
        self._warm_start = val

    @property
    def max_iter(self):
        return self._solver_obj.max_iter

    @max_iter.setter
    def max_iter(self, val):
        self._solver_obj.max_iter = val

    @property
    def verbose(self):
        return self._solver_obj.verbose

    @verbose.setter
    def verbose(self, val):
        self._solver_obj.verbose = val

    @property
    def tol(self):
        return self._solver_obj.tol

    @tol.setter
    def tol(self, val):
        self._solver_obj.tol = val

    @property
    def step(self):
        if self.solver in self._solvers_with_step:
            return self._solver_obj.step
        else:
            return None

    @step.setter
    def step(self, val):
        if self.solver in self._solvers_with_step:
            self._solver_obj.step = val
        elif val is not None:
            warn('Solver "%s" has no settable step' % self.solver,
                 RuntimeWarning)

    def _set_random_state(self, val):
        if self.solver in self._solvers_stochastic:
            if val is not None and val < 0:
                raise ValueError(
                    'random_state must be positive, got %s' % str(val))
            self.random_state = val
        else:
            if val is not None:
                warn('Solver "%s" has no settable random_state' % self.solver,
                     RuntimeWarning)
            self.random_state = None

    @property
    def _seed(self):
        if self.solver in self._solvers_stochastic:
            if self.random_state is None:
                return -1
            else:
                return self.random_state
        else:
            warn('Solver "%s" has no _seed' % self.solver, RuntimeWarning)

    @property
    def print_every(self):
        return self._solver_obj.print_every

    @print_every.setter
    def print_every(self, val):
        self._solver_obj.print_every = val

    @property
    def record_every(self):
        return self._solver_obj.record_every

    @record_every.setter
    def record_every(self, val):
        self._solver_obj.record_every = val

    @property
    def C(self):
        if self.penalty == 'none':
            return 0
        elif np.isinf(self._prox_obj.strength):
            return 0
        elif self._prox_obj.strength == 0:
            return None
        else:
            return 1. / self._prox_obj.strength

    @C.setter
    def C(self, val):
        if val is None:
            strength = 0.
        elif val <= 0:
            raise ValueError("``C`` must be positive, got %s" % str(val))
        else:
            strength = 1. / val

        if self.penalty != 'none':
            self._prox_obj.strength = strength
        else:
            if val is not None:
                warn('You cannot set C for penalty "%s"' % self.penalty,
                     RuntimeWarning)

    @property
    def elastic_net_ratio(self):
        if self.penalty == 'elasticnet':
            return self._prox_obj.ratio
        else:
            return None

    @elastic_net_ratio.setter
    def elastic_net_ratio(self, val):
        if self.penalty == 'elasticnet':
            self._prox_obj.ratio = val
        else:
            warn(
                'Penalty "%s" has no elastic_net_ratio attribute' %
                self.penalty, RuntimeWarning)

    @property
    def blocks_start(self):
        if self.penalty == 'binarsity':
            return self._prox_obj.blocks_start
        else:
            return None

    @blocks_start.setter
    def blocks_start(self, val):
        if self.penalty == 'binarsity':
            if type(val) is list:
                val = np.array(val, dtype=np.uint64)
            if val.dtype is not np.uint64:
                val = val.astype(np.uint64)
            self._prox_obj.blocks_start = val
        else:
            warn('Penalty "%s" has no blocks_start attribute' % self.penalty,
                 RuntimeWarning)

    @property
    def blocks_length(self):
        if self.penalty == 'binarsity':
            return self._prox_obj.blocks_length
        else:
            return None

    @blocks_length.setter
    def blocks_length(self, val):
        if self.penalty == 'binarsity':
            if type(val) is list:
                val = np.array(val, dtype=np.uint64)
            if val.dtype is not np.uint64:
                val = val.astype(np.uint64)
            self._prox_obj.blocks_length = val
        else:
            warn('Penalty "%s" has no blocks_length attribute' % self.penalty,
                 RuntimeWarning)

    @property
    def sdca_ridge_strength(self):
        if self.solver == 'sdca':
            return self._solver_obj._solver.get_l_l2sq()
        else:
            return None

    @sdca_ridge_strength.setter
    def sdca_ridge_strength(self, val):
        if self.solver == 'sdca':
            self._solver_obj.l_l2sq = val
        else:
            warn(
                'Solver "%s" has no sdca_ridge_strength attribute' %
                self.solver, RuntimeWarning)

    @staticmethod
    def _safe_array(X, dtype="float64"):
        return safe_array(X, dtype)


class CoxRegression(LearnerOptim):
    """Cox regression learner, using the partial Cox likelihood for
    proportional risks, with many choices of penalization.

    Note that this learner does not have predict functions

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : {'none', 'l1', 'l2', 'elasticnet', 'tv', 'binarsity'}, default='l2'
        The penalization to use. Default is 'l2', namely Ridge penalization

    solver : {'gd', 'agd'}, default='agd'
        The name of the solver to use.

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning. Used when solver is 'gd' or
        'agd'.

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Other Parameters
    ----------------
    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    random_state : int seed, RandomState instance, or None (default)
        The seed that will be used by stochastic solvers. Used in 'sgd',
        'svrg', and 'sdca' solvers

    blocks_start : `numpy.array`, shape=(n_features,), default=None
        The indices of the first column of each binarized feature blocks. It
        corresponds to the ``feature_indices`` property of the
        ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    blocks_length : `numpy.array`, shape=(n_features,), default=None
        The length of each binarized feature blocks. It corresponds to the
        ``n_values`` property of the ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    Attributes
    ----------
    coeffs : np.array, shape=(n_features,)
        The learned coefficients of the model
    """

    _solvers = {'gd': 'GD', 'agd': 'AGD'}

    _attrinfos = {"_actual_kwargs": {"writable": False}}

    @actual_kwargs
    def __init__(self, penalty='l2', C=1e3, solver='agd', step=None, tol=1e-5,
                 max_iter=100, verbose=False, warm_start=False, print_every=10,
                 record_every=10, elastic_net_ratio=0.95, random_state=None,
                 blocks_start=None, blocks_length=None):

        self._actual_kwargs = CoxRegression.__init__.actual_kwargs
        LearnerOptim.__init__(
            self, penalty=penalty, C=C, solver=solver, step=step, tol=tol,
            max_iter=max_iter, verbose=verbose, warm_start=warm_start,
            print_every=print_every, record_every=record_every,
            sdca_ridge_strength=0, elastic_net_ratio=elastic_net_ratio,
            random_state=random_state, blocks_start=blocks_start,
            blocks_length=blocks_length)
        self.coeffs = None

    def _construct_model_obj(self):
        return ModelCoxRegPartialLik()

    def _all_safe(self, features: np.ndarray, times: np.array,
                  censoring: np.array):
        if not set(np.unique(censoring)).issubset({0, 1}):
            raise ValueError('``censoring`` must only have values in {0, 1}')
        # All times must be positive
        if not np.all(times >= 0):
            raise ValueError('``times`` array must contain only non-negative '
                             'entries')
        features = safe_array(features)
        times = safe_array(times)
        censoring = safe_array(censoring, np.ushort)
        return features, times, censoring

    def fit(self, features: np.ndarray, times: np.array, censoring: np.array):
        """Fit the model according to the given training data.

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `CoxRegression`
            The current instance with given data
        """
        # The fit from Model calls the _set_data below

        solver_obj = self._solver_obj
        model_obj = self._model_obj
        prox_obj = self._prox_obj

        features, times, censoring = self._all_safe(features, times, censoring)

        # Pass the data to the model
        model_obj.fit(features, times, censoring)

        if self.step is None and self.solver in self._solvers_with_step:
            if self.solver in self._solvers_with_linesearch:
                self._solver_obj.linesearch = True

        # No intercept in this model
        prox_obj.range = (0, model_obj.n_coeffs)

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        coeffs_start = None
        if self.warm_start and self.coeffs is not None:
            coeffs = self.coeffs
            # ensure starting point has the right format
            if coeffs.shape == (model_obj.n_coeffs,):
                coeffs_start = coeffs

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start)

        # Get the learned coefficients
        self._set("coeffs", coeffs)
        self._set("_fitted", True)
        return self

    def score(self, features=None, times=None, censoring=None):
        """Returns the negative log-likelihood of the model, using the current
        fitted coefficients on the passed data.
        If no data is passed, the negative log-likelihood is computed using the
        data used for training.

        Parameters
        ----------
        features : `None` or `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `None` or `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `None` or `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `float`
            The value of the negative log-likelihood
        """
        if self._fitted:
            all_none = all(e is None for e in [features, times, censoring])
            if all_none:
                return self._model_obj.loss(self.coeffs)
            else:
                if features is None:
                    raise ValueError('Passed ``features`` is None')
                elif times is None:
                    raise ValueError('Passed ``times`` is None')
                elif censoring is None:
                    raise ValueError('Passed ``censoring`` is None')
                else:
                    features, times, censoring = self._all_safe(
                        features, times, censoring)
                    model = ModelCoxRegPartialLik().fit(
                        features, times, censoring)
                    return model.loss(self.coeffs)
        else:
            raise RuntimeError('You must fit the model first')






class ModelCoxRegPartialLik(ModelFirstOrder):
    """Partial likelihood of the Cox regression model (proportional
    hazards).
    This class gives first order information (gradient and loss) for
    this model.

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features), (read-only)
        The features matrix

    times : `numpy.ndarray`, shape = (n_samples,), (read-only)
        Obverved times

    censoring : `numpy.ndarray`, shape = (n_samples,), (read-only)
        Boolean indicator of censoring of each sample.
        ``True`` means true failure, namely non-censored time

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_failures : `int` (read-only)
        Number of true failure times

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    censoring_rate : `float`
        The censoring_rate (percentage of ???)

    Notes
    -----
    There is no intercept in this model
    """

    _attrinfos = {
        "features": {
            "writable": False
        },
        "times": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "n_samples": {
            "writable": False
        },
        "n_features": {
            "writable": False
        },
        "n_failures": {
            "writable": False
        },
        "censoring_rate": {
            "writable": False
        }
    }

    def __init__(self):
        ModelFirstOrder.__init__(self)
        self.features = None
        self.times = None
        self.censoring = None
        self.n_samples = None
        self.n_features = None
        self.n_failures = None
        self.censoring_rate = None
        self._model = None

    def fit(self, features: np.ndarray, times: np.array,
            censoring: np.array) -> Model:
        """Set the data into the model object

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `ModelCoxRegPartialLik`
            The current instance with given data
        """
        # The fit from Model calls the _set_data below
        return Model.fit(self, features, times, censoring)

    def _set_data(self, features: np.ndarray, times: np.array,
                  censoring: np.array):  #

        if self.dtype is None:
            self.dtype = features.dtype
            if self.dtype != times.dtype:
                raise ValueError("Features and labels differ in data types")

        n_samples, n_features = features.shape
        if n_samples != times.shape[0]:
            raise ValueError(("Features has %i samples while times "
                              "have %i" % (n_samples, times.shape[0])))
        if n_samples != censoring.shape[0]:
            raise ValueError(("Features has %i samples while censoring "
                              "have %i" % (n_samples, censoring.shape[0])))

        features = safe_array(features, dtype=self.dtype)
        times = safe_array(times, dtype=self.dtype)
        censoring = safe_array(censoring, np.ushort)

        self._set("features", features)
        self._set("times", times)
        self._set("censoring", censoring)
        self._set("n_samples", n_samples)
        self._set("n_features", n_features)
        self._set(
            "_model", dtype_class_mapper[self.dtype](self.features, self.times,
                                                     self.censoring))

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _get_n_coeffs(self, *args, **kwargs):
        return self.n_features

    @property
    def _epoch_size(self):
        return self.n_failures

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_failures

    def _as_dict(self):
        dd = ModelFirstOrder._as_dict(self)
        del dd["features"]
        del dd["times"]
        del dd["censoring"]
        return dd



class FeaturesBinarizer(Base, BaseEstimator, TransformerMixin):
    """Transforms continuous data into bucketed binary data.

    This is a scikit-learn transformer that transform an input
    pandas DataFrame X of shape (n_samples, n_features) into a binary
    matrix of size (n_samples, n_new_features).
    Continous features are modified and extended into binary features, using
    linearly or inter-quantiles spaced bins.
    Discrete features are binary encoded with K columns, where K is the number
    of modalities.
    Other features (none of the above) are left unchanged.

    Parameters
    ----------
    n_cuts : `int`, default=10
        Number of cut points for continuous features.

    method : "quantile" or "linspace", default="quantile"
        * If ``"quantile"`` quantile-based cuts are used.
        * If ``"linspace"`` linearly spaced cuts are used.
        * If ``"given"`` bins_boundaries needs to be provided.

    detect_column_type : "auto" or "column_names", default="auto"
        * If ``"auto"`` feature type detection done automatically.
        * If ``"column_names"`` feature type detection done using column names.
          In this case names ending by ":continuous" means continuous
          while ":discrete" means a discrete feature

    remove_first : `bool`
        If `True`, first column of each binarized continuous feature block is
        removed.

    bins_boundaries : `list`, default="none"
        Bins boundaries for continuous features.

    Attributes
    ----------
    one_hot_encoder : `OneHotEncoder`
        OneHotEncoders for continuous and discrete features.

    bins_boundaries : `list`
        Bins boundaries for continuous features.

    mapper : `dict`
        Map modalities to column indexes for categorical features.

    feature_type : `dict`
        Features type.

    blocks_start : `list`
        List of indices of the beginning of each block of binarized features

    blocks_length : `list`
        Length of each block of binarized features

    References
    ----------
    http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

    Examples
    --------
    >>> import numpy as np
    >>> from tick.preprocessing import FeaturesBinarizer
    >>> features = np.array([[0.00902084, 0., 'z'],
    ...                      [0.46599565, 0., 2.],
    ...                      [0.52091721, 1., 2.],
    ...                      [0.47315496, 1., 1.],
    ...                      [0.08180209, 0., 0.],
    ...                      [0.45011727, 0., 0.],
    ...                      [2.04347947, 1., 20.],
    ...                      [-0.9890938, 0., 0.],
    ...                      [-0.3063761, 1., 1.],
    ...                      [0.27110903, 0., 0.]])
    >>> binarizer = FeaturesBinarizer(n_cuts=3)
    >>> binarized_features = binarizer.fit_transform(features)
    >>> # output comes as a sparse matrix
    >>> binarized_features.__class__
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> # column type is automatically detected
    >>> sorted(binarizer.feature_type.items())
    [('0', 'continuous'), ('1', 'discrete'), ('2', 'discrete')]
    >>> # features is binarized (first column is removed to avoid colinearity)
    >>> binarized_features.toarray()
    array([[1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
           [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
           [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.]])
    """

    _attrinfos = {
        "one_hot_encoder": {
            "writable": False
        },
        "bins_boundaries": {
            "writable": False
        },
        "mapper": {
            "writable": False
        },
        "feature_type": {
            "writable": False
        },
        "_fitted": {
            "writable": False
        }
    }

    def __init__(self, method="quantile", n_cuts=10, detect_column_type="auto",
                 remove_first=False, bins_boundaries=None):
        Base.__init__(self)

        self.method = method
        self.n_cuts = n_cuts
        self.detect_column_type = detect_column_type
        self.remove_first = remove_first
        self.bins_boundaries = bins_boundaries
        self.reset()

    def reset(self):
        self._set("one_hot_encoder", OneHotEncoder(sparse_output=True))
        self._set("mapper", {})
        self._set("feature_type", {})
        self._set("_fitted", False)
        if self.method != "given":
            self._set("bins_boundaries", {})

    @property
    def boundaries(self):
        """Get bins boundaries for all features.

        Returns
        -------
        output : `dict`
            The bins boundaries for each feature.
        """
        if not self._fitted:
            raise ValueError("cannot get bins_boundaries if object has not "
                             "been fitted")
        return self.bins_boundaries

    @property
    def blocks_start(self):
        """Get the first column indices of each binarized feature blocks.

        Returns
        -------
        output : `np.ndarray`
            The indices of the first column of each binarized feature blocks.
        """
        if not self._fitted:
            raise ValueError("cannot get blocks_start if object has not "
                             "been fitted")
        # construct from encoder
        return self._get_feature_indices()[:-1,]

    @property
    def blocks_length(self):
        """Get the length of each binarized feature blocks.

        Returns
        -------
        output : `np.ndarray`
            The length of each binarized feature blocks.
        """
        if not self._fitted:
            raise ValueError("cannot get blocks_length if object has not been "
                             "fitted")
        # construct from encoder
        return self._get_n_values()

    @staticmethod
    def cast_to_array(X):
        """Cast input matrix to `np.ndarray`.

        Returns
        -------
        output : `np.ndarray`, `np.ndarray`
            The input matrix and the corresponding column names.
        """
        if X.__class__ == pd.DataFrame:
            columns = X.columns
            X = X.values
        else:
            columns = [str(i) for i in range(X.shape[1])]

        return X, columns

    def fit(self, X):
        """Fit the binarization using the features matrix.

        Parameters
        ----------
        X : `pd.DataFrame`  or `np.ndarray`, shape=(n_samples, n_features)
            The features matrix.

        Returns
        -------
        output : `FeaturesBinarizer`
            The fitted current instance.
        """
        self.reset()
        X, columns = FeaturesBinarizer.cast_to_array(X)
        categorical_X = np.empty_like(X)
        for i, column in enumerate(columns):
            feature = X[:, i]
            binarized_feat = self._assign_interval(column, feature, fit=True)
            categorical_X[:, i] = binarized_feat

        self.one_hot_encoder.fit(categorical_X)

        self._set("_fitted", True)
        return self

    def transform(self, X):
        """Apply the binarization to the given features matrix.

        Parameters
        ----------
        X : `pd.DataFrame` or `np.ndarray`, shape=(n_samples, n_features)
            The features matrix.

        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized.
        """
        X, columns = FeaturesBinarizer.cast_to_array(X)

        categorical_X = np.empty_like(X)
        for i, column in enumerate(columns):
            feature = X[:, i]
            binarized_feat = self._assign_interval(columns[i], feature,
                                                   fit=False)
            categorical_X[:, i] = binarized_feat

        binarized_X = self.one_hot_encoder.transform(categorical_X)

        if self.remove_first:
            feature_indices = self._get_feature_indices()
            mask = np.ones(binarized_X.shape[1], dtype=bool)
            mask[feature_indices[:-1]] = False
            binarized_X = binarized_X[:, mask]

        return binarized_X

    def fit_transform(self, X, y=None, **kwargs):
        """Fit and apply the binarization using the features matrix.

        Parameters
        ----------
        X : `pd.DataFrame` or `np.ndarray`, shape=(n_samples, n_features)
            The features matrix.

        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized.
        """
        self.fit(X)
        binarized_X = self.transform(X)

        return binarized_X

    @staticmethod
    def _detect_feature_type(feature, detect_column_type="auto",
                             feature_name=None, continuous_threshold="auto"):
        """Detect the type of a single feature.

        Parameters
        ----------
        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature

        detect_column_type : "auto" or "column_names", default="auto"
            * If ``"auto"`` an automatic type detection procedure is followed.
            * If ``"column_names"`` columns with name ending with
            ":continuous" means continuous features and columns with name ending
            with ":discrete" means discrete features

        feature_name : `str`
            The name of the feature

        continuous_threshold : `int` or `str`, default "auto"
            If "auto", we consider the feature as "discrete" if the feature
            gets more than `threshold`=15 distinct values (if there are more
            than 30 examples, else `threshold` is set to half the number of
            examples).
            If a number is given, then we consider the feature as "discrete" if
            the feature has more distinct values than this number

        Returns
        -------
        output : `str`
            The type of the feature (either `continuous` or `discrete`).
        """
        if detect_column_type == "column_names":
            if feature_name is None:
                raise ValueError("feature_name must be set in order to use"
                                 "'column_names' detection type")

            if feature_name.endswith(":continuous"):
                feature_type = "continuous"
            elif feature_name.endswith(":discrete"):
                feature_type = "discrete"
            else:
                raise ValueError("feature name '%s' should end with "
                                 "':continuous' or ':discrete'" % feature_name)

        elif detect_column_type == "auto":
            if continuous_threshold == "auto":
                # threshold choice depending on whether one has more than 30
                # examples or not
                if len(feature) > 30:
                    threshold = 15
                else:
                    threshold = len(feature) / 2
            else:
                threshold = continuous_threshold

            # count distinct realizations and compare to threshold
            uniques = np.unique(feature)
            n_uniques = len(uniques)
            if n_uniques > threshold:
                # feature_type is `continuous` only is all feature values are
                # convertible to float
                try:
                    uniques.astype(float)
                    feature_type = "continuous"
                except ValueError:
                    feature_type = "discrete"
            else:
                feature_type = "discrete"

        else:
            raise ValueError("detect_type should be one of 'column_names' or "
                             "'auto'" % detect_column_type)

        return feature_type

    def _get_feature_type(self, feature_name, feature, fit=False):
        """Get the type of a single feature.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature

        fit : `bool`
            If `True`, we save the feature type.
            If `False`, we take the corresponding saved feature type.

        Returns
        -------
        output : `str`
            The type of the feature (either `continuous` or `discrete`).
        """
        if fit:
            feature_type = FeaturesBinarizer._detect_feature_type(
                feature, feature_name=feature_name,
                detect_column_type=self.detect_column_type)
            self.feature_type[feature_name] = feature_type

        elif self._fitted:
            feature_type = self.feature_type[feature_name]
        else:
            raise ValueError("cannot call method with fit=True if object "
                             "has not been fitted")

        return feature_type

    @staticmethod
    def _detect_boundaries(feature, n_cuts, method):
        """Boundaries detection of a single feature.

        Parameters
        ----------
        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature

        n_cuts : `int`
            Number of cut points

        method : `str`
            If `quantile`, we use quantiles to construct the intervals.
            If `linspace`, we construct linearly spaced intervals

        Returns
        -------
        output : `np.ndarray`
           The intervals boundaries for the feature.
        """
        if not isinstance(feature.dtype, (int, float)):
            feature = feature.astype(float)

        if method == 'quantile':
            quantile_cuts = np.linspace(0, 100, n_cuts + 2)
            boundaries = np.percentile(feature, quantile_cuts,
                                       interpolation="nearest")
            # Only keep distinct bins boundaries
            boundaries = np.unique(boundaries)
        elif method == 'linspace':
            # Maximum and minimum of the feature
            feat_max = np.max(feature)
            feat_min = np.min(feature)
            # Compute the cuts
            boundaries = np.linspace(feat_min, feat_max, n_cuts + 2)
        else:
            raise ValueError(
                "Method '%s' should be 'quantile' or 'linspace'" % method)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

        return boundaries

    def _get_boundaries(self, feature_name, feature, fit=False):
        """Get bins boundaries of a single continuous feature.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature to be binarized

        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature

        Returns
        -------
        output : `np.ndarray`, shape=(?,)
            The bins boundaries. The number of lines is smaller or
            equal to ``n_cuts``, depending on the ``method`` and/or on
            the actual number of distinct boundaries for this feature.
        """
        if fit:
            if self.method == 'given':
                if self.bins_boundaries is None:
                    raise ValueError("bins_boundaries required when `method` "
                                     "equals 'given'")

                if not isinstance(self.bins_boundaries[feature_name], np.ndarray):
                    raise ValueError("feature %s not found in bins_boundaries" % feature_name)
                boundaries = self.bins_boundaries[feature_name]
            else:
                boundaries = FeaturesBinarizer._detect_boundaries(
                    feature, self.n_cuts, self.method)
                self.bins_boundaries[feature_name] = boundaries
        elif self._fitted:
            boundaries = self.bins_boundaries[feature_name]
        else:
            raise ValueError("cannot call method with fit=True as object has "
                             "not been fit")
        return boundaries

    def _categorical_to_interval(self, feature, feature_name, fit=False):
        """Assign intervals to a single feature considered as `discrete`.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature. Could contain `str` values

        fit : `bool`
            If `True`, we need to fit (compute indexes) for this feature

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            The discretized feature.
        """
        if fit:
            uniques = np.unique(feature)
            uniques.sort()

            mapper = {
                category: interval
                for interval, category in enumerate(uniques)
            }

            self.mapper[feature_name] = mapper

        else:
            mapper = self.mapper[feature_name]

        def category_to_interval(category):
            if category in mapper:
                return mapper.get(category)
            else:
                return len(list(mapper.keys())) + 1

        return np.vectorize(category_to_interval)(feature)

    def _assign_interval(self, feature_name, feature, fit=False):
        """Assign intervals to a single feature.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature to be binarized

        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            The discretized feature.
        """
        feature_type = self._get_feature_type(feature_name, feature, fit)

        if feature_type == "continuous":
            if feature.dtype != float:
                feature = feature.astype(float)

            # Get bins boundaries for the feature
            boundaries = self._get_boundaries(feature_name, feature, fit)

            # Discretize feature
            feature = pd.cut(feature, boundaries, labels=False)

        else:
            feature = self._categorical_to_interval(feature, feature_name,
                                                    fit=fit)
        return feature

    def _is_sklearn_older_than(self, ver):
        from packaging import version
        import sklearn
        return version.parse(sklearn.__version__) < version.parse(ver)

    def _get_n_values(self):
        if self._is_sklearn_older_than("0.22.0"):
            return self.one_hot_encoder.n_values_
        else:
            return [len(x) for x in self.one_hot_encoder.categories_]

    def _get_feature_indices(self):
        if self._is_sklearn_older_than("0.22.0"):
            return self.one_hot_encoder.feature_indices_
        else:
            feature_indices = [0]
            for cat in self.one_hot_encoder.categories_:
                feature_indices.append(feature_indices[-1] + len(cat))
            return np.asarray(feature_indices)

def features_normal_cov_uniform(n_samples: int = 200, n_features: int = 30,
                                dtype="float64"):
    """Normal features generator with uniform covariance

    An example of features obtained as samples of a centered Gaussian
    vector with a specific covariance matrix given by 0.5 * (U + U.T),
    where U is uniform on [0, 1] and diagonal filled by ones.

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance
    """
    C = np.random.uniform(size=(n_features, n_features), dtype=dtype)
    np.fill_diagonal(C, 1.0)
    cov = 0.5 * (C + C.T)
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


def features_normal_cov_toeplitz(n_samples: int = 200, n_features: int = 30,
                                 cov_corr: float = 0.5, dtype="float64"):
    """Normal features generator with toeplitz covariance

    An example of features obtained as samples of a centered Gaussian
    vector with a toeplitz covariance matrix

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    cov_corr : `float`, default=0.5
        correlation coefficient of the Toeplitz correlation matrix

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance

    """
    cov = toeplitz(cov_corr ** np.arange(0, n_features))
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


class Simu(ABC, Base):
    """
    Abstract simulation class. It does nothing besides printing and
    verbosing.

    Parameters
    ----------
    seed : `int`
        The seed of the random number generator

    verbose : `bool`
        If True, print things

    Attributes
    ----------
    time_start : `str`
        Start date of the simulation

    time_elapsed : `int`
        Duration of the simulation, in seconds

    time_end : `str`
        End date of the simulation
    """

    _attrinfos = {
        "time_start": {
            "writable": False
        },
        "time_elapsed": {
            "writable": False
        },
        "time_end": {
            "writable": False
        },
        "_time_start": {
            "writable": False
        }
    }

    def __init__(self, seed: int = None, verbose: bool = True):
        Base.__init__(self)
        self.seed = seed
        self.verbose = verbose
        if seed is not None and seed >= 0:
            self._set_seed()
        self._set("time_start", None)
        self._set("time_elapsed", None)
        self._set("time_end", None)
        self._set("_time_start", None)

    def _set_seed(self):
        np.random.seed(self.seed)

    def _start_simulation(self):
        self._set("time_start", self._get_now())
        self._set("_time_start", time())
        if self.verbose:
            msg = "Launching simulation using {class_}..." \
                    .format(class_=self.name)
            print("-" * len(msg))
            print(msg)

    def _end_simulation(self):
        self._set("time_end", self._get_now())
        t = time()
        self._set("time_elapsed", t - self._time_start)
        if self.verbose:
            msg = "Done simulating using {class_} in {time:.2e} " \
                  "seconds." \
                .format(class_=self.name, time=self.time_elapsed)
            print(msg)

    @abstractmethod
    def _simulate(self):
        pass

    def simulate(self):
        """Launch the simulation of data
        """
        self._start_simulation()
        result = self._simulate()
        self._end_simulation()
        return result

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("coeffs", None)
        return dd

class SimuWithFeatures(Simu):
    """Abstract class for the simulation of a model with a features
    matrix.

    Parameters
    ----------
    intercept : `float`, default=`None`
        The intercept. If None, then no intercept is used

    features : `numpy.ndarray`, shape=(n_samples, n_features), default=`None`
        The features matrix to use. If None, it is simulated

    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    features_type : `str`, default="cov_toeplitz"
        The type of features matrix to simulate

        * If ``"cov_toeplitz"`` : a Gaussian distribution with
          Toeplitz correlation matrix

        * If ``"cov_uniform"`` : a Gaussian distribution with
          correlation matrix given by O.5 * (U + U.T), where U is
          uniform on [0, 1] and diagonal filled with ones.

    cov_corr : `float`, default=0.5
        Correlation to use in the Toeplitz correlation matrix

    features_scaling : `str`, default="none"
        The way the features matrix is scaled after simulation

        * If ``"standard"`` : the columns are centered and
          normalized

        * If ``"min-max"`` : remove the minimum and divide by
          max-min

        * If ``"norm"`` : the columns are normalized but not centered

        * If ``"none"`` : nothing is done to the features

    seed : `int`
        The seed of the random number generator

    verbose : `bool`
        If True, print things

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the generated arrays.
        Used in the case features is None
    """

    _attrinfos = {
        "_features_type": {
            "writable": False
        },
        "_features_scaling": {
            "writable": False
        }
    }

    def __init__(self, intercept: float = None, features: np.ndarray = None,
                 n_samples: int = 200, n_features: int = 30,
                 features_type: str = "cov_toeplitz", cov_corr: float = 0.5,
                 features_scaling: str = "none", seed: int = None,
                 verbose: bool = True, dtype="float64"):

        Simu.__init__(self, seed, verbose)
        self.intercept = intercept
        self.features = features
        self.n_samples = n_samples
        self.n_features = n_features
        self.features_type = features_type
        self.cov_corr = cov_corr
        self.features_scaling = features_scaling
        self.features = None
        self.dtype = dtype

        if features is not None:
            if n_features != features.shape[1]:
                raise ValueError("``n_features`` does not match size of"
                                 "``features``")
            if n_samples != features.shape[0]:
                raise ValueError("``n_samples`` does not match size of"
                                 "``features``")
            features_type = 'given'

            self.features = features
            n_samples, n_features = features.shape
            self.n_samples = n_samples
            self.n_features = n_features
            self.features_type = features_type
            self.dtype = self.features.dtype

        # TODO: check and correct also n_samples, n_features and cov_corr and features_scaling

    def _scale_features(self, features: np.ndarray):
        features_scaling = self.features_scaling
        if features_scaling == "standard":
            features -= features.mean(axis=0)
            features /= features.std(axis=0)
        elif features_scaling == "min-max":
            raise NotImplementedError()
        elif features_scaling == "norm":
            raise NotImplementedError()
        return features

    @property
    def features_type(self):
        return self._features_type

    @features_type.setter
    def features_type(self, val):
        if val not in ["given", "cov_toeplitz", "cov_uniform"]:
            warn(linesep + "features_type was not understood, using" +
                 " cov_toeplitz instead.")
            val = "cov_toeplitz"
        self._set("_features_type", val)

    @property
    def features_scaling(self):
        return self._features_scaling

    @features_scaling.setter
    def features_scaling(self, val):
        if val not in ["standard", "min-max", "norm", "none"]:
            warn(linesep + "features_scaling was not understood, " +
                 "using ``'none'`` instead.")
            val = "none"
        self._set("_features_scaling", val)

    def simulate(self):
        """Launch the simulation of data.
        """
        self._start_simulation()
        features_type = self.features_type
        if features_type != "given":
            n_samples = self.n_samples
            n_features = self.n_features
            if features_type == "cov_uniform":
                features = features_normal_cov_uniform(n_samples, n_features,
                                                       dtype=self.dtype)
            else:
                cov_corr = self.cov_corr
                features = features_normal_cov_toeplitz(
                    n_samples, n_features, cov_corr, dtype=self.dtype)
        else:
            features = self.features

        features = self._scale_features(features)
        self.features = features

        # Launch the overloaded simulation
        result = self._simulate()

        self._end_simulation()
        # self._set("data", result)
        return result

    def _as_dict(self):
        dd = Simu._as_dict(self)
        dd.pop("features", None)
        dd.pop("labels", None)
        return dd

class SimuCoxReg(SimuWithFeatures):
    """Simulation of a Cox regression for proportional hazards

    Parameters
    ----------
    coeffs : `numpy.ndarray`, shape=(n_coeffs,)
        The array of coefficients of the model

    features : `numpy.ndarray`, shape=(n_samples, n_features), default=`None`
        The features matrix to use. If None, it is simulated

    n_samples : `int`, default=200
        Number of samples

    times_distribution : `str`, default="weibull"
        The distrubution of times. Only ``"weibull"``
        is implemented for now

    scale : `float`, default=1.0
        Scaling parameter to use in the distribution of times

    shape : `float`, default=1.0
        Shape parameter to use in the distribution of times

    censoring_factor : `float`, default=2.0
        Level of censoring. Increasing censoring_factor leads
        to less censored times and conversely.

    features_type : `str`, default="cov_toeplitz"
        The type of features matrix to simulate

        * If ``"cov_toeplitz"`` : a Gaussian distribution with
          Toeplitz correlation matrix

        * If ``"cov_uniform"`` : a Gaussian distribution with
          correlation matrix given by O.5 * (U + U.T), where U is
          uniform on [0, 1] and diagonal filled with ones.

    cov_corr : `float`, default=0.5
        Correlation to use in the Toeplitz correlation matrix

    features_scaling : `str`, default="none"
        The way the features matrix is scaled after simulation

        * If ``"standard"`` : the columns are centered and
          normalized

        * If ``"min-max"`` : remove the minimum and divide by
          max-min

        * If ``"norm"`` : the columns are normalized but not centered

        * If ``"none"`` : nothing is done to the features

    seed : `int`, default=None
        The seed of the random number generator. If `None` it is not
        seeded

    verbose : `bool`, default=True
        If True, print things

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated (or given) features matrix

    times : `numpy.ndarray`, shape=(n_samples,)
        Simulated times

    censoring : `numpy.ndarray`, shape=(n_samples,)
        Simulated censoring indicator, where ``censoring[i] == 1``
        indicates that the time of the i-th individual is a failure
        time, and where ``censoring[i] == 0`` means that the time of
        the i-th individual is a censoring time

    time_start : `str`
        Start date of the simulation

    time_elapsed : `int`
        Duration of the simulation, in seconds

    time_end : `str`
        End date of the simulation

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the generated arrays.
        Used in the case features is None

    Notes
    -----
    There is no intercept in this model
    """

    _attrinfos = {
        "times": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "_times_distribution": {
            "writable": False
        },
        "_scale": {
            "writable": False
        },
        "_shape": {
            "writable": False
        }
    }

    def __init__(self, coeffs: np.ndarray,
                 features: np.ndarray = None, n_samples: int = 200,
                 times_distribution: str = "weibull",
                 shape: float = 1., scale: float = 1.,
                 censoring_factor: float = 2.,
                 features_type: str = "cov_toeplitz",
                 cov_corr: float = 0.5, features_scaling: str = "none",
                 seed: int = None, verbose: bool = True, dtype="float64"):

        n_features = coeffs.shape[0]
        # intercept=None in this model
        SimuWithFeatures.__init__(self, None, features, n_samples,
                                  n_features, features_type, cov_corr,
                                  features_scaling, seed, verbose, dtype=dtype)
        self.coeffs = coeffs
        self.shape = shape
        self.scale = scale
        self.censoring_factor = censoring_factor
        self.times_distribution = times_distribution
        self.features = None
        self.times = None
        self.censoring = None

    def simulate(self):
        """Launch simulation of the data

        Returns
        -------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The simulated (or given) features matrix

        times : `numpy.ndarray`, shape=(n_samples,)
            Simulated times

        censoring : `numpy.ndarray`, shape=(n_samples,)
            Simulated censoring indicator, where ``censoring[i] == 1``
            indicates that the time of the i-th individual is a failure
            time, and where ``censoring[i] == 0`` means that the time of
            the i-th individual is a censoring time
        """
        return SimuWithFeatures.simulate(self)

    @property
    def times_distribution(self):
        return self._times_distribution

    @times_distribution.setter
    def times_distribution(self, val):
        if val != "weibull":
            raise ValueError("``times_distribution`` was not "
                             "understood, try using 'weibull' instead")
        self._set("_times_distribution", val)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if val <= 0:
            raise ValueError("``shape`` must be strictly positive")
        self._set("_shape", val)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        if val <= 0:
            raise ValueError("``scale`` must be strictly positive")
        self._set("_scale", val)

    def _simulate(self):
        # The features matrix already exists, and is created by the
        # super class
        features = self.features
        n_samples, n_features = features.shape
        u = features.dot(self.coeffs)
        # Simulation of true times
        E = np.random.exponential(scale=1., size=n_samples)
        E *= np.exp(-u)
        scale = self.scale
        shape = self.shape
        if self.times_distribution == "weibull":
            T = 1. / scale * E ** (1. / shape)
        else:
            # There is not point in this test, but let's do it like that
            # since we're likely to implement other distributions
            T = 1. / scale * E ** (1. / shape)
        m = T.mean()
        # Simulation of the censoring
        c = self.censoring_factor
        C = np.random.exponential(scale=c * m, size=n_samples)
        # Observed time
        self._set("times", np.minimum(T, C).astype(self.dtype))
        # Censoring indicator: 1 if it is a time of failure, 0 if it's
        #   censoring. It is as int8 and not bool as we might need to
        #   construct a memory access on it later
        censoring = (T <= C).astype(np.ushort)
        self._set("censoring", censoring)
        return self.features, self.times, self.censoring

    def _as_dict(self):
        dd = SimuWithFeatures._as_dict(self)
        dd.pop("features", None)
        dd.pop("times", None)
        dd.pop("censoring", None)
        return dd


class SimuCoxRegWithCutPoints(SimuWithFeatures):
    """Simulation of a Cox regression for proportional hazards with cut-points
    effects in the features

    Parameters
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features), default=`None`
        The features matrix to use. If None, it is simulated

    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=5
        Number of features

    times_distribution : `str`, default="weibull"
        The distrubution of times. Only ``"weibull"``
        is implemented for now

    scale : `float`, default=1.0
        Scaling parameter to use in the distribution of times

    shape : `float`, default=1.0
        Shape parameter to use in the distribution of times

    censoring_factor : `float`, default=2.0
        Level of censoring. Increasing censoring_factor leads
        to less censored times and conversely.

    features_type : `str`, default="cov_toeplitz"
        The type of features matrix to simulate

        * If ``"cov_toeplitz"`` : a Gaussian distribution with
          Toeplitz correlation matrix

        * If ``"cov_uniform"`` : a Gaussian distribution with
          correlation matrix given by O.5 * (U + U.T), where U is
          uniform on [0, 1] and diagonal filled with ones.

    cov_corr : `float`, default=0.5
        Correlation to use in the Toeplitz correlation matrix

    features_scaling : `str`, default="none"
        The way the features matrix is scaled after simulation

        * If ``"standard"`` : the columns are centered and
          normalized

        * If ``"min-max"`` : remove the minimum and divide by
          max-min

        * If ``"norm"`` : the columns are normalized but not centered

        * If ``"none"`` : nothing is done to the features

    seed : `int`, default=None
        The seed of the random number generator. If `None` it is not
        seeded

    verbose : `bool`, default=True
        If True, print things

    n_cut_points : `int`, default="none"
        Number of cut-points generated per feature. If `None` it is sampled from
        a geometric distribution of parameter n_cut_points_factor.

    n_cut_points_factor : `float`, default=0.7
        Parameter of the geometric distribution used to generate the number of
        cut-points when n_cut_points is `None`. Increasing n_cut_points_factor
        leads to less cut-points per feature on average.

    sparsity : `float`, default=0
        Percentage of block sparsity induced in the coefficient vector. Must be
        in [0, 1].

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated (or given) features matrix

    times : `numpy.ndarray`, shape=(n_samples,)
        Simulated times

    censoring : `numpy.ndarray`, shape=(n_samples,)
        Simulated censoring indicator, where ``censoring[i] == 1``
        indicates that the time of the i-th individual is a failure
        time, and where ``censoring[i] == 0`` means that the time of
        the i-th individual is a censoring time

    Notes
    -----
    There is no intercept in this model
    """

    _attrinfos = {
        "times": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "_times_distribution": {
            "writable": False
        },
        "_scale": {
            "writable": False
        },
        "_shape": {
            "writable": False
        },
        "_sparsity": {
            "writable": False
        }
    }

    def __init__(self, features: np.ndarray = None, n_samples: int = 200,
                 n_features: int = 5, n_cut_points: int = None,
                 n_cut_points_factor: float = .7,
                 times_distribution: str = "weibull",
                 shape: float = 1., scale: float = 1.,
                 censoring_factor: float = 2.,
                 features_type: str = "cov_toeplitz",
                 cov_corr: float = 0.5, features_scaling: str = "none",
                 seed: int = None, verbose: bool = True, sparsity=0):

        # intercept=None in this model
        SimuWithFeatures.__init__(self, None, features, n_samples,
                                  n_features, features_type, cov_corr,
                                  features_scaling, seed, verbose)

        self.shape = shape
        self.scale = scale
        self.censoring_factor = censoring_factor
        self.times_distribution = times_distribution
        self.n_cut_points = n_cut_points
        self.n_cut_points_factor = n_cut_points_factor
        self.sparsity = sparsity
        self.features = None
        self.times = None
        self.censoring = None

    def simulate(self):
        """Launch simulation of the data

        Returns
        -------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The simulated (or given) features matrix

        times : `numpy.ndarray`, shape=(n_samples,)
            Simulated times

        censoring : `numpy.ndarray`, shape=(n_samples,)
            Simulated censoring indicator, where ``censoring[i] == 1``
            indicates that the time of the i-th individual is a failure
            time, and where ``censoring[i] == 0`` means that the time of
            the i-th individual is a censoring time
        """
        return SimuWithFeatures.simulate(self)

    @property
    def times_distribution(self):
        return self._times_distribution

    @times_distribution.setter
    def times_distribution(self, val):
        if val != "weibull":
            raise ValueError("``times_distribution`` was not "
                             "understood, try using 'weibull' instead")
        self._set("_times_distribution", val)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if val <= 0:
            raise ValueError("``shape`` must be strictly positive")
        self._set("_shape", val)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        if val <= 0:
            raise ValueError("``scale`` must be strictly positive")
        self._set("_scale", val)

    @property
    def sparsity(self):
        return self._sparsity

    @sparsity.setter
    def sparsity(self, val):
        if not 0 <= val <= 1:
            raise ValueError("``sparsity`` must be in (0, 1)")
        self._set("_sparsity", val)

    def _simulate(self):
        # The features matrix already exists, and is created by the
        # super class
        features = self.features
        n_samples, n_features = features.shape
        # Simulation of cut-points
        n_cut_points = self.n_cut_points
        n_cut_points_factor = self.n_cut_points_factor
        sparsity = self.sparsity
        s = round(n_features * sparsity)
        # sparsity index set
        S = np.random.choice(n_features, s, replace=False)

        if n_cut_points is None:
            n_cut_points = np.random.geometric(n_cut_points_factor, n_features)
        else:
            n_cut_points = np.repeat(n_cut_points, n_features)

        cut_points = {}
        coeffs_binarized = np.array([])
        for j in range(n_features):
            feature_j = features[:, j]
            quantile_cuts = np.linspace(10, 90, 10)
            candidates = np.percentile(feature_j, quantile_cuts,
                                       interpolation="nearest")
            cut_points_j = np.random.choice(candidates, n_cut_points[j],
                                            replace=False)
            cut_points_j = np.sort(cut_points_j)
            cut_points_j = np.insert(cut_points_j, 0, -np.inf)
            cut_points_j = np.append(cut_points_j, np.inf)
            cut_points[str(j)] = cut_points_j
            # generate beta star
            if j in S:
                coeffs_block = np.zeros(n_cut_points[j] + 1)
            else:
                coeffs_block = np.random.normal(1, .5, n_cut_points[j] + 1)
                # make sure 2 consecutive coeffs are different enough
                coeffs_block = np.abs(coeffs_block)
                coeffs_block[::2] *= -1
            # sum-to-zero constraint in each block
            coeffs_block = coeffs_block - coeffs_block.mean()
            coeffs_binarized = np.append(coeffs_binarized, coeffs_block)

        binarizer = FeaturesBinarizer(method='given',
                                      bins_boundaries=cut_points)
        binarized_features = binarizer.fit_transform(features)

        u = binarized_features.dot(coeffs_binarized)
        # Simulation of true times
        E = np.random.exponential(scale=1., size=n_samples)
        E *= np.exp(-u)
        scale = self.scale
        shape = self.shape
        if self.times_distribution == "weibull":
            T = 1. / scale * E ** (1. / shape)
        else:
            # There is not point in this test, but let's do it like that
            # since we're likely to implement other distributions
            T = 1. / scale * E ** (1. / shape)

        m = T.mean()
        # Simulation of the censoring
        c = self.censoring_factor
        C = np.random.exponential(scale=c * m, size=n_samples)
        # Observed time
        self._set("times", np.minimum(T, C).astype(self.dtype))
        # Censoring indicator: 1 if it is a time of failure, 0 if censoring.
        censoring = (T <= C).astype(np.ushort)
        self._set("censoring", censoring)
        return self.features, self.times, self.censoring, cut_points, \
               coeffs_binarized, S

    def _as_dict(self):
        dd = SimuWithFeatures._as_dict(self)
        dd.pop("features", None)
        dd.pop("times", None)
        dd.pop("censoring", None)
        return dd


def compute_score(features, features_binarized, times, censoring,
                  blocks_start, blocks_length, boundaries, C=10, n_folds=10,
                  features_names=None, shuffle=True, n_jobs=1, verbose=False,
                  validation_data=None):
    scores = cross_val_score(features, features_binarized, times,
                             censoring, blocks_start, blocks_length, boundaries,
                             n_folds=n_folds, shuffle=shuffle, C=C,
                             features_names=features_names, n_jobs=n_jobs,
                             verbose=verbose, validation_data=validation_data)
    scores_test = scores[:, 0]
    scores_validation = scores[:, 1]
    if validation_data is not None:
        scores_validation_mean = scores_validation.mean()
        scores_validation_std = scores_validation.std()
    else:
        scores_validation_mean, scores_validation_std = None, None

    scores_mean = scores_test.mean()
    scores_std = scores_test.std()
    if verbose:
        print("\nscore %0.3f (+/- %0.3f)" % (scores_mean, scores_std))
    scores = [scores_mean, scores_std, scores_validation_mean,
              scores_validation_std]
    return scores


def cross_val_score(features, features_binarized, times, censoring,
                    blocks_start, blocks_length, boundaries, n_folds, shuffle,
                    C, features_names, n_jobs, verbose, validation_data):
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    cv_iter = list(cv.split(features))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    scores = parallel(
        delayed(fit_and_score)(features, features_binarized, times,
                               censoring, blocks_start, blocks_length,
                               boundaries, features_names, idx_train, idx_test,
                               validation_data, C)
        for (idx_train, idx_test) in cv_iter)
    return np.array(scores)


def fit_and_score(features, features_bin, times, censoring,
                  blocks_start, blocks_length, boundaries, features_names,
                  idx_train, idx_test, validation_data, C):
    if features_names is None:
        features_names = [str(j) for j in range(features.shape[1])]
    X_train, X_test = features_bin[idx_train], features_bin[idx_test]
    Y_train, Y_test = times[idx_train], times[idx_test]
    delta_train, delta_test = censoring[idx_train], censoring[idx_test]

    learner = CoxRegression(penalty='binarsity', tol=1e-5,
                            verbose=False, max_iter=100, step=0.3,
                            blocks_start=blocks_start,
                            blocks_length=blocks_length,
                            warm_start=True)
    learner._solver_obj.linesearch = False
    learner.C = C
    learner.fit(X_train, Y_train, delta_train)
    coeffs = learner.coeffs

    cut_points_estimates = {}
    for j, start in enumerate(blocks_start):
        coeffs_j = coeffs[start:start + blocks_length[j]]
        all_zeros = not np.any(coeffs_j)
        if all_zeros:
            cut_points_estimate_j = np.array([-np.inf, np.inf])
        else:
            groups_j = get_groups(coeffs_j)
            jump_j = np.where(groups_j[1:] - groups_j[:-1] != 0)[0] + 1
            if jump_j.size == 0:
                cut_points_estimate_j = np.array([-np.inf, np.inf])
            else:
                cut_points_estimate_j = boundaries[features_names[j]][
                    jump_j]
                if cut_points_estimate_j[0] != -np.inf:
                    cut_points_estimate_j = np.insert(cut_points_estimate_j,
                                                      0, -np.inf)
                if cut_points_estimate_j[-1] != np.inf:
                    cut_points_estimate_j = np.append(cut_points_estimate_j,
                                                      np.inf)
        cut_points_estimates[features_names[j]] = cut_points_estimate_j
    binarizer = FeaturesBinarizer(method='given',
                                  bins_boundaries=cut_points_estimates)
    binarized_features = binarizer.fit_transform(features)
    blocks_start = binarizer.blocks_start
    blocks_length = binarizer.blocks_length
    X_bin_train = binarized_features[idx_train]
    X_bin_test = binarized_features[idx_test]
    learner_ = CoxRegression(penalty='binarsity', tol=1e-5,
                             verbose=False, max_iter=100, step=0.3,
                             blocks_start=blocks_start,
                             blocks_length=blocks_length,
                             warm_start=True, C=1e10)
    learner_._solver_obj.linesearch = False
    learner_.fit(X_bin_train, Y_train, delta_train)
    score = learner_.score(X_bin_test, Y_test, delta_test)

    if validation_data is not None:
        X_validation = validation_data[0]
        X_bin_validation = binarizer.fit_transform(X_validation)
        Y_validation = validation_data[1]
        delta_validation = validation_data[2]
        score_validation = learner_.score(X_bin_validation, Y_validation,
                                          delta_validation)
    else:
        score_validation = None

    return score, score_validation


def get_groups(coeffs):
    n_coeffs = len(coeffs)
    jumps = np.where(coeffs[1:] - coeffs[:-1] != 0)[0] + 1
    jumps = np.insert(jumps, 0, 0)
    jumps = np.append(jumps, n_coeffs)
    groups = np.zeros(n_coeffs)
    for i in range(len(jumps) - 1):
        groups[jumps[i]:jumps[i + 1]] = i
        if jumps[i + 1] - jumps[i] <= 2:
            if i == 0:
                groups[jumps[i]:jumps[i + 1]] = 1
            elif i == len(jumps) - 2:
                groups[jumps[i]:jumps[i + 1]] = groups[jumps[i - 1]]
            else:
                coeff_value = coeffs[jumps[i]]
                group_before = groups[jumps[i - 1]]
                coeff_value_before = coeffs[
                    np.nonzero(groups == group_before)[0][0]]
                try:
                    k = 0
                    while coeffs[jumps[i + 1] + k] != coeffs[
                                        jumps[i + 1] + k + 1]:
                        k += 1
                    coeff_value_after = coeffs[jumps[i + 1] + k]
                except:
                    coeff_value_after = coeffs[jumps[i + 1]]
                if np.abs(coeff_value_before - coeff_value) < np.abs(
                                coeff_value_after - coeff_value):
                    groups[np.where(groups == i)] = group_before
                else:
                    groups[np.where(groups == i)] = i + 1
    return groups.astype(int)


def get_m_1(cut_points_estimates, cut_points, S):
    m_1, d = 0, 0
    n_features = len(cut_points)
    for j in set(range(n_features)) - set(S):
        mu_star_j = cut_points[str(j)][1:-1]
        hat_mu_star_j = cut_points_estimates[str(j)][1:-1]
        if len(hat_mu_star_j) > 0:
            d += 1
            m_1 += get_H(mu_star_j, hat_mu_star_j)
    if d == 0:
        m_1 = np.nan
    else:
        m_1 *= (1 / d)
    return m_1


def get_H(A, B):
    return max(get_E(A, B), get_E(B, A))


def get_E(A, B):
    return max([min([abs(a - b) for a in A]) for b in B])


def get_m_2(hat_K_star, S):
    return (1 / len(S)) * hat_K_star[S].sum()


def plot_screening(screening_strategy, screening_marker, cancer, P):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    alpha = .8
    lw = 2
    label = 'Selected'
    n_features = len(screening_marker)
    ax.plot(range(P), screening_marker[:P], 'r',
            lw=lw, alpha=alpha, label=label)
    label = 'Rejected'
    ax.plot(range(P, n_features), screening_marker[P:],
            'b', lw=lw, alpha=alpha, label=label)
    pl.legend(fontsize=18)
    pl.xlabel(r'$j$', fontsize=25)
    pl.tick_params(axis='x', which='both', top='off')
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.title("%s screening on %s" % (screening_strategy, cancer),
             fontsize=20)
    pl.tight_layout()
    pl.show()


def get_p_values_j(feature, mu_k, times, censoring, values_to_test, epsilon):
    if values_to_test is None:
        p1 = np.percentile(feature, epsilon)
        p2 = np.percentile(feature, 100 - epsilon)
        values_to_test = mu_k[np.where((mu_k <= p2) & (mu_k >= p1))]
    p_values, t_values = [], []
    for val in values_to_test:
        feature_bin = feature <= val
        mod = sm.PHReg(endog=times, status=censoring,
                       exog=feature_bin.astype(int), ties="efron")
        fitted_model = mod.fit()
        p_values.append(fitted_model.pvalues[0])
        t_values.append(fitted_model.tvalues[0])
    p_values = pd.DataFrame({'values_to_test': values_to_test,
                             'p_values': p_values,
                             't_values': t_values})
    p_values.sort_values('values_to_test', inplace=True)
    return p_values


def multiple_testing(X, boundaries, Y, delta, values_to_test=None,
                     features_names=None, epsilon=5):
    if values_to_test is None:
        values_to_test = X.shape[1] * [None]
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    X = np.array(X)
    result = Parallel(n_jobs=5)(
        delayed(get_p_values_j)(X[:, j],
                                boundaries[features_names[j]].copy()[1:-1], Y,
                                delta, values_to_test[j], epsilon=epsilon)
        for j in range(X.shape[1]))
    return result


def t_ij(i, j, n):
    return (1 - i * (n - j) / ((n - i) * j)) ** .5


def d_ij(i, j, z, n):
    return (2 / np.pi) ** .5 * norm.pdf(z) * (
        t_ij(i, j, n) - (z ** 2 / 4 - 1) * t_ij(i, j, n) ** 3 / 6)


def p_value_cut(p_values, values_to_test, feature, epsilon=5):
    n_tested = p_values.size
    p_value_min = np.min(p_values)
    l = np.zeros(n_tested)
    l[-1] = n_tested
    d = np.zeros(n_tested - 1)
    z = norm.ppf(1 - p_value_min / 2)
    values_to_test_sorted = np.sort(values_to_test)

    epsilon /= 100
    p_corr_1 = norm.pdf(1 - p_value_min / 2) * (z - 1 / z) * np.log(
        (1 - epsilon) ** 2 / epsilon ** 2) + 4 * norm.pdf(z) / z

    for i in np.arange(n_tested - 1):
        l[i] = np.count_nonzero(feature <= values_to_test_sorted[i])
        if i >= 1:
            d[i - 1] = d_ij(l[i - 1], l[i], z, feature.shape[0])
    p_corr_2 = p_value_min + np.sum(d)

    p_value_min_corrected = np.min((p_corr_1, p_corr_2, 1))
    if np.isnan(p_value_min_corrected) or np.isinf(p_value_min_corrected):
        p_value_min_corrected = p_value_min
    return p_value_min_corrected


def multiple_testing_perm(n_samples, X, boundaries, Y, delta, values_to_test_init,
                     features_names, epsilon):
    np.random.seed()
    perm = np.random.choice(n_samples, size=n_samples, replace=True)
    multiple_testing_rslt = multiple_testing(X[perm], boundaries, Y[perm],
                                   delta[perm], values_to_test_init,
                                   features_names=features_names,
                                   epsilon=epsilon)
    return multiple_testing_rslt


def bootstrap_cut_max_t(X, boundaries, Y, delta, multiple_testing_rslt, B=10,
                        features_names=None, epsilon=5):
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    n_samples, n_features = X.shape
    t_values_init, values_to_test_init, t_values_B = [], [], []
    for j in range(n_features):
        t_values_init.append(multiple_testing_rslt[j].t_values)
        values_to_test_j = multiple_testing_rslt[j].values_to_test
        values_to_test_init.append(values_to_test_j)
        n_tested_j = values_to_test_j.size
        t_values_B.append(pd.DataFrame(np.zeros((B, n_tested_j))))

    result = Parallel(n_jobs=10)(
        delayed(multiple_testing_perm)(n_samples, X, boundaries, Y, delta,
                                  values_to_test_init, features_names, epsilon)
        for _ in np.arange(B))

    for b in np.arange(B):
        for j in range(n_features):
            t_values_B[j].ix[b, :] = result[b][j].t_values

    adjusted_p_values = []
    for j in range(n_features):
        sd = t_values_B[j].std(0)
        sd[sd < 1] = 1
        mean = t_values_B[j].mean(0)
        t_val_B_H0_j = (t_values_B[j] - mean) / sd
        maxT_j = t_val_B_H0_j.abs().max(1)
        adjusted_p_values.append(
            [(maxT_j > np.abs(t_k)).mean() for t_k in t_values_init[j]])
    return adjusted_p_values


def refit_and_predict(cut_points_estimates, X_train, X_test, Y_train,
                      delta_train, Y_test, delta_test):

    binarizer = FeaturesBinarizer(method='given',
                                  bins_boundaries=cut_points_estimates,
                                  remove_first=True)
    binarizer.fit(pd.concat([X_train, X_test]))
    X_bin_train = binarizer.transform(X_train)
    X_bin_test = binarizer.transform(X_test)

    learner = CoxRegression(penalty='none', tol=1e-5,
                            solver='agd', verbose=False,
                            max_iter=100, step=0.3,
                            warm_start=True)
    learner._solver_obj.linesearch = False
    learner.fit(X_bin_train, Y_train, delta_train)
    coeffs = learner.coeffs
    marker = X_bin_test.dot(coeffs)
    lp_train = X_bin_train.dot(coeffs)
    c_index = concordance_index(Y_test, marker, delta_test)
    c_index = max(c_index, 1 - c_index)

    return c_index, marker, lp_train
