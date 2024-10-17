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
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)

    def astype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        new_model = tick.base.dtype_to_cpp_type.copy_with(
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
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)

    def _extract_dtype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.extract_dtype(
            dtype_or_object_with_dtype)

    def astype(self, dtype_or_object_with_dtype):
        import tick.base.dtype_to_cpp_type
        new_prox = tick.base.dtype_to_cpp_type.copy_with(
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

    _solvers = {
        'gd': 'GD',
        'agd': 'AGD',
        'sgd': 'SGD',
        'svrg': 'SVRG',
        'bfgs': 'BFGS',
        'sdca': 'SDCA'
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
        from tick.solver import AGD, GD, BFGS, SGD, SVRG, SDCA
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
