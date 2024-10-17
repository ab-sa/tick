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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


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
