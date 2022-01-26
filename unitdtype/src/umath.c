#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"


/*
 * WARNING: The following function relies on the fact that all NumPy builtins\
 *          (that we use) do provide a singleton.
 *
 * This function could possibly be cleaned up/refined a bit, but it should be
 * completely fine in practice.
 * (It assumes there is nothing non-unit specific to preserve for example)
 */
static int
strip_unit_translate_given_descrs(
        int nin, int nout, PyArray_DTypeMeta *wrapped_dtypes[],
        PyArray_Descr *given_descrs[], PyArray_Descr *new_descrs[])
{
    for (int i = 0; i < nin + nout; i++) {
        new_descrs[i] = wrapped_dtypes[i]->singleton;
    }
    return 0;
}


static int
same_unit_translate_loop_descrs(int nin, int nout,
        PyArray_DTypeMeta *new_dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *original_descrs[], PyArray_Descr *loop_descrs[])
{
    for (int i = 0; i < nin + nout; i++) {
        if (!PyArray_ISNBO(original_descrs[i]->byteorder)) {
            PyErr_SetString(PyExc_TypeError,
                    "Loop descriptor is not compatible with wrapped one!");
            return -1;
        }
    }

    // TODO: It is unclear to me if I should prefer the common dtypes with
    //       or without outputs!
    //       (That is what should be cast, inputs or outputs?)
    PyArray_Descr *result_descr = PyArray_ResultType(
            0, NULL, nin, given_descrs);
    if (result_descr == NULL) {
        return -1;
    }
    /*
     * We used the normal ResultType function, but the input is known to be
     * one of ours, a UnitDtype.
     */
    assert(Py_TYPE(result_descr) == &UnitDType_Type);
    for (int i = 0; i < nin + nout; i++) {
        Py_INCREF(result_descr);
        loop_descrs[i] = result_descr;
    }
    Py_DECREF(result_descr);
    return 0;
}


static int
same_unit_unmodified_out_translate_loop_descrs(int nin, int nout,
        PyArray_DTypeMeta *new_dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *original_descrs[], PyArray_Descr *loop_descrs[])
{
    for (int i = 0; i < nin + nout; i++) {
        if (!PyArray_ISNBO(original_descrs[i]->byteorder)) {
            PyErr_SetString(PyExc_TypeError,
                    "Loop descriptor is not compatible with wrapped one!");
            return -1;
        }
    }

    // TODO: It is unclear to me if I should prefer the common dtypes with
    //       or without outputs!
    //       (That is what should be cast, inputs or outputs?)
    PyArray_Descr *result_descr = PyArray_ResultType(
            0, NULL, nin, given_descrs);
    if (result_descr == NULL) {
        return -1;
    }
    /*
     * We used the normal ResultType function, but the input is known to be
     * one of ours, a UnitDtype.
     */
    assert(Py_TYPE(result_descr) == &UnitDType_Type);
    for (int i = 0; i < nin; i++) {
        Py_INCREF(result_descr);
        loop_descrs[i] = result_descr;
    }
    Py_DECREF(result_descr);

    /* Keep the outputs unmodified (e.g. for comparisons) */
    for (int i = nin; i < nin + nout; i++) {
        Py_INCREF(original_descrs[i]);
        loop_descrs[i] = original_descrs[i];
    }

    return 0;
}


static int
multiply_translate_loop_descrs(int nin, int nout,
        PyArray_DTypeMeta *new_dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *original_descrs[], PyArray_Descr *loop_descrs[])
{
    for (int i = 0; i < nin + nout; i++) {
        if (!PyArray_ISNBO(original_descrs[i]->byteorder)) {
            PyErr_SetString(PyExc_TypeError,
                    "Loop descriptor is not compatible with wrapped one!");
            return -1;
        }
    }

    PyObject *unit1 = ((UnitDTypeObject *)given_descrs[0])->unit;
    PyObject *unit2 = ((UnitDTypeObject *)given_descrs[0])->unit;

    PyObject *new_unit = PyNumber_Multiply(unit1, unit2);
    if (new_unit == NULL) {
        return -1;
    }

    loop_descrs[2] = (PyArray_Descr *)new_unitdtype_instance(new_unit);
    Py_DECREF(new_unit);
    if (loop_descrs[2] == NULL) {
        return -1;
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return 0;
}

static int
division_translate_loop_descrs(int nin, int nout,
        PyArray_DTypeMeta *new_dtypes[], PyArray_Descr *given_descrs[],
        PyArray_Descr *original_descrs[], PyArray_Descr *loop_descrs[])
{
    for (int i = 0; i < nin + nout; i++) {
        if (!PyArray_ISNBO(original_descrs[i]->byteorder)) {
            PyErr_SetString(PyExc_TypeError,
                    "Loop descriptor is not compatible with wrapped one!");
            return -1;
        }
    }

    PyObject *unit1 = ((UnitDTypeObject *)given_descrs[0])->unit;
    PyObject *unit2 = ((UnitDTypeObject *)given_descrs[0])->unit;

    PyObject *new_unit = PyNumber_TrueDivide(unit1, unit2);
    if (new_unit == NULL) {
        return -1;
    }

    loop_descrs[2] = (PyArray_Descr *)new_unitdtype_instance(new_unit);
    Py_DECREF(new_unit);
    if (loop_descrs[2] == NULL) {
        return -1;
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return 0;
}



/*
 * Define a lists of to be wrapped ufuncs, read and used at import time
 * in `init_wrapped_ufuncs`.
 */
typedef struct {
    char *name;
    translate_loop_descrs_func *translate_loop_descrs;
} ufunc_info_struct;


static ufunc_info_struct homogeneous_ufuncs[] = {
    /* Unary ufuncs */
    {"negative", same_unit_translate_loop_descrs},
    {"positive", same_unit_translate_loop_descrs},
    /* binary ones */
    {"add",      same_unit_translate_loop_descrs},
    {"subtract", same_unit_translate_loop_descrs},
    {"fmod",     same_unit_translate_loop_descrs},
    {"hypot",    same_unit_translate_loop_descrs},
    {"maximum",  same_unit_translate_loop_descrs},
    {"minimum",  same_unit_translate_loop_descrs},
    {"fmax",     same_unit_translate_loop_descrs},
    {"fmin",     same_unit_translate_loop_descrs},
    /* Common, special ones: */
    {"multiply", multiply_translate_loop_descrs},
    {"divide",   division_translate_loop_descrs},
    {NULL, NULL}
};


static ufunc_info_struct binary_boolean_output_ufuncs[] = {
    /* Comparisons */
    {"equal", same_unit_unmodified_out_translate_loop_descrs},
    {"not_equal", same_unit_unmodified_out_translate_loop_descrs},
    {"greater", same_unit_unmodified_out_translate_loop_descrs},
    {"greater_equal", same_unit_unmodified_out_translate_loop_descrs},
    {"less", same_unit_unmodified_out_translate_loop_descrs},
    {"less_equal", same_unit_unmodified_out_translate_loop_descrs},
    {NULL, NULL}
};

static ufunc_info_struct unary_boolean_output_ufuncs[] = {
    {"isfinite", same_unit_unmodified_out_translate_loop_descrs},
    {"isnan", same_unit_unmodified_out_translate_loop_descrs},
    // not sure about signbit, if an offset is involved?!
    {NULL, NULL}
};


int
add_wrapping_loops(
        ufunc_info_struct *ufunc_infos,
        PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[])
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }

    ufunc_info_struct *ufunc_info = ufunc_infos;
    while ((*ufunc_info).name != NULL) {
        PyObject *ufunc = PyObject_GetAttrString(numpy, (*ufunc_info).name);
        if (ufunc == NULL) {
            Py_DECREF(numpy);
            return -1;
        }

        if (PyUFunc_AddWrappingLoop(
                ufunc, new_dtypes, wrapped_dtypes,
                &strip_unit_translate_given_descrs,
                (*ufunc_info).translate_loop_descrs) < 0) {
            Py_DECREF(numpy);
            Py_DECREF(ufunc);
            return -1;
        }
        Py_DECREF(ufunc);
        ufunc_info++;
    }
    Py_DECREF(numpy);
    return 0;
}


int
init_wrapped_ufuncs(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }

    /* 3 is enough for all current ufuncs, i.e. binary ones */
    PyArray_DTypeMeta *new_dtypes[] = {
            &UnitDType_Type, &UnitDType_Type, &UnitDType_Type};
    PyArray_DTypeMeta *wrapped_dtypes[] = {
            PyArray_DoubleDType, PyArray_DoubleDType, PyArray_DoubleDType};

    if (add_wrapping_loops(
            homogeneous_ufuncs, new_dtypes, wrapped_dtypes) < 0) {
        return -1;
    }

    new_dtypes[2] = PyArray_BoolDType;
    wrapped_dtypes[2] = PyArray_BoolDType;
    if (add_wrapping_loops(
            binary_boolean_output_ufuncs, new_dtypes, wrapped_dtypes) < 0) {
        return -1;
    }

    new_dtypes[1] = PyArray_BoolDType;
    wrapped_dtypes[1] = PyArray_BoolDType;
    if (add_wrapping_loops(
            unary_boolean_output_ufuncs, new_dtypes, wrapped_dtypes) < 0) {
        return -1;
    }

    return 0;
}
