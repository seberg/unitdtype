#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"


static int
strip_unit_translate_given_descrs(
        int nin, int nout, PyArray_DTypeMeta *wrapped_dtypes[],
        PyArray_Descr *given_descrs[], PyArray_Descr *new_descrs[])
{
    /*
     * Anything we wrap, we can always fill in the double singleton
     * (even if unspecified).  We do not even support e.g. byte-swapping!
     */
    for (int i = 0; i < nin + nout; i++) {
        new_descrs[i] = PyArray_DoubleDType->singleton;
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




static char *simple_ufuncs[] = {
    /* unary ones: */
    "negative", "positive",
    /* binary ones: */
    "add", "subtract", "fmod", "hypot", "maximum", "minimum", "fmax", "fmin",
    /* Potentially interesting: nextafter, heaviside */
    NULL
};


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

    char **ufunc_name = simple_ufuncs;
    while (*ufunc_name != NULL) {
        PyObject *ufunc = PyObject_GetAttrString(numpy, *ufunc_name);
        if (ufunc == NULL) {
            Py_DECREF(numpy);
            return -1;
        }

        if (PyUFunc_AddWrappingLoop(
                ufunc, new_dtypes, wrapped_dtypes,
                &strip_unit_translate_given_descrs,
                &same_unit_translate_loop_descrs) < 0) {
            Py_DECREF(numpy);
            Py_DECREF(ufunc);
            return -1;
        }
        Py_DECREF(ufunc);
        ufunc_name++;
    }

    PyObject *multiply = PyObject_GetAttrString(numpy, "multiply");
    if (multiply == NULL) {
        Py_DECREF(numpy);
        return -1;
    }
    if (PyUFunc_AddWrappingLoop(
            multiply, new_dtypes, wrapped_dtypes,
            &strip_unit_translate_given_descrs,
            &multiply_translate_loop_descrs) < 0) {
        Py_DECREF(numpy);
        Py_DECREF(multiply);
        return -1;
    }

    PyObject *divide = PyObject_GetAttrString(numpy, "divide");
    if (divide == NULL) {
        Py_DECREF(numpy);
        return -1;
    }
    if (PyUFunc_AddWrappingLoop(
            divide, new_dtypes, wrapped_dtypes,
            &strip_unit_translate_given_descrs,
            &division_translate_loop_descrs) < 0) {
        Py_DECREF(numpy);
        Py_DECREF(multiply);
        return -1;
    }

    Py_DECREF(numpy);
    Py_DECREF(multiply);
    return 0;
}